import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import cv2 as cv
import time
from core import build_depth_resnet, build_lensblur, build_sr
import re

# Restore ete model
def restore_ete_model(session, variables, eval_config):
    # Get parameters
    saver = tf.compat.v1.train.Saver(variables)
    saver.restore(sess=session, save_path=eval_config.ete_ckpt)

# Calculate depth map and render lens blur
def restore_piecewise_model(session, pre_trained_params, eval_config):
    var_depth = dict([('Network' + re.split('Depth', param.op.name)[-1], param) for param in pre_trained_params if re.search('Depth', param.op.name)])
    var_lensblur = dict([(param.op.name, param) for param in pre_trained_params if re.search('Lensblur', param.op.name) or re.search('Feature', param.op.name) or re.search('SR', param.op.name)])
    
    # Restore depth and lensblur
    saver_depth = tf.compat.v1.train.Saver(var_depth)
    saver_lensblur = tf.compat.v1.train.Saver(var_lensblur)
    saver_depth.restore(session, eval_config.depth_ckpt)
    saver_lensblur.restore(session, eval_config.lensblur_ckpt)

# Construct graph
def construct_graph(eval_config):
    # Get input
    img_320_input = tf.compat.v1.placeholder(dtype=tf.float32, name='image_320', shape=[None, None, 3])
    img_640_input = tf.compat.v1.placeholder(dtype=tf.float32, name='image_640', shape=[None, None, 3])
    img_1280_input = tf.compat.v1.placeholder(dtype=tf.float32, name='image_1280', shape=[None, None, 3])
    img_320 = tf.expand_dims(img_320_input, 0)
    img_640 = tf.expand_dims(img_640_input, 0)
    img_1280 = tf.expand_dims(img_1280_input, 0)
    # Get apeture, focal points
    ape = tf.compat.v1.placeholder(tf.float32, name='aperture', shape=[])
    f_x = tf.compat.v1.placeholder(tf.int32, name='focal_x', shape=[])
    f_y = tf.compat.v1.placeholder(tf.int32, name='focal_y', shape=[])
    is_training = tf.constant(False, dtype=tf.bool, shape=[])

    with tf.compat.v1.variable_scope('Network'):
        # build depth map with image of 320 size
        depth_320, depth_net = build_depth_resnet.build(img_320, is_training)
        f_depth = depth_320[0, f_y, f_x, 0]
        depth_320_signed = (depth_320 - f_depth) * ape

        # Get dofs
        pre_dof_320, lensblur_net, feature_net = build_lensblur.build(img_320, depth_320_signed, is_training)
        pre_dof_640, sr_net = build_sr.build(img_320, depth_320_signed, pre_dof_320, img_640, is_training)
        shape_640 = tf.shape(input=pre_dof_640)
        depth_640_s = tf.image.resize(depth_320_signed, [shape_640[1], shape_640[2]], method=tf.image.ResizeMethod.BILINEAR)
        pre_dof_1280, sr_net = build_sr.build(img_640, depth_640_s, pre_dof_640, img_1280, is_training)

    # Modification
    variables = tf.compat.v1.global_variables()
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.compat.v1.Session(config=session_config)

    # Restore parameters of network according to the config
    if eval_config.use_moving_average:
        m = tf.train.ExponentialMovingAverage(0.9)
        variables = m.variables_to_restore()

    # Restore piecewise or ete model according to the config
    if eval_config.HasField('depth_ckpt') and eval_config.HasField('lensblur_ckpt'):
        restore_piecewise_model(session, variables, eval_config)
    elif eval_config.HasField('ete_ckpt'):
        restore_ete_model(session, variables, eval_config)

    # Predict DoF
    def pre_dof(img):
        # Get the focal point and aperture
        focal_x, focal_y = img.get_focal_point()
        aperture_size = img.aperture_size
        # According to different scale
        if img.scale == 4:
            return session.run(pre_dof_1280,
                            feed_dict={img_320_input: img.img_320, img_640_input: img.img_640,
                                       img_1280_input: img.img_1280, f_x: focal_x,
                                       f_y: focal_y, ape: aperture_size})
        elif img.scale == 2:
            return session.run(pre_dof_640,
                            feed_dict={img_320_input: img.img_320, img_640_input: img.img_640,
                                       f_x: focal_x, f_y: focal_y, ape: aperture_size})
        elif img.scale == 1:
            return session.run(pre_dof_320, feed_dict={img_320_input: img.img_320, f_x: focal_x,
                                                    f_y: focal_y, ape: aperture_size})

    # Predict depth
    def pre_depth(im_320):
        return session.run(depth_320, feed_dict={img_320_input: im_320})

    return pre_dof, pre_depth

# Evaluate process
def evaluate(eval_config):
    c_dof, c_depth = construct_graph(eval_config)

    # Read in image path
    image_path = eval_config.image_path
    print(image_path)

    # Change this to display differnt images
    idx = 5
    image_names = [image_path + _ for _ in os.listdir(image_path)]

    # Read image
    image = cv.imread(image_names[idx])

    # Preprocess the image
    image = ImageWrapper(image.astype(np.float32) / 255)

    # Calculate depth
    depth = c_depth(image.img_320)
    image.depth = depth[0, :, :, 0]

    # Display original image
    plt.figure(1)
    img_h = plt.imshow(image.img[:, :, -1::-1])

    # Display apeture radius
    plt.figure(2, figsize=[5, 1])
    axis_aperture = plt.axes([0.25, 0.5, 0.65, 0.3], facecolor='lightgoldenrodyellow')
    slider = Slider(axis_aperture, 'Aperture Radius', 0.0, 10.0, valinit=5.0)
    image.aperture_size = slider.val / 10.0

    # Display depth map
    plt.figure(3)
    depth_handle = plt.imshow(depth[0, :, :, 0])
    RenderDoF(img_h, depth_handle, slider, image, c_dof, c_depth, idx, image_names)
    plt.show()

# Define wrapped image
class ImageWrapper(object):
    def __init__(self, img, depth=None, dof=None, x=0, y=0, aperture_size=1.0):
        # Resize the image
        if np.max(img.shape) > 1280:
            img_scale = 1280.0 / np.max(img.shape)
            new_shape = np.int32(img_scale * np.array(img.shape))
            img = cv.resize(img, (new_shape[1], new_shape[0]), interpolation=cv.INTER_AREA)
        self.img = img

        # If shape greater than 1280, split image into 3 kind of shapes
        if np.max(img.shape) >= 1280:
            self.scale = 4
            self.img_1280 = img
            self.img_640 = cv.resize(self.img_1280, (int(self.img_1280.shape[1] / 2), int(self.img_1280.shape[0] / 2)),
                                       interpolation=cv.INTER_AREA)
            self.img_320 = cv.resize(self.img_640, (int(self.img_640.shape[1] / 2), int(self.img_640.shape[0] / 2)),
                                       interpolation=cv.INTER_AREA)

        # If shape greater than 640, split image into 2 kind of shapes
        elif np.max(img.shape) >= 640:
            self.scale = 2
            self.img_1280 = None
            self.img_640 = img
            self.img_320 = cv.resize(self.img_640, (int(self.img_640.shape[1] / 2), int(self.img_640.shape[0] / 2)),
                                       interpolation=cv.INTER_AREA)

        # If shape greater than 320, do not change the image
        else:
            self.scale = 1
            self.img_1280 = None
            self.img_640 = None
            self.img_320 = img

        # Param setting
        self.x = x
        self.y = y
        self.depth = depth
        self.dof = dof
        self.aperture_size = aperture_size

    # Set focal point
    def set_focal_point(self, x, y):
        self.x = np.int32(x)
        self.y = np.int32(y)

    # Get the coordinate of existing focal point
    def get_focal_point(self):
        return self.x / self.scale, self.y / self.scale

    # Get the focal depth
    def get_focal_depth(self):
        x, y = self.get_focal_point()
        return self.depth[y, x]

    # Set DoF
    def set_dof(self, dof):
        self.dof = cv.resize(dof, (self.img_320.shape[1], self.img_320.shape[0]), interpolation=cv.INTER_AREA)

# Rendering the depth of field
class RenderDoF(object):
    def __init__(self, img_h, depth_handle, slider, image, calculate_dof, calculate_depth, idx, img_names):
        self.img_h = img_h
        self.depth_h = depth_handle
        self.img = image
        self.calculate_dof = calculate_dof
        self.calculate_depth = calculate_depth
        self.aperture_size = 1.0
        self.id = idx
        self.img_names = img_names
        self.cid_click = img_h.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = img_h.figure.canvas.mpl_connect('key_press_event', self.on_key)
        self.slider = slider

        # Updata aperture radius
        def update_slider(value):
            self.img.aperture_size = slider.val / 10.0

        # Apply change
        slider.on_changed(update_slider)

    # Define the event on clicking
    def on_click(self, event):
        x, y = event.xdata, event.ydata
        # If clicking
        if event.button == 1:

            # If clicking in the middle
            if x is not None and y is not None:
                print('X_coordinate: %f, Y_coordinate: %f \n' % (x, y))
                self.img.set_focal_point(x, y)
                self.render()

    # Render
    def render(self):

        # Record starting time
        start_time = time.time()

        # Get DoF
        DoF = self.calculate_dof(self.img)[0]

        # Rescale Dof into [0, 1]
        DoF = np.clip(DoF, 0, 1)

        # Stop recording time
        end_time = time.time()
        print('Spend time: %fs' % (end_time - start_time))
        try:
            self.scat_handle.remove()
        except:
            pass

        # Show the DoF image
        self.img_h.figure.clear()
        plt.figure(1)
        # self.im.set_dof(dof)
        self.img_h = plt.imshow(DoF[:, :, -1::-1])
        # self.im_handle.set_data(im_640[:,:,-1::-1])
        # self.im_handle.set_data(blured_im)
        self.img_h.figure.canvas.draw()

    def on_key(self, event):

        # turn back
        if event.key == 'j':
            self.id -= 2

        # next image
        self.id += 1
        try:
            # read new image
            img = cv.imread(self.img_names[self.id])

            # reformat the image
            img = img.astype(np.float32) / 255

            # turn into constructed type
            img = ImageWrapper(img, aperture_size=self.slider.val / 10.0)
            self.img = img

            # print out image number
            print(self.img_names[self.id])
        except:
            start_id = self.id
            exit()

        # calculate depth image
        depth_img = self.calculate_depth(img.img_320)
        self.img.depth = depth_img[0, :, :, 0]
        try:
            self.scat_handle.remove()
        except:
            pass

        # display
        self.img_h.figure.clear()
        plt.figure(1)
        self.img_h = plt.imshow(img.img[:, :, -1::-1])
        self.depth_h.figure.clear()
        plt.figure(3)
        self.depth_h = plt.imshow(depth_img[0, :, :, 0])
        self.img_h.figure.canvas.draw()
        self.depth_h.figure.canvas.draw()
        print('Processing Img: %d/%d' % (self.id, len(self.img_names)))
