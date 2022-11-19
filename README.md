# DeepLens: Shallow Depth Of Field From A Single Image
In this work, we implemented a novel neural network which consists of 3 modules: depth prediction module, lens blur module, and guided upsampling module based on the work as shown in Paper.

This project is for ENGN8501 in ANU.

## Paper
[DeepLens: Shallow Depth Of Field From A Single Image](https://arxiv.org/abs/1810.08100) <br>

## Enviornment
Main enviornment: Tensorflow v2.6.0<br>

## File instruction
### network.py
Write some functions, build the base layer, build the structure of network for inherition.

### resnet50.py
Build structure of the first 14 layers of resnet50

### kernel_net.py
Network structure of kernel net.

### feature_net.py
Network structure of feature net.

### srnet.py
Network structure of srnet.

### build_depth_resnet
Build of pretrained resnet.

### build_lensblur.py
Build of lensblur module.

### build_sr.py
Build of sr net.

### evaluator_interactive.py
Evaluation of interactive function

### eval.py
Execute file.

### Author

Yiding Qiu<br>
Haixu Liu<br>
Wenjia Cheng<br>


