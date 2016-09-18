# Robust Optimization for Deep Regression

This is a [MatConvNet](http://www.vlfeat.org/matconvnet) implementation of the work:

Robust Optimization for Deep Regression
Vasileios Belagiannis, Christian Rupprecht, Gustavo Carneiro, and Nassir Navab, International Conference on Computer Vision (ICCV), Santiago, Chile, December 2015. [Project](http://campar.in.tum.de/twiki/pub/Chair/DeepReg/deepreg.html), [Arxiv](http://arxiv.org/abs/1505.06606).

There is an example of training a model using the robust loss. To train the example model, one have to:

1. Download the LSP dataset (*fetchNprepareLSP.m* in the *data* folder).
2. Run the script *train_LSP_Model.m* that is place in the *keypoint-regressor* folder.
3. To Evaluate a trained model, check the script *cnn_eval.m* in the *keypoint-regressor*.

There is also a trained model *net-epoch-40.mat* in the *data* folder.

[MatConvNet](http://www.vlfeat.org/matconvnet) beta-21 is included into the project.

**Lisence**: This code and MatConvNet are made available under the terms of the BSD license (see the COPYING file).
