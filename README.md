斯坦福计算机视觉基础课程cs231n

# 学习安排

## week1
- [python_numpy](https://github.com/lizhe960118/cs231n/blob/master/python_numpy/python_and_numpy.ipynb) python和numpy的基础使用
- [KNN_classifier.py](https://github.com/lizhe960118/cs231n/blob/master/assignment1/cs231n/classifiers/KNN_classifier.py) KKN的实现
- [data_utils.py](https://github.com/lizhe960118/cs231n/blob/master/assignment1/cs231n/data_utils.py) CIFA10数据集处理
- [KNN_classifier.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment1/KNN_classifier.ipynb) KNN的使用

## week2
- [backpropagation]() 反向传播
- [gradient_check.py](https://github.com/lizhe960118/cs231n/blob/master/assignment1/cs231n/gradient_check.py) 梯度对比
- [linear_svm.py](https://github.com/lizhe960118/cs231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py) svm模型的实现
- [SVM.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment1/LinearSVM_classifier.ipynb) svm的使用

## week3
- [softmax_loss]() softmax的损失函数
- [softmax_grad](https://juejin.im/post/5b3cd0516fb9a04fb21288df) softmax的求导
- [linear_softmax.py](https://github.com/lizhe960118/cs231n/blob/master/assignment1/cs231n/classifiers/linear_softmax.py) softmax的实现
- [Softmax.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment1/LinearSoftmax_classifier.ipynb) softmax的使用

## week4
- [激活函数](https://zhuanlan.zhihu.com/p/21462488?refer=intelligentunit) 常见的激活函数
- [neural_network.py](https://github.com/lizhe960118/cs231n/blob/master/assignment1/cs231n/classifiers/Neural_network.py) 两层神经网络的实现
- [Two_Layer_Network.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment1/Two_layer_network.ipynb) 两层神经网络的使用

## week5
- [optimizer.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/optimizer.py) 梯度更新算法
- [PCA_white.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment2/PCA_white.ipynb) 数据预处理
- [layer_basic.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/layer_basic.py) 批处理归一化,dropout,relu,affine的基础结构前向和反向的实现
- [layer_utils.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/layer_utils.py) 多层网络的模块化
- [fc_net.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/classifiers/fc_net.py) 全连接网络的搭建
- [solver.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/solver.py) 搭建一个整体的训练框架
- [Fully_Connected_Nets.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment2/Fully_Connected_Nets.ipynb) 全连接网络的使用
- [Batch_Normalization.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment2/Batch_Normalization.ipynb) 批处理归一化的使用
- [Dropout.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment2/Dropout.ipynb) dropout的使用

## week6
- [layer_basic.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/layer_basic.py) 卷积层、池化层前向和反向的实现
- [layer_utils.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/layer_utils.py) 多层模块的实现
- [im2col.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/im2col.py) 把二维图像转为一维向量进行计算
- [cnn.py](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cs231n/classifiers/cnn.py) 全连接网络的搭建
- [ConvolutionalNetworks.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment2/ConvolutionalNetworks.ipynb) 卷积网络的使用

## week7
- [cifar10_tensorflow.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cifar10-tensorflow.ipynb) 使用tensorflow完成CIFA10分类
- [cifar10_Pytorch.ipynb](https://github.com/lizhe960118/cs231n/blob/master/assignment2/cifar10-pytorch.ipynb) 使用pytorch完成CIFA10分类

## week8
- [rnn_layers.py]()  VanillaRNN,word_embeddding, LSTM前向反向的实现
- [coco_utils.py]() coco数据集的处理
- [pretrained_cnn.py]() 预训练的cnn的实现
- [rnn.py]() 搭建RNN模型
- [captioning_solver.py]() 搭建一个整体的训练框架
- [RNN_Captioning.ipynb]() 使用普通RNN进行图像标注
- [LSTM_Captioning.ipynb]() 实现LSTM，并应用于在微软COCO数据集上进行图像标注。

## week9
- [ImageGradients.ipynb]() 使用TinyImageNet数据集计算梯度
- [ImageGeneration.ipynb]() 使用一个训练好的TinyImageNet模型来生成图像
