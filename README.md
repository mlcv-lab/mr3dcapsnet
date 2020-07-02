# End-to-End Multi-Resolution 3D Capsule Network for People Action Detection

A multi-resolution 3D capsule network system to process input videos of action scenes with multiple subjects and use a specialized 3DCNN based module to classify the actions performed by the subjects. The segmented video chunks thanks to a Faster-RCNN detector are then passed on to an action recognition module to detect the actions performed by the subjects. Our approach combines multiple paths 3DCNN on different input data resolutions as input features to a specifically built 3D capsule network for action detection. We follow the implementation of the 2D capsule network for the MNIST dataset, but we replace the 2D convolutions with 3D convolutions so the capsule can learn spatio-temporal features. Unlike the original capsule network implementation of for 2D dataset, we do not use any strides in the convolution layer but rather use 3D max-pooling on the produced feature maps. We find that even though simpler than capsule pooling through voting inside dynamic routing mechanism \cite{Duarte2018}, our 3D max-pooling can also helps improve the performance and reduce the computation. We also find that avoiding a fully connected network on the decoder part of the capsule network can significantly reduce the number of parameters involved. 


![](mr3dcapsnet-demo.gif)

