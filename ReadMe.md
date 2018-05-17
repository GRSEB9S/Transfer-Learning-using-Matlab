# Transfer Learning :

Transfer learning is commonly used in deep learning applications. You can 
take a pretrained network and use it as a starting point to learn a new task. 
Fine-tuning a network with transfer learning is usually much faster and easier 
than training a network from scratch with randomly initialized weights. You 
can quickly transfer learned features to a new task using a smaller number of 
training images.


## Some of existing neural networks for image classification for 

**-> AlexNet**

**-> Vgg16**

**-> Vgg19**

**-> ResNet-50**

**-> ResNet-101**

**-> GoogleNet**

**-> Inception-v3**

**-> Inception-ResNet-v2**

## In this implementation, two major steps are present: 

**Step 1:** The last three layers : "Fully-Connected-Layer", "SoftMax" and "Classification Predictions" are removed.
**Step 2:** New three layers : "Fully-Connected-Layer", "SoftMax" and "Classification Predictions" 
		are added but based on number of classes in our dataset.
**Step 3:** Connect Original Network's "Pooling-Layer" to newly created layers in Step 2

The difference in the using different neural network implementation (as given above) is defining the neural network model and 
the identification of these layers and replacing them.

## References:

https://www.mathworks.com/help/nnet/ug/pretrained-convolutional-neural-networks.html
