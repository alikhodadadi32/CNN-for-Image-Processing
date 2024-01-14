# A Combination of ResNet, DenseNet, and GoogleNet for Image Processing

## Model description 

This project introduces a novel network structure that combines elements of the ResNet, GoogleNet, and DenseNet architectures. It aims to leverage the unique strengths of these networks to improve accuracy and efficiency.

Recent advancements in convolutional neural networks (CNNs) have led to a variety of structures and parameterizations, each seeking improvements in accuracy and efficiency. ResNet utilizes residual connections for deeper networks without accuracy loss. DenseNet strengthens feature propagation through feature reuse. This project proposes a network that integrates key characteristics of ResNet, GoogleNet, and DenseNet, adjusting and applying these features to enhance performance.

he proposed network is composed of three parts, each corresponding to elements from ResNet, GoogleNet, and DenseNet. The network architucture 

ResNet: The foundation is based on Residual blocks, but with modified Inception modules for each network level.
GoogleNet: Inception modules are simplified versions of Inception-V4 for computational efficiency.
DenseNet: Features highly connected layers with limited connections to levels, and uses concatenated feature maps from the inputs of the last level.

The following adjustments were included to the proposed model for better results:
Layer Adjustments: Adjustments to the filter sizes and configurations were made to align with the ResNet's bottleneck building block logic.
Skip Connections: Traditional skip connections between levels are replaced with convolution layers for better accuracy.
Projection-Dense: A new projection method for level-wise feature map adjustment.


## Code Structure

### main.py:
Includes the code that loads the dataset and performs the training, testing and
prediction.

### DataLoader.py:
Includes the code that defines functions related to data I/O.

### ImageUtils.py:
Includes the code that defines functions for any (pre-)processing of the
images.

### Model.py: 
Includes the code that defines the your model in a class. The class is initialized
with the configuration dictionaries and should have at least the methods “train(X, Y, configs,
[X_valid, Y_valid,])”, “evaluate(X, Y)”, “predict_prob(X)”. The defined model class is
imported to and referenced in main.py.

### Network.py: 
Includes the code that defines the network architecture. The defined network
will be imported and referenced in Model.py.


## How to run

To reproduce the validation results and test probabilities, following lines of codes needs to be run in the CMD. 

To reproduce validation accuracies (public test data): "python main.py --List_residual_layers 5 5 5  --mode test"
To reproduce test probabilities (private test data)  : "python main.py --List_residual_layers 5 5 5  --mode predict"


*   The List_residual_layers must be specidfied as shown
**  The Final_model folder and the ".py" files in the "code" folder must be in the same directory to run (you may need to copy the "Final_model" folder into the "code" folder)


$ python main.py --List_residual_layers 5 5 5  --mode test
2024-01-13 01:31:54.575670: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2024-01-13 01:31:54.576024: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.