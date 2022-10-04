# Urban_sound_classifier
Classification of street sounds using DNN and CNN.

The labeled data is obtained from https://urbansounddataset.weebly.com/urbansound8k.html

I extracted features by using the Mel-Frequency Cepstral Coefficients (MFCC) and used two architectures.


## Model architecture:

1. Deep Neural Network (DNN)
    * Input layer: 40 input nodes (one per each MFCC  coefficient, i.e. column , i.e X) -> 40 = np.size(X[1000]) = np.size(featuresdf.iloc[1000][0])
    * 2 Hidden layers: 256 nodes each with a ReLU activation function. 50% Dropout value.
    * Output layer: 10 nodes , one per each audio label (possible classification) with Softmax activation function (~probabilities [0,1])                 
    
    
2. Convolutional Neural Network (CNN)    
    * 4 Conv2D layers. Each layer will increase in size (i.e. Nb output filters) from 16, 32, 64 to 128. 
                    The kernel size is 2x2
    * 1 final dense layer with 10 output nodes (one for each possible sound classification) 



### Notes:
Dropout value of 50%. This will randomly exclude nodes from each update cycle 
which in turn results in a network that is capable of better generalisation 
and is less likely to overfit the training data.

Softmax makes the output sum up to 1 so the output can be interpreted as probabilities.
The model will then make its prediction based on which option has the highest probability.



