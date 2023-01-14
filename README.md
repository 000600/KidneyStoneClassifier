# Kidney Stone Classifier

## The Neural Network
This neural network (found in the **.py** file) predicts whether or not a patient has a kidney stone based on urine analysis that takes into account multiple different factors. The model will predict a value close to 0 if the patient is predicted to not have a kidney stone and a 1 if the patient is predicted to have a kidney stone. Since the model only predicts binary categorical values, the model uses a binary crossentropy loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001 and multiple dropout layers to prevent overfitting. The model has an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 7 input neurons and a ReLU activation function)
- 1 Hidden layer (with 256 neurons and a ReLU activation function)
- 1 Dropout layers (with a dropout rate of 0.3)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The XGB Regressor
