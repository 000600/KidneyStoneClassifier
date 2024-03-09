# Kidney Stone Classifier

## The Neural Network
This neural network (found in the **kidney_stone_classifier.py** file) predicts whether or not a patient has a kidney stone based on urine analysis that takes into account multiple different factors. The model will predict a value close to 0 if the patient is predicted to not have a kidney stone and a 1 if the patient is predicted to have a kidney stone. Since the model only predicts binary categorical values, the model uses a binary cross entropy loss function and has 1 output neuron. The model uses a standard Adam optimizer with a learning rate of 0.001 in addition to multiple dropout layers and early stopping to prevent overfitting. The model has an architecture consisting of:
- 1 Batch Normalization layer
- 1 Input layer (with 7 input neurons and a ReLU activation function)
- 1 Hidden layer (with 256 neurons and a ReLU activation function)
- 1 Dropout layer (with a dropout rate of 0.3)
- 1 Output layer (with 1 output neuron and a sigmoid activation function)

Feel free to further tune the hyperparameters or build upon the model!

## The XGB Classifier
An XGBoost Classifier model is also included in the **kidney_stone_xgb.py** file file to compare the neural networks to the classifier. The XGBoost classifier has 100 estimators, a learning rate of 0.001, and early stopping based on validation sets. The classifier predicts the likelihood someone has a kidney stone based on the same inputs as the model in the **kidney_stone_classifier.py** file. Although the number of estimators is lower than usual, I found that it achieved similar results.

As with the neural network, feel free to tune the hyperparameters or build upon the classifier!

## The Dataset
The dataset can be found at this link: https://www.kaggle.com/datasets/vuppalaadithyasairam/kidney-stone-prediction-based-on-urine-analysis. Credit for the dataset collection goes to **Atilla Özkaymak**, **Philippe Billet**, **Vuppala Adithya Sairam**, and others on *Kaggle*. It describes whether or not a person has a kidney stone (encoded as 0 or 1) based on urine analysis that includes multiple factors, such as:
- Calcium concentration in urine
- PH of urine
- Osmolarity of urine
- Conductivity of urine

Note that the initial dataset is biased (this statistic can be found on the data's webpage); it contains a higher representation of non-kidney stone cases (encoded as 0's in this model) than kidney stone cases (encoded as 1's in this model). This issue is addressed within the classifier file using Imbalanced-Learn's **SMOTE()**, which oversamples the minority class within the dataset.

## Libraries
These neural networks and XGBoost Regressor were created with the help of the Tensorflow, Scikit-Learn, and XGBoost libraries.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
- Scikit-Learn's Website: https://scikit-learn.org/stable/
- Scikit-Learn's Installation Instructions: https://scikit-learn.org/stable/install.html
- XGBoost's Website: https://xgboost.readthedocs.io/en/stable/#
- XGBoost's Installation Instructions: https://xgboost.readthedocs.io/en/stable/install.html
- Imbalanced-Learn's Website: https://imbalanced-learn.org/stable/about.html
- Imbalanced-Learn's Installation Instructions: https://pypi.org/project/imbalanced-learn/

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
