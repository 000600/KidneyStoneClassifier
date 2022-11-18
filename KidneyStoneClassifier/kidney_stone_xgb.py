# Imports
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('kidney_analysis.csv')
df = pd.DataFrame(df)

# Assign x and y values
y = df['target']
x = df.drop('target', axis = 1)

# Balance dataset (make sure there are an even representation of instances with label 0 and label 1)
smote = SMOTE()
x, y = smote.fit_resample(x, y)

# Divide the x and y values into two sets: train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1, stratify = y)

# Create model
model = XGBClassifier(n_estimators = 100)

# Train model
model.fit(x_train, y_train, eval_set = [(x_test, y_test)], verbose = 1)

# Add test and train accuracies to appropriate lists
test_acc = model.score(x_test, y_test)
train_acc = model.score(x_train, y_train)

# View metrics
print(f"\nTest Accuracy: {test_acc * 100}%")
print(f"\nTrain Accuracy: {train_acc * 100}%")
