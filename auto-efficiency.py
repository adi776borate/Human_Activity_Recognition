import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk value
data.drop('car name', axis=1, inplace=True)

data['horsepower'].replace('?', 0, inplace=True)
data['horsepower'] = pd.to_numeric(data['horsepower'])

y = data['mpg']
X = data.drop('mpg', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare the performance of your model with the decision tree module from scikit learn
sklearn_tree = DecisionTreeRegressor(max_depth=5, random_state=21)
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn = sklearn_tree.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
sklearn_mae = mean_absolute_error(y_test, y_pred_sklearn)
print("Root Mean Square Error (RMSE):", rms)
print("Mean Absolute Error (MAE): {:.4f}".format(sklearn_mae))



model = DecisionTree(criterion="gini_index", max_depth=5)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

score = rmse(y_hat, y_test)
score2 = mae(y_hat, y_test)
print("Self implemented Decision Tree R^2: {:.4f}".format(score))
print("Mean Absolute Error (MAE): {:.4f}".format(score2))