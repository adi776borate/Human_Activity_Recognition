import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import *
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# Write the code for Q2 a) and b) below. Show your results.
for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=10)
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))




# Define the number of folds (k)
k = 5

predictions = {}
accuracies = []
super_hypers = []
fold_size = len(X_train) // k

for i in range(k):
    # Split the data into training and test sets
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    test_set = pd.DataFrame(X_train[test_start:test_end])
    test_labels = pd.Series(y_train[test_start:test_end])
    
    training_set = pd.DataFrame(np.concatenate((X_train[:test_start], X_train[test_end:]), axis=0))
    training_labels = pd.Series(np.concatenate((y_train[:test_start], y_train[test_end:]), axis=0))

    
    best_hypers = []
    inner_fold = len(training_set) // k
    for j in range(k):
        val_beg = j * inner_fold
        val_fin = (j+1) * inner_fold
        val_X = pd.DataFrame(training_set[val_beg:val_fin])
        val_y = pd.Series(training_labels[val_beg:val_fin])
        training_X = pd.DataFrame(np.concatenate((training_set[:val_beg], training_set[val_fin:]), axis=0))
        training_y = pd.Series(np.concatenate((training_labels[:val_beg], training_labels[val_fin:]), axis=0))

        hyperparameters = {}
        hyperparameters['max_depth'] = [4,6,8,10,12,14]
        hyperparameters['criterion'] = ['information_gain', 'gini_index']

        max_score = 0
        best_depth = 3
        best_criterion = "information_gain"
        for d in hyperparameters['max_depth']:
            for c in hyperparameters['criterion']:
                model = DecisionTree(criterion=c, max_depth=d)
                model.fit(training_X, training_y)
                y_hat = model.predict(val_X)
                score = accuracy(y_hat, val_y)
                if (score>max_score):
                    best_depth = d
                    best_criterion = c
                    max_score = score

        best_hypers.append([best_depth, best_criterion])
    
    super_depth = 0
    super_criterion = "information_gain"
    max_accuracy = 0
    for l in range(len(best_hypers)):
        model = DecisionTree(criterion=best_hypers[l][1], max_depth=best_hypers[l][0])
        model.fit(training_set, training_labels)
        y_pred = model.predict(test_set)
        acc = accuracy(y_pred, test_labels)
        if (acc>max_accuracy):
            max_accuracy = acc
            super_depth = best_hypers[l][0]
            super_criterion = best_hypers[l][1]

    
    super_hypers.append([super_depth, super_criterion])
    accuracies.append(max_accuracy)
                

# Print the predictions and accuracies of each fold
for i in range(k):
    print("Fold {}: Accuracy: {:.4f}".format(i+1, accuracies[i]))
    super_model = DecisionTree(criterion=super_hypers[i][1], max_depth=super_hypers[i][0])
    super_model.fit(X_train, y_train)
    Y_hat = super_model.predict(X_test)
    Acc = accuracy(Y_hat, y_test)
    print("Model {}: Accuracy on test: {:.4f}".format(i+1, Acc))