"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *


np.random.seed(42)

class Node:
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split on
        self.children = []
        self.value = value


def create_Tree(X: pd.DataFrame, y: pd.Series, depth: int, max_depth: int, real_target: bool, real_features, criterion: str):
    """
    Recursive function to create a decision tree.
    """
    # Check if we're at max depth or if the node is pure
    if depth >= max_depth or len(np.unique(y)) == 1:
        # Create a leaf node
        branch = Node()
        if len(y) == 0:
            leaf_value = None  # or a default value, like y.mean() from the parent node
        else:
            leaf_value = y.mode()[0] if not real_target else y.mean()
        branch.value=leaf_value
        return branch

    # Get the best feature and split point
    features = X.columns
    best_feature, best_split = opt_split_attribute(X, y, features, real_target, real_features, criterion)

    # If no valid split is found, create a leaf node
    if best_feature is None:
        branch = Node()
        if len(y) == 0:
            leaf_value = None
        else:
            leaf_value = y.mode()[0] if not real_target else y.mean()
        branch.value=leaf_value
        return branch

    # Create a decision node
    if best_split != None:
        branch = Node(feature=best_feature, threshold=best_split)
    else:
        branch = Node(feature=best_feature, threshold=X[best_feature].unique())

    # Split the dataset and create child nodes
    splits = split_data(X, y, best_feature, best_split)
    for X_split, y_split in splits:
        child_node = create_Tree(X_split, y_split, depth + 1, max_depth, real_target, real_features, criterion)
        branch.children.append(child_node)

    return branch



def predict_single(root: Node, X_row: pd.Series):
    current_node = root
    
    while current_node.children:
        feature_value = X_row[current_node.feature]
        
        if isinstance(current_node.threshold, (int, float)):
            # Real feature: Compare with the threshold
            if feature_value <= current_node.threshold:
                current_node = current_node.children[0]
            else:
                current_node = current_node.children[1]
        else:
            # Discrete feature: Handle multiple children
            if feature_value in current_node.threshold:
                # Find the corresponding child based on the feature value
                index = list(current_node.threshold).index(feature_value)
                current_node = current_node.children[index]
            else:
                print(f"Warning: Feature value {feature_value} not found in threshold {current_node.threshold}.")
                return -1  # Or another default value
     
    return current_node.value


def predict_result(root: Node, X_test: pd.DataFrame) -> pd.Series:
    predictions = pd.Series(index=X_test.index)  # Ensure the index is aligned with X_test
    for i in X_test.index:
        predictions.loc[i] = predict_single(root, X_test.loc[i, :])
    
    return predictions

def Display_Node(root: Node, depth=0, decision="?(X"):
    indent = "    " * depth  # Indentation based on depth
    
    if root.value is not None:
        print(f" Leaf: Value = {root.value: .3f}")
    elif isinstance(root.threshold, (int, float)):
        # Handle real-valued splits
        print(f" {decision}{root.feature} <= {root.threshold: .3f})")
        # Yes branch (<= threshold)
        print(f"{indent}    Y:", end="")
        Display_Node(root.children[0], depth + 1)
        # No branch (> threshold)
        print(f"{indent}    N:", end="")
        Display_Node(root.children[1], depth + 1)
    else:
        # Handle discrete-valued splits
        print(f" {decision} {root.feature} in {list(root.threshold)})")
        for i, value in enumerate(root.threshold):
            # Properly indent and print each branch corresponding to discrete values
            branch_decision = f"{value}:"
            print(f"{indent}    {branch_decision}", end="")
            Display_Node(root.children[i], depth + 1)



@dataclass
class DecisionTree:
    max_depth: int
    root: Node
    predicted: pd.Series
    criterion: str

    def __init__(self, criterion, max_depth=6):
        self.criterion = criterion
        self.root = None
        self.max_depth = max_depth
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: 
        # Performing one hot encoding on categorical features       
        X_encoded = one_hot_encoding(X)
        # Variable to store the nature of output label
        real_target = check_ifreal(y)
        # List to store the nature of input features
        real_features = []
        for i in range(X_encoded.shape[1]):
            feature_col = X_encoded.iloc[:, i]
            if check_ifreal(feature_col):
                real_features.append(1)
            else:
                real_features.append(0)

        head = create_Tree(X_encoded, y, 0, self.max_depth, real_target, real_features, self.criterion)
        self.root = head
            

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        X_encoded = one_hot_encoding(X)
        self.predicted = predict_result(self.root, X_encoded)
        return self.predicted


    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        
        # Display the tree starting from the root
        Display_Node(self.root)

    def visualise(self, y):
        plt.scatter(y, self.predicted)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.show()

