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
from utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self,criterion="gini_index" ,feature=None, threshold=None, max_depth = 5,value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split on
        self.children = []
        self.value = value          # Value if it's a leaf node (used for prediction)
        self.criterion = criterion
        self.max_depth = max_depth
        self.depth=0


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # if output is discrete we need a classifier
        # if output is real we need a regressor
        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        def create_Tree(X: pd.DataFrame, y: pd.Series, depth: int, max_depth: int, real_target: bool, real_features):
            """
            Recursive function to create a decision tree.
            """
            # Check if we're at max depth or if the node is pure
            if (depth >= max_depth) or (np.unique(y).shape[0] == 1):
                # Create a leaf node
                branch = DecisionTree(self.criterion)
                leaf_value = y.mode()[0] if not real_target else y.mean()
                branch.value=leaf_value
                return branch

            # Get the best feature and split point
            features = X.columns
            best_feature, best_split = opt_split_attribute(X, y, features, real_target, real_features)

            # If no valid split is found, create a leaf node
            if best_feature is None:
                branch = DecisionTree(self.criterion)
                leaf_value = y.mode()[0] if not real_target else y.mean()
                branch.value=leaf_value
                return branch

            # Create a decision node
            if best_split != None:
                branch = DecisionTree(self.criterion,feature=best_feature, threshold=best_split)
            else:
                print(X[best_feature].unique())
                branch = DecisionTree(self.criterion,feature=best_feature, threshold=X[best_feature].unique())

            # Split the dataset and create child nodes
            splits = split_data(X, y, best_feature, best_split)
            for X_split, y_split in splits:
                child_node = create_Tree(X_split, y_split, depth + 1, max_depth, real_target, real_features)
                branch.children.append(child_node)

            return branch
        
        real_target = check_ifreal(y)
        real_features = []
        for i in range(X.shape[1]):
            print(X.columns[i])
            if check_ifreal(X.iloc[:, i]):
                real_features.append(1)
            else:
                real_features.append(0)
        
        self.tree = create_Tree(X,y,self.depth,self.max_depth,real_target,real_features)
        


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        def predict_single(root, X_row: pd.Series):
            """
            Predict the output for a single data row using the decision tree.
            
            Parameters:
            - root: The root node of the decision tree.
            - X_row: A pandas Series containing a single row of features.
            
            Returns:
            - The predicted value.
            """
            current_node = root
            
            while current_node.children:
                # Check if the feature is real or discrete
                feature_value = X_row[current_node.feature]
                
                if isinstance(current_node.threshold, (int, float)):
                    # Real feature: Compare with the threshold
                    if feature_value <= current_node.threshold:
                        current_node = current_node.children[0]
                    else:
                        current_node = current_node.children[1]
                else:
                    # Discrete feature: Check if the value matches any in the threshold list
                    if feature_value in current_node.threshold:
                        current_node = current_node.children[0]
                    else:
                        current_node = current_node.children[1]
            
            # Return the value in the leaf node
            return current_node.value

        def predict(root, X_test: pd.DataFrame):
            """
            Predict the output for each row in the test DataFrame using the decision tree.
            
            Parameters:
            - root: The root node of the decision tree.
            - X_test: A pandas DataFrame containing test data.
            
            Returns:
            - A pandas Series containing the predictions for each row in X_test.
            """
            predictions = X_test.apply(lambda row: predict_single(root, row), axis=1)
            return predictions
        
        self.predictions = predict(self.tree,X)


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
        def Display_Node(root, depth=0):
            indent = "    " * depth  # Create indentation based on the depth of the node
            
            if root.feature is not None:
                if root.threshold is None:
                    print(f"{indent}Node: Feature = {root.feature}, Discrete Value = {root.threshold}")
                else:
                    print(f"{indent}Node: Feature = {root.feature}, Threshold = {root.threshold}")
            else:
                print(f"{indent}Leaf: Value = {root.value}")
            
            for child in root.children:
                Display_Node(child, depth + 1)  # Recursively display child nodes with increased depth

        # Display the tree starting from the root
        Display_Node(self.tree)
