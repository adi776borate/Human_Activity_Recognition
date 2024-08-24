"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_numeric_dtype(y):
        unique=len(y.unique())
        total=y.count()
        if (unique/total)<0.1:  
            return False
        else:
            return True
    else:
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    n=Y.size
    classes=Y.unique()
    H=0
    for i in range(len(classes)):
        m=0
        for j in range(len(Y)):
            if Y.iloc[j]==classes[i]:
                m+=1
        p=m/n
        H+=p*np.log2(p)
    return -1*H


def mse(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    avg=Y.mean()
    sq_err=(Y-avg)**2
    mean_sq_err=sq_err.mean()
    return mean_sq_err


def gini(y: pd.Series) -> float:
        classes = y.value_counts(normalize=True)
        return 1 - np.sum(classes ** 2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    Gain=0
    if (criterion=="entropy"):
        HS=entropy(Y)
        Gain+=HS
        c = attr.unique()
        for i in range(len(c)):
            Si=[]
            for j in range(len(attr)):
                if attr.iloc[j] == c[i]:
                    Si.append(Y.iloc[j])
            Gain-=(len(Si)/len(Y))*entropy(pd.Series(Si))

        return Gain
    
    elif criterion == "gini_index":
        HS = gini(Y)
        Gain = HS
        c = attr.unique()
        for i in range(len(c)):
            Si=[]
            for j in range(len(attr)):
                if attr.iloc[j] == c[i]:
                    Si.append(Y.iloc[j])
            Gain-=(len(Si)/len(Y))*gini(pd.Series(Si))

        return Gain

    elif criterion == "mse":
        HS = mse(Y)
        Gain = HS
        c = attr.unique()
        for i in range(len(c)):
            Si = Y[attr == c[i]]
            Gain -= (len(Si) / len(Y)) * mse(Si)
            
        return Gain

# This function is called when the output label is real.
def get_best_split(X: pd.Series, y: pd.Series,  real, criterion: str='information_gain') -> float:
    best_split_value = None
    best_score = -float('inf') 

    # INput feature is real
    if (real):
        # Sort the indices based on value of X
        sorted_indices = np.argsort(X)
        sorted_X = X.iloc[sorted_indices]
        sorted_y = y.iloc[sorted_indices]
        
        for i in range(1, len(sorted_X)):
            # checks various possible splits
            split_value = (sorted_X.iloc[i-1] + sorted_X.iloc[i]) / 2
            # segregating the series based on split values
            left = sorted_X <= split_value
            if (criterion=='information_gain'):
                info_gain = information_gain(y, left, 'mse')
            else:
                info_gain = information_gain(y, left, 'mse')
            # storing the best info gain
            if info_gain > best_score:
                best_score = info_gain
                best_split_value = split_value

    # INput feature is discrete
    else:
        if (criterion=='information_gain'):
            best_score = information_gain(y, X, 'mse')
        else:
            best_score = information_gain(y, X, 'mse')
        
    return best_split_value, best_score

# This function is called when the output label is discrete
def get_best_val(X: pd.Series, y: pd.Series, real, criterion: str='information_gain') -> str:
    best_val = None
    best_info_gain = -float('inf')
    
    # Discrete feature 
    if (real==0):
        if (criterion=='information_gain'):
            best_info_gain=information_gain(y, X, 'entropy')
        else:
            best_info_gain=information_gain(y, X, 'gini_index')
    
    # Real feature
    else:
        # Sort the indices based on value of X
        sorted_indices = np.argsort(X)
        sorted_X = X.iloc[sorted_indices]
        sorted_y = y.iloc[sorted_indices]
        
        for i in range(1, len(sorted_X)):
            # checks various possible splits
            split_value = (sorted_X.iloc[i-1] + sorted_X.iloc[i]) / 2
            # segregating the series based on split values
            left = sorted_X <= split_value
            if (criterion=='information_gain'):
                info_gain = information_gain(y, left, 'entropy')
            else:
                info_gain = information_gain(y, left, 'gini_index')
            # storing the best info gain
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_val = split_value
            
    return best_val, best_info_gain


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features: pd.Series, real_target: bool, real_feature, criterion: str):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to whether the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or variance based on the type of output) or minimum gini index (discrete output).
    best_feature=None
    best_split=None
    best_info_gain=-float('inf')
    if real_target:
        c = 0
        for i in features:
            split, info_gain = get_best_split(X.loc[:,i], y, real_feature[c], criterion)
            c += 1
            # Gets the best info gain and split among various features 
            if info_gain>best_info_gain:
                best_info_gain=info_gain
                best_split=split
                best_feature=i
    else:
        c = 0
        for i in features:
            split, info_gain = get_best_val(X.loc[:,i], y, real_feature[c], criterion)
            c += 1
            # Gets the best info gain and split among various features 
            if info_gain>best_info_gain:
                best_info_gain=info_gain
                best_split=split
                best_feature=i
    return best_feature, best_split


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Function to split the data according to an attribute.
    Handles both discrete and real-valued features.
    """

    if value is None:
        # For discrete features
        splits = []
        unique_values = X[attribute].unique()
        
        for u in unique_values:
            X_split = X[X[attribute] == u]
            y_split = y[X[attribute] == u]
            splits.append((X_split, y_split))
    else:
        # For real-valued features
        X_split_left = X[X[attribute] <= value]
        y_split_left = y[X[attribute] <= value]
        
        X_split_right = X[X[attribute] > value]
        y_split_right = y[X[attribute] > value]
        
        splits = [(X_split_left, y_split_left), (X_split_right, y_split_right)]
        
    return splits
