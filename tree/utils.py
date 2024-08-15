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

    pass

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if np.issubdtype(y.dtype, np.number):
        unique=len(y.unique())
        total=y.count()
        if (unique/total)<0.1:  
            return False
        else:
            return True
    else:
        return False
    pass


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
    pass


def mse(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    avg=Y.mean()
    sq_err=(Y-avg)**2
    mean_sq_err=sq_err.mean()
    return mean_sq_err
    pass


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    Gain=0

    if (criterion=="entropy"):
        HS=entropy(Y)
        Gain+=HS
        c=attr.unique()
        for i in range(len(c)):
            Si=[]
            for j in range(len(attr)):
                if attr.iloc[j] == c[i]:
                    Si.append(Y.iloc[j])
            Gain-=(len(Si)/len(Y))*entropy(pd.Series(Si))

        return Gain
    
    elif (criterion=="mse"):
        HS=mse(Y)
        Gain+=HS
        c=attr.unique()
        for i in range(len(c)):
            Si=[]
            for j in range(len(attr)):
                if attr.iloc[j]==c[i]:
                    Si.append(Y.iloc[j])
            Gain-=(len(Si)/len(Y))*mse(pd.Series(Si))
        
        return Gain
    pass



def get_best_split(X: pd.Series, y: pd.Series, criterion: str='mse') -> float:
    best_split_value = None
    best_score = -float('inf') 

    if (check_ifreal(X)):
        sorted_indices = np.argsort(X)
        sorted_X = X.iloc[sorted_indices]
        sorted_y = y.iloc[sorted_indices]
        
        for i in range(1, len(sorted_X)):
            split_value = (sorted_X.iloc[i-1] + sorted_X.iloc[i]) / 2
            
            left = sorted_X <= split_value
            info_gain = information_gain(y, left, criterion)
            if info_gain > best_score:
                best_score = info_gain
                best_split_value = split_value
    else:
        best_score=information_gain(y, X, criterion)
        
    return best_split_value, best_score


def get_best_val(X: pd.Series, y: pd.Series, criterion: str='entropy') -> str:
    best_val = None
    best_info_gain = -float('inf')
    
    if (check_ifreal(X)==False):
        best_info_gain=information_gain(y, X, criterion)
        # for val in X.unique:
        #     info_gain = information_gain(y, X, criterion)
        #     print(info_gain)
        #     if info_gain > best_info_gain:
        #         best_info_gain = info_gain
        #         best_val = val
        
    else:
        sorted_indices = np.argsort(X)
        sorted_X = X.iloc[sorted_indices]
        sorted_y = y.iloc[sorted_indices]
        
        for i in range(1, len(sorted_X)):
            split_value = (sorted_X.iloc[i-1] + sorted_X.iloc[i]) / 2
            
            left = sorted_X <= split_value
            info_gain = information_gain(y, left, criterion)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_val = split_value
            
    return best_val, best_info_gain



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features: pd.Series):
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
    if check_ifreal(y):
        for i in features:
                split, info_gain=get_best_split(X.loc[:,i], y)
                if info_gain>best_info_gain:
                    best_info_gain=info_gain
                    best_split=split
                    best_feature=i
    else:
        for i in features:
            split, info_gain=get_best_val(X.loc[:,i], y)
            if info_gain>best_info_gain:
                best_info_gain=info_gain
                best_split=split
                best_feature=i
    return best_feature, best_split

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    splits = []

    if value is None:
        # Split based on all unique values of the attribute
        unique_values = X[attribute].unique()
        
        for u in unique_values:
            X_split = X[X[attribute] == u]
            y_split = y[X[attribute] == u]
            # Append both DataFrame and Series
            splits.append((X_split, y_split))
    else:
        # Split based on the provided value
        X_split_left = X[X[attribute] <= value]
        y_split_left = y[X[attribute] <= value]
        splits.append((X_split_left, y_split_left))
        
        X_split_right = X[X[attribute] > value]
        y_split_right = y[X[attribute] > value]
        splits.append((X_split_right, y_split_right))


    return splits


class Node:
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split on
        self.children = []
        self.value = value          # Value if it's a leaf node (used for prediction)

def create_Tree(X: pd.DataFrame, y: pd.Series, depth: int, max_depth: int):
    # Choosing the best attribute and split
    # Splitting the dataframe
    # Doing this until depth is reached


    if depth==max_depth:
        print("Max. depth reached")
        return 
    
    features = X.columns
    best_feature, best_split = opt_split_attribute(X, y, features)
    splits=split_data(X, y, best_feature, best_split)
    branch = Node(best_feature, best_split)
    for X_split, y_split in splits:
        child_node = create_Tree(X_split, y_split, depth + 1, max_depth)
        branch.children.append(child_node)
    print(branch.children)
    return branch




# Combination 1: Real Input, Real Output
X_real_real = pd.DataFrame({
    'Education Level': ['High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'High School', 'PhD', 'Bachelors', 'Masters'],
    'Age': [15, 25, 30, 40, 28, 35, 18, 35, 22, 32],
})
y_real = pd.Series([5500, 60000, 90000, 120000, 35000, 80000, 15000, 190000, 45000, 90000], name='Salary')

print(create_Tree(X_real_real, y_real, 0, 2))

# Combination 2: Real Input, Discrete Output
X_real_discrete = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    'Income': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]
})
y_discrete = pd.Series(['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'], name='Will Buy')


# Combination 3: Discrete Input, Real Output
X_discrete_real = pd.DataFrame({
    'Education Level': ['High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'High School', 'PhD', 'Bachelors', 'Masters'],
    'City': ['CityA', 'CityB', 'CityA', 'CityC', 'CityB', 'CityA', 'CityC', 'CityB', 'CityA', 'CityC']
})
y_real = pd.Series([60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000], name='Salary')


# Combination 4: Discrete Input, Discrete Output
X_discrete_discrete = pd.DataFrame({
    'Education Level': ['High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'High School', 'PhD', 'Bachelors', 'Masters'],
    'City': ['CityA', 'CityB', 'CityA', 'CityC', 'CityB', 'CityA', 'CityC', 'CityB', 'CityA', 'CityC']
})
y_discrete = pd.Series(['No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'], name='Will Buy')

