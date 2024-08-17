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
        
    elif criterion == "mse":
        HS = mse(Y)  # Initial MSE of the whole dataset
        Gain = HS    # Start with the initial MSE
        left = Y[attr]  # Subset of Y where attr is True (left side of the split)
        right = Y[~attr]  # Subset of Y where attr is False (right side of the split)
        
        # Weighted average of the MSE after the split
        left_weight = len(left) / len(Y)
        right_weight = len(right) / len(Y)
        Gain -= left_weight * mse(left) + right_weight * mse(right)

        return Gain



def get_best_split(X: pd.Series, y: pd.Series,  real, criterion: str='mse') -> float:
    best_split_value = None
    best_score = -float('inf') 

    if (real):
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


def get_best_val(X: pd.Series, y: pd.Series, real, criterion: str='entropy') -> str:
    best_val = None
    best_info_gain = -float('inf')
    
    if (real==0):
        best_info_gain=information_gain(y, X, criterion)
        
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


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features: pd.Series, real_target: bool, real_feature):
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
                split, info_gain=get_best_split(X.loc[:,i], y, real_feature[c])
                c += 1
                if info_gain>best_info_gain:
                    best_info_gain=info_gain
                    best_split=split
                    best_feature=i
    else:
        c = 0
        for i in features:
            split, info_gain=get_best_val(X.loc[:,i], y, real_feature[c])
            c += 1
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


class Node:
    def __init__(self, feature=None, threshold=None, value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split on
        self.children = []
        self.value = value          # Value if it's a leaf node (used for prediction)



def create_Tree(X: pd.DataFrame, y: pd.Series, depth: int, max_depth: int, real_target: bool, real_features):
    """
    Recursive function to create a decision tree.
    """
    # Check if we're at max depth or if the node is pure
    if depth >= max_depth or len(np.unique(y)) == 1:
        # Create a leaf node
        branch = Node()
        leaf_value = y.mode()[0] if not real_target else y.mean()
        branch.value=leaf_value
        return branch

    # Get the best feature and split point
    features = X.columns
    best_feature, best_split = opt_split_attribute(X, y, features, real_target, real_features)

    # If no valid split is found, create a leaf node
    if best_feature is None:
        branch = Node()
        leaf_value = y.mode()[0] if not real_target else y.mean()
        branch.value=leaf_value
        return branch

    # Create a decision node
    if best_split != None:
        branch = Node(feature=best_feature, threshold=best_split)
    else:
        print(X[best_feature].unique())
        branch = Node(feature=best_feature, threshold=X[best_feature].unique())

    # Split the dataset and create child nodes
    splits = split_data(X, y, best_feature, best_split)
    for X_split, y_split in splits:
        child_node = create_Tree(X_split, y_split, depth + 1, max_depth, real_target, real_features)
        branch.children.append(child_node)

    return branch




from sklearn.datasets import load_iris
import pandas as pd
url = "train.csv"
df = pd.read_csv(url)
df = df.iloc[:200, :]
df = df.drop(columns=['Name','PassengerId','Ticket'])
X = df.drop(columns=['Survived'])
y = df['Survived']

print(X)
print("---------------------")
print(y)

real_target = check_ifreal(y)
real_features = []
for i in range(X.shape[1]):
    print(X.columns[i])
    if check_ifreal(X.iloc[:, i]):
        real_features.append(1)
    else:
        real_features.append(0)

# Display the first few rows of the DataFrame
root = Node()
root = create_Tree(X, y, 0, 3, real_target, real_features)

def Display_Node(root: Node, depth=0):
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


def predict_single(root: Node, X_row: pd.Series):
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

def predict(root: Node, X_test: pd.DataFrame):
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

# Example usage with a test DataFrame X_test
# X_test = pd.DataFrame(...)  # Define your test DataFrame
# predictions = predict(root, X_test)
# print(predictions)


predictions = predict(root, X)
score = np.sum(predictions==y)
print(score/2)
print(predictions)




