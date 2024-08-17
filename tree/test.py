from base import *
from utils import *

from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns


data = sns.load_dataset('iris')

cols = list(data.columns)

X = data[cols[:-1]]
y = data[cols[-1]]

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

# # Display the first few rows of the DataFrame
# root = Node()
# root = create_Tree(X, y, 0, 3, real_target, real_features)
# def Display_Node(root: Node, depth=0):
#     indent = "    " * depth  # Create indentation based on the depth of the node
    
#     if root.feature is not None:
#         if root.threshold is None:
#             print(f"{indent}Node: Feature = {root.feature}, Discrete Value = {root.threshold}")
#         else:
#             print(f"{indent}Node: Feature = {root.feature}, Threshold = {root.threshold}")
#     else:
#         print(f"{indent}Leaf: Value = {root.value}")
    
#     for child in root.children:
#         Display_Node(child, depth + 1)  # Recursively display child nodes with increased depth

# # Display the tree starting from the root
# Display_Node(root)

dt = DecisionTree("entropy")
dt.fit(X,y)
dt.plot()
dt.predict(X[50:100])
print(dt.predictions)
print("---------------------")

from sklearn.tree import DecisionTreeClassifier, export_text
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Print the decision tree structure
tree_rules = export_text(clf, feature_names=X.columns.tolist())
print(tree_rules)

print("---------------------")






