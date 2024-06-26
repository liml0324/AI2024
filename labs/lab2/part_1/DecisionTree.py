from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import pandas as pd 

continue_features = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF', 'TUE', 'FCVC', ]
discrete_features = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
discrete_features_value_range = {}
split_threshold = 16    # if the number of samples in a node is less than split_threshold, stop split

# metrics
def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

class DecisionTreeNode:
    def __init__(self):
        self.predicted_class = None
        self.feature_index = None
        self.threshold = [] # 如果是连续值，阈值数量比children数量少1
                            # 如果是离散值，阈值数量等于children数量
        self.children = []
        
def entropy(y):
    # y: [n_samples, ]
    # return: entropy
    ent = 0
    for i in np.unique(y):
        p = sum(y == i) / len(y)
        ent -= p * np.log2(p)
    return ent
    
        
def tree_generate(X, y, A):
    node = DecisionTreeNode()
    # if all samples in X belong to the same class, return a leaf node with the class label
    if len(np.unique(y)) == 1:
        node.predicted_class = y[0]
        return node
    # if A is empty, return a leaf node with the majority class label
    if len(A) == 0 or (A in continue_features and len(y) < split_threshold):
        node.predicted_class = np.argmax(np.bincount(y))
        return node
    # find the best split
    best_ent_gain = -10000
    best_feature = None
    ent_X = entropy(y)
    for feature in A:
        if feature in continue_features:
            # split the feature into two parts
            thresholds = np.unique(X[feature].values)
            thresholds.sort()
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2
                y_left = y[X[feature].values <= threshold]
                y_right = y[X[feature].values > threshold]
                if y_right.size == 0 or y_left.size == 0:
                    continue
                ent_left = entropy(y_left)
                ent_right = entropy(y_right)
                ent_gain = ent_X - (len(y_left) * ent_left + len(y_right) * ent_right) / len(y)
                if ent_gain > best_ent_gain:
                    best_ent_gain = ent_gain
                    best_feature = feature
                    best_threshold = threshold 
        else:
            ent_gain = ent_X
            for value in discrete_features_value_range[feature]:
                y_sub = y[X[feature].values == value]
                if y_sub.size == 0:
                    continue
                ent_gain -= len(y_sub) * entropy(y_sub) / len(y)
            if ent_gain > best_ent_gain:
                best_ent_gain = ent_gain
                best_feature = feature
    
    # split the node
    node.feature_index = best_feature
    if best_feature in continue_features:
        node.threshold.append(best_threshold)
        node.children = [tree_generate(X[X[best_feature] <= best_threshold], y[X[best_feature] <= best_threshold], A),
                         tree_generate(X[X[best_feature] > best_threshold], y[X[best_feature] > best_threshold], A)]
    else:
        new_A = [col for col in A if col != best_feature]
        # new_A.remove(best_feature)
        most_class = np.argmax(np.bincount(y))
        for value in discrete_features_value_range[best_feature]:
            node.threshold.append(value)
            if(len(y[X[best_feature] == value])) == 0:
                child = DecisionTreeNode()
                child.predicted_class = most_class
                node.children.append(child)
            else:
                node.children.append(tree_generate(X[X[best_feature] == value], y[X[best_feature] == value], new_A))
    return node
    

# model
class DecisionTreeClassifier:
    def __init__(self) -> None:
        self.tree = None

    def fit(self, X, y):
        # X: [n_samples_train, n_features], 
        # y: [n_samples_train, ],
        # TODO: implement decision tree algorithm to train the model
        A = [col.strip() for col in X.columns]
        # A = ['Height', 'Weight',]
        A.remove('Height')
        A.remove('Weight')
        self.tree = tree_generate(X, y, A)

    def predict(self, X):
        # X: [n_samples_test, n_features],
        # return: y: [n_samples_test, ]
        y = np.zeros(X.shape[0])
        # TODO:
        for i in range(X.shape[0]):
            row = X.iloc[i]
            node = self.tree
            while len(node.children) > 0:
                if node.feature_index in continue_features:
                    if X[node.feature_index].iloc[i] <= node.threshold[0]:
                        node = node.children[0]
                    else:
                        node = node.children[1]
                else:
                    for j in range(len(node.threshold)):
                        # print(node.feature_index, X[node.feature_index].values)
                        # print(discrete_features_value_range[node.feature_index])
                        if X[node.feature_index].iloc[i] == node.threshold[j]:
                            node = node.children[j]
                            break
            y[i] = node.predicted_class
            # print("finished", i)
        return y

def load_data(datapath:str='./data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    df = pd.read_csv(datapath)
    continue_features = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF', ]
    discrete_features = ['Gender', 'CALC', 'FAVC', 'FCVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'TUE', 'CAEC', 'MTRANS']
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # columns = [col.strip() for col in X.columns]
    # encode discrete str to number, eg. male&female to 0&1
    labelencoder = LabelEncoder()
    for col in discrete_features:
        X[col] = labelencoder.fit(X[col]).transform(X[col])
        # X[col] = X[col].astype(int)
        discrete_features_value_range[col] = np.unique(X[col].values)
    y = labelencoder.fit(y).fit_transform(y)
        
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_data('./data/ObesityDataSet_raw_and_data_sinthetic.csv')
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(accuracy(y_test, y_pred))