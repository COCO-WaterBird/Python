import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import _tree
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


#Input diabetes data
org_df = pd.read_csv("diabetes.csv")

##Detect Outliers for Glucose
q3, q1 = np.percentile(org_df['Glucose'], [75,25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
org_df.loc[(org_df['Glucose'] < lower_band) | (org_df['Glucose'] > upper_band) , 'Glucose'] = None

##Detect Outliers for BMI
q3, q1 = np.percentile(org_df['BMI'], [75,25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
org_df.loc[(org_df['BMI'] < lower_band) | (org_df['BMI'] > upper_band) , 'BMI'] = None

##MICE
imputer = IterativeImputer(max_iter=10,random_state=0)
imputer.dataset = imputer.fit_transform(org_df)
data = pd.DataFrame(imputer.dataset,columns=org_df.columns)

#Define features to predict Resistance label
label_df = org_df.loc[:, org_df.columns == "Outcome"]
feat_df = org_df.loc[:, org_df.columns != 'Outcome']

#Seperate test/train/validation data
train_feat, temp_feat, train_label, temp_label = train_test_split(feat_df, label_df,test_size=0.28)
test_feat, val_feat, test_label, val_label = train_test_split(temp_feat, temp_label,test_size=0.28)

#D-Tree Hyper-parameters
min_impurity_thr = [0.001,0.0001]
min_samples_split_thr = [5,10]
max_depth_thr = [3,5]
min_samples_leaf_thr = [3,5]
ccp_thr = [0.001,0.0001]

#Creating a model using Hyper-parameter
best_accuracy = 0
best_tree = None

for min_impurity in min_impurity_thr:
    for min_samples_split in min_samples_split_thr:
        for max_depth in max_depth_thr:
            for min_samples_leaf in min_samples_leaf_thr:
                for ccp_alpha in ccp_thr:
                    treemodel = tree.DecisionTreeClassifier(
                        min_impurity_decrease=min_impurity,
                        min_samples_split = min_samples_split,
                        max_depth = max_depth,
                        min_samples_leaf = min_samples_leaf,
                        ccp_alpha = ccp_alpha
                    )

                    treemodel.fit(train_feat, train_label)  #Train the model

                    val_label_pred = treemodel.predict(val_feat)                #Accuracy on validation data
                    val_accuracy = accuracy_score(val_label, val_label_pred)   #Accuracy on validation data

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_tree = treemodel
print('min_impurity =', min_impurity,
      'min_samples_split =', min_samples_split,
      'max_depth =', max_depth,
      'min_samples_leaf =', min_samples_leaf,
      'ccp_alpha =', ccp_alpha)

##Visualize the model
plt.figure(figsize=(9,9))
tree.plot_tree(treemodel, feature_names=train_feat.columns, class_names=('No Diabetes', 'Diabetes'), filled=True)
plt.show()

##Confusion matrix
Confusion_Matrix = confusion_matrix(val_label, val_label_pred)
ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix,display_labels= ['No Diabetes', 'Diabetes']).plot()
plt.show()

#Calculate accuracy of the best Dtree on test data
test_label_pred = best_tree.predict(test_feat)
testing_accuracy = accuracy_score(test_label, test_label_pred)
print('testing result =',testing_accuracy)

#Define get_rules
def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules

#Extract three rules form the D-tree
rules = get_rules(treemodel, ['Glucose', 'BMI', 'Age'] ,["No diabetes","Diabetes"])
i = 0
for r in rules:
    i += 1
    if i <= 3:
        print(r)
    else:
        break

