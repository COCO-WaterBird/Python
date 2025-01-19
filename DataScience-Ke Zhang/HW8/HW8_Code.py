import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mlxtend.evaluate import accuracy_score

# Train two SVM models, one with ‘linear’ kernel and another with ‘rbf’ kernel on dataset
# ( ‘label’ as label and all other attributes as features)
# using 75% of data as training and remaining data as test.
# Calculate accuracy of both models and report the best accuracy.

# Input Dataset
org_df = pd.read_csv("high_income.csv")

# Feature and label
label_df = org_df["label"]
feat_df = org_df.loc[:, org_df.columns != "label"]

#Train and test split
train_x, test_x, train_y, test_y = train_test_split(feat_df, label_df, test_size=0.25, random_state=42)

#Linear kernel
linear_model = SVC(kernel='linear')
linear_model.fit(train_x, train_y)
test_y_prediction = linear_model.predict(test_x)
linear_accuracy = accuracy_score(test_y, test_y_prediction)
print(f"Linear Kernel result = {linear_accuracy:.2f}")

# RBF kernel
rbf_model = SVC(kernel='rbf')
rbf_model.fit(train_x, train_y)

#Accuracy of model
rbf_accuracy = rbf_model.score(test_x, test_y)
# print(f"RBF Kernel result = {rbf_accuracy:.2f}")

# Select the best model between linear and rbf
if rbf_accuracy > linear_accuracy:
    print("The best accuracy is rbf model, accuracy is ", rbf_accuracy)
elif rbf_accuracy < linear_accuracy:
    print("The best accuracy is linear model, accuracy is ", linear_accuracy)
else:
    print("Linear and rbf kernel is equal")


