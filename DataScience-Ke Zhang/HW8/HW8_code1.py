from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
data = pd.read_csv('high_income.csv')

# Separate features and label
X = data.drop(columns=['label'])  # Assuming 'label' is the name of the target column
y = data['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Initialize and train SVM models with linear and rbf kernels
svm_linear = SVC(kernel='linear')
svm_rbf = SVC(kernel='rbf')

svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

accuracy_linear = accuracy_score(y_test, y_pred_linear)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

# Determine best model
best_model = 'linear' if accuracy_linear > accuracy_rbf else 'rbf'
best_accuracy = max(accuracy_linear, accuracy_rbf)

# Output the results
print(f'Accuracy of linear kernel SVM: {accuracy_linear:.2f}')
print(f'Accuracy of rbf kernel SVM: {accuracy_rbf:.2f}')
print(f'Best model: {best_model} kernel with accuracy of {best_accuracy:.2f}')
