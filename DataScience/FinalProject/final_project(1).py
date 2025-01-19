import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# Step 1: Data Preprocessing
# If Data Need Encode
def encoding_and_Normalize(data):
    for column in data.columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].astype('category').cat.codes
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

# remove outliers , Impute missing,Normalize
def preprocess(data,name):
    # Remove outliers using IQR
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        data.loc[(data[column] < fence_low) | (data[column] > fence_high), column] = np.nan

    # Impute missing values using IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=0)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Normalize all columns
    data_normalized = (data_imputed - data_imputed.mean()) / data_imputed.std()

    data_normalized.to_csv(name, index=False)
    return data_normalized


# Step 2: Unsupervised Learning for generating labels:
def cluster_and_label(data,name):
    # Use K-means clustering on three features of Glucose, BMI and Age to cluster data into two clusters
    features = data[['Glucose', 'BMI', 'Age']]
    kmeans = KMeans(n_clusters=2, random_state=0)
    data['Cluster'] = kmeans.fit_predict(features)

    # Add a new column (Outcome) to the dataset containing 1 for ‘Diabetes’ and 0 for ‘No Diabetes’. Use these values as labels for classification (step 4)

    data['Outcome'] = data['Cluster'].apply(
        lambda x: 1 if x == data.groupby('Cluster')['Glucose'].mean().idxmax() else 0
    )
    # Assign ‘Diabetes’ name to the cluster with higher average Glucose and ‘No Diabetes’ to the other cluster
    data['Cluster_Label'] = data['Outcome'].apply(
        lambda x: 'Diabetes' if x == 1 else 'No Diabetes'
    )

    data.to_csv(name, index=False)
    return data.drop(columns=['Cluster', 'Cluster_Label'])


# Step 3: Feature Extraction
def apply_pca(data):
    feature_data = data.iloc[:, :-1]
    label_data = data.iloc[:, -1]
    feature_train, feature_test, label_train, label_test = train_test_split(feature_data, label_data, test_size=0.2,
                                                                            random_state=0)
    #Use PCA on the training data to create 3 new components from existing features (all columns except output)
    pca = PCA(n_components=3)
    transformed_train = pca.fit_transform(feature_train)
    transformed_test = pca.transform(feature_test)
    #Transfer training and test data to the new dimensions (PCs)
    train_pca_df = pd.DataFrame(transformed_train, columns=['PC1', 'PC2', 'PC3'])
    test_pca_df = pd.DataFrame(transformed_test, columns=['PC1', 'PC2', 'PC3'])

    return train_pca_df, test_pca_df, label_train, label_test


# Step 4: Train Super Learner
def train_super_learner(x_train, x_test, y_train, y_test, max_iter, param_grid, cv):
    #Define three classification models as base classifiers consisting of Naïve Bayes, Neural Network, and KNN
    nb_model = GaussianNB()
    knn_model = KNeighborsClassifier()
    nn_model = MLPClassifier(random_state=42, max_iter=max_iter, verbose=0)

    base_classifiers = [
        ('nb', nb_model),
        ('knn', knn_model),
        ('nn', nn_model)
    ]
    #Define a decision tree as the meta learner
    meta_learner = DecisionTreeClassifier(random_state=0)
    #Train decision tree (meta learner) on outputs of three base classifiers using 5-fold cross validation.
    super_learner = StackingClassifier(estimators=base_classifiers, final_estimator=meta_learner, cv=cv)
    #Find hyperparameters for all these models which provide the best accuracy rate
    grid_search = GridSearchCV(super_learner, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Get the best cross-validated accuracy score

    test_predictions = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    return best_params,test_accuracy


# diabetes dataset
diabetes_data = pd.read_csv("diabetes_project.csv")
diabetes_data_cleaned = preprocess(diabetes_data,"cleaned_diabetes.csv")
# print(diabetes_data_cleaned.head())
diabetes_data_labeled = cluster_and_label(diabetes_data_cleaned,name="clustered_diabetes.csv")
# print(diabetes_data_labeled.head())
train_pca, test_pca, train_labels, test_labels = apply_pca(diabetes_data_labeled)
# print(train_pca, test_pca, train_labels, test_labels)
param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'nn__hidden_layer_sizes': [(25,), (50,), (100,)],
    'final_estimator__max_depth': [3, 5, 8],
    }
best_params_diabetes,test_accuracy_diabetes = train_super_learner(train_pca, test_pca, train_labels, test_labels,max_iter=1000,cv=5,param_grid=param_grid)
print(f"Best Hyperparameters on Diabetes Dataset: {best_params_diabetes}")
print(f"Test Accuracy on Diabetes Dataset: {test_accuracy_diabetes}")

# Employing the model on breast_cancer_survival datasets
breast_cancer_data = pd.read_csv("Breast_Cancer_Survival.csv")
breast_cancer_data = breast_cancer_data.sample(frac=0.5, random_state=42)
label_column_name = breast_cancer_data.columns[-1]
label_breast_cancer = breast_cancer_data[label_column_name]
# print(label_column_name)
# print(label_breast_cancer.head())
feature_encode = encoding_and_Normalize(breast_cancer_data.iloc[:, :-1])
# print(feature_encode.head())
breast_cancer_data_cleaned = preprocess(feature_encode,"cleaned_breast_cancer.csv")
breast_cancer_data_cleaned[label_column_name] = label_breast_cancer.reset_index(drop=True)
# print(breast_cancer_data_cleaned)

train_pca_df, test_pca_df, label_train, label_test = apply_pca(breast_cancer_data_cleaned)
param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'nn__hidden_layer_sizes': [(25,), (50,), (100,)],
    'final_estimator__max_depth': [3, 5, 8],
}
best_params_breast_cancer,test_accuracy_breast_cancer = train_super_learner(train_pca_df, test_pca_df, label_train, label_test,max_iter=500,cv=3,param_grid=param_grid)
print(f"Best Hyperparameters on Breast_Cancer_Survival Dataset: {best_params_breast_cancer}")
print(f"Test Accuracy on Breast_Cancer_Survival Dataset: {test_accuracy_breast_cancer}")
