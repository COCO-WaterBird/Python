import pandas as pd
from mlxtend.evaluate import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#####################
# Sorting head and tail PC
#####################
def sortPC(df):
    for i in range(df.shape[1]):
        sorted_ascend = df.sort_values(df.columns[i], ascending=False).head(3)
        sorted_descend = df.sort_values(df.columns[i], ascending=False).tail(3)
        top_indices = sorted_ascend.index.tolist()
        bottom_indices = sorted_descend.index.tolist()
        print('PC',i+1, '= Highly correlated directly with', top_indices, 'and indirectly with', bottom_indices)

# Input dataframe
org_df = pd.read_csv('world_ds.csv', index_col=0)

#Create the dataframe of features and label
label_df = org_df.loc[:, org_df.columns == 'development_status']
feat_df = org_df.loc[:, org_df.columns != 'development_status']

# Wrapper method_Forward
#Best KNN_model as classifier

best_std_err = 0
k = 0
for i in range(2, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    sfs = SFS(knn, k_features='best', forward=True,scoring='accuracy',cv=5)
    sfs.fit(feat_df, label_df.values.ravel())
    sfs_metric_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    value_std_err = sfs_metric_df.iloc[2,6]
    if value_std_err > best_std_err:
        best_std_err = value_std_err
        k = i
print("best_std_err is", k)
# print(best_std_err)

# # SFG
# Sequential Feature Selector to select best three features
knn = KNeighborsClassifier(n_neighbors = k)
sfs = SFS(knn, k_features='best', forward=True,scoring='accuracy',cv=5)
sfs.fit(feat_df, label_df.values.ravel())
sfs_metric_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
# print(sfs_metric_df)
selected_features_df = feat_df[list(sfs_metric_df['feature_names'][3])]
# print(selected_features_df)
print("The best three features is ", list(sfs_metric_df['feature_names'][3]))

# Accuracy by KNN
knn.fit(feat_df, label_df)
y_pred_knn = knn.predict(feat_df)
accuracy_knn = accuracy_score(label_df, y_pred_knn.ravel())
print('Accuracy by Original Features:', accuracy_knn)

# PCA model
# Normalize all features
norm_feat_df = (feat_df - feat_df.mean()) / feat_df.std()

# Create and train PCA model
pca_model = PCA(n_components=3)
pca_model.fit(norm_feat_df)
transformed_features_df = pca_model.transform(norm_feat_df)
pca_data = pd.DataFrame(transformed_features_df, columns=['PC1', 'PC2', 'PC3'])
# print(pca_data)
pca_feature_relationship = pd.DataFrame(pca_model.components_.T, columns=['PC1', 'PC2', 'PC3'], index=norm_feat_df.columns)
# print(pca_feature_relationship)
sortPC(pca_feature_relationship)

# Accuracy by PCA
knn.fit(transformed_features_df, label_df)
y_pred_pca = knn.predict(transformed_features_df)
accuracy_pca = accuracy_score(label_df, y_pred_pca)
print('Accuracy by PCA:',accuracy_pca)

# Prediction by variables made using LDA
# Create and train LDA model
lda = LDA(n_components=2)
lda.fit(feat_df, label_df)

# Transform existing features to new features
transformed_features_lda = lda.transform(feat_df)

# Predict using new features
knn.fit(transformed_features_lda, label_df)
y_pred_lda = knn.predict(transformed_features_lda)

#Display the accuracy
accuracy_lda = accuracy_score(label_df, y_pred_lda)
print('Accuracy by LDA:', accuracy_lda )