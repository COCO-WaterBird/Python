import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt



# import Dataframe
df = pd.read_csv('diabetes.csv')

#Remove outlier as nan
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

#Mice
imputer = IterativeImputer(max_iter=10, random_state=0)
imputer_data = imputer.fit_transform(df)
df = pd.DataFrame(imputer_data, columns=df.columns)

#Split dataset with outcome as label and all other attributes as features
df_label = df['Outcome']
df_feat = df.loc[:, df.columns != 'Outcome']

#Define RF and Adaboost models
rf_3 = RandomForestClassifier(n_estimators=3, random_state=0)
rf_50 = RandomForestClassifier(n_estimators=50, random_state=0)
ada_3 = AdaBoostClassifier(n_estimators=3, algorithm='SAMME')
ada_50 = AdaBoostClassifier(n_estimators=50, algorithm='SAMME')

#Define K_models
k_models = KFold(n_splits=5)
score_rf_3 = cross_val_score(rf_3, df_feat, df_label, cv=k_models)
score_rf_50 = cross_val_score(rf_50, df_feat, df_label, cv=k_models)
score_ada_3 = cross_val_score(ada_3, df_feat, df_label, cv=k_models)
score_ada_50 = cross_val_score(ada_50, df_feat, df_label, cv=k_models)

#Calculate score means
score_rf_3_mean = score_rf_3.mean()
score_ada_3_mean = score_ada_3.mean()
score_rf_50_mean = score_rf_50.mean()
score_ada_50_mean = score_ada_50.mean()

#Plot to compare mean of scores
scores = [score_rf_3_mean, score_ada_3_mean, score_rf_50_mean,score_ada_50_mean]
models = ['score_rf_3_mean', 'score_ada_3_mean', 'score_rf_50_mean', 'score_ada_50_mean']

plt.figure(figsize=(15, 8))
plt.plot(models, scores, color='blue')
plt.xlabel('Model')
plt.ylabel('Score')
plt.show()

print('score_rf_3_mean =', score_rf_3.mean(),
      'score_ada_3_mean =', score_ada_3.mean(),
      'score_rf_50_mean =', score_rf_50.mean(),
      'score_ada_50_mean =', score_ada_50.mean()
      )


