import pandas as pd
import numpy as np
from numpy.f2py.crackfortran import externalpattern
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


#input dataset
train_df = pd.read_csv('hw4_train.csv')
test_df = pd.read_csv('hw4_test.csv')

#define features and outcome fot Regression
y_train = train_df['BloodPressure']
x_train = train_df.drop(columns=['BloodPressure'])
y_test = test_df['BloodPressure']
x_test = test_df.drop(columns=['BloodPressure'])

#replace x_train from 0 to nan
x_train['Pregnancies'] = x_train['Pregnancies'].replace(0, np.nan)
x_train.loc[x_train['Pregnancies'] > 10, 'Pregnancies'] = np.nan
x_train['SkinThickness'] = x_train['SkinThickness'].replace(0, np.nan)
x_train['Insulin'] = x_train['Insulin'].replace(0, np.nan)

#detect outlier in x_train_df for Pregnancies
q3, q1 = np.nanpercentile(x_train['Pregnancies'], [75, 25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
x_train.loc[(x_train['Pregnancies'] < lower_band) | (x_train['Pregnancies'] > upper_band) , 'Pregnancies'] = None

#detect outlier in x_train_df for SkinThickness
q3, q1 = np.nanpercentile(x_train['SkinThickness'], [75, 25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
x_train.loc[(x_train['SkinThickness'] < lower_band) | (x_train['SkinThickness'] > upper_band) , 'SkinThickness'] = None

#detect outlier in x_train for Insulin
q3, q1 = np.percentile(x_train['Insulin'], [75, 25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
x_train.loc[(x_train['Insulin'] < lower_band) | (x_train['Insulin'] > upper_band) , 'Insulin'] = None

#MICE in x_train
imputer = IterativeImputer(max_iter=10, random_state=0)
x_train_imputed = imputer.fit_transform(x_train)
x_train = pd.DataFrame(x_train_imputed, columns = x_train.columns)
x_train.to_csv('x_train02.csv', index=False)

#Smooth Noises using Binning
x_train['Pregnancies_level'] = pd.qcut(x_train['Pregnancies'], q=6, duplicates='drop')
x_train['Pregnancies'] = pd.Series([interval.mid if pd.notnull(interval) else None for interval in x_train['Pregnancies_level']])
del x_train['Pregnancies_level']

# Binning 'SkinThickness' into 5 bins and smoothing with midpoints
x_train['SkinThickness_level'] = pd.qcut(x_train['SkinThickness'], q=6, duplicates='drop')
x_train['SkinThickness'] = pd.Series([interval.mid if pd.notnull(interval) else None for interval in x_train['SkinThickness_level']])
del x_train['SkinThickness_level']

# Binning 'Insulin' into 4 bins and smoothing with midpoints
x_train['Insulin_level'] = pd.qcut(x_train['Insulin'], q=6, duplicates='drop')
x_train['Insulin'] = pd.Series([interval.mid if pd.notnull(interval) else None for interval in x_train['Insulin_level']])
del x_train['Insulin_level']
print(x_train)

#replace x_test from 0 to nan
x_test['SkinThickness'] = x_test['SkinThickness'].replace(0, np.nan)
x_test['Insulin'] = x_test['Insulin'].replace(0, np.nan)

#Smooth Noises using Binning
x_test['Pregnancies_level'] = pd.qcut(x_test['Pregnancies'], q=3)
x_test['Pregnancies'] = pd.Series ([interval.mid for interval in x_test['Pregnancies_level']])
del x_test['Pregnancies_level']

#detect outlier in x_test for SkinThickness
q3, q1 = np.nanpercentile(x_test['SkinThickness'], [75, 25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
x_test.loc[(test_df['SkinThickness'] < lower_band) | (x_test['SkinThickness'] > upper_band) , 'SkinThickness'] = None

#detect outlier in x_test for Insulin
q3, q1 = np.nanpercentile(x_test['Insulin'], [75, 25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
x_test.loc[(x_test['Insulin'] < lower_band) | (x_test['Insulin'] > upper_band) , 'Insulin'] = None

#MICE in x_test
imputer = IterativeImputer(max_iter=10,random_state=0)
x_test_imputed = imputer.fit_transform(x_test)
x_test = pd.DataFrame(x_test_imputed, columns= x_test.columns)
x_train.to_csv('x_train01.csv', index=False)

#Create a multiple linear regression
model = LinearRegression()
model.fit(x_train, y_train)
print('slope=', model.coef_)
print('intercept=', model.intercept_)

#test_pred_y = model.coef_ * test_x + model.intercept_
train_pred = model.predict(x_train)
train_r2 = r2_score(y_train, train_pred)
print('train r2 =', train_r2)
test_pred = model.predict(x_test)
test_df['BloodPressure'] = test_pred

##KNN models from 1 to 19

#define features and outcome fot Regression
y1_train = train_df.loc[:, train_df.columns == 'Outcome' ].values.ravel()
x1_train = train_df.loc[:, train_df.columns != 'Outcome']
y1_test = test_df.loc[:, test_df.columns == 'Outcome' ].values.ravel()
x1_test = test_df.loc[:, test_df.columns != 'Outcome']
best_k = 1
best_accuracy = 0
k = {}
accuracy = {}
for i in range(1 , 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x1_train, y1_train)
    test_pred_y = knn.predict(x1_test)
    accuracy[i] = accuracy_score(y1_test, test_pred_y)
    k[i] = i
    plt.plot(k[i],accuracy[i], marker='o')

    if accuracy[i] > best_accuracy:
        best_accuracy = accuracy[i]
        best_k = i
    print(f'k={i}, Accuracy: {accuracy[i]:.4f}')

# Scatter chart
plt.title("KNN models from 1 to 19")
plt.xlabel('KNN')
plt.ylabel('Accuracy')
plt.show()
print(f'The best k is {best_k} with an accuracy of {best_accuracy:.4f}')

