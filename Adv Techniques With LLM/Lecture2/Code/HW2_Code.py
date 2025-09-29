import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt



# Import the weather data
df = pd.read_csv('weather.csv')
# print(df)

# split the features and label for data
feat_df = df.loc[:, df.columns != 'RainTomorrow']
label_df = df.loc[:, df.columns == 'RainTomorrow']

# Split features and label
X = df.loc[:, df.columns != "RainTomorrow"]
y = df.loc[:, df.columns == "RainTomorrow"].astype(int)

# Simple z-score normalization on all numeric columns (same as code 2 style)
num_cols = X.select_dtypes(include=[np.number]).columns
X_norm = X.copy()
X_norm[num_cols] = (X[num_cols] - X[num_cols].mean()) / X[num_cols].std()

# Train/test split 75/25
x_train, x_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.25
)

# Define the One hidden layer with 8 units
def hidden_layer(i, j):
    nn = Sequential()
    if i:
        nn.add(Dense(units=i, activation='relu'))
    if j:
        nn.add(Dense(units=j, activation='relu'))
    nn.add(Dense(units=1, activation='sigmoid'))
    return nn

# Calculate accuracy of both models on test data
def nn_para(nn, lr, size, epochs, x_train, y_train):
    nn.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(x_train, y_train, batch_size=size, epochs=epochs)

arr_layer= [(8, 0), (8, 5)]
arr_nn = [[0.0001, 32, 10], [0.01, 4, 30]]
ans = []
for i, j in arr_layer:
    for lr, size, epoch in arr_nn:
        nn = hidden_layer(i, j)
        nn_para(nn, lr, size, epoch, x_train, y_train)
        loss, accuracy = nn.evaluate(x_test, y_test)
        ans.append((loss, accuracy))
print(ans)

losses = [x[0] for x in ans]
accs = [x[1] for x in ans]
x = range(1, len(ans) + 1)
plt.figure(figsize=(10, 7))
plt.plot(x, losses, marker="s", label="Loss")
plt.plot(x, accs, marker="o", label="Accuracy")
plt.xlabel("Experiment Index")
plt.ylabel("Value")
plt.title("Loss & Accuracy in Weather")
plt.show()
