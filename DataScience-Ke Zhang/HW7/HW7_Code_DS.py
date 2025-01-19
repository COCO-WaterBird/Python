import pandas as pd
from matplotlib import pyplot as plt
from pydtmc import MarkovChain, plot_graph, plot_redistributions, HiddenMarkovModel, plot_sequence
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from mlxtend.evaluate import accuracy_score


def getNumber(df, column1, column2):
    numbers = df[(df[column1] == 1) & (df[column2] == 1)].shape[0]
    return numbers


# Input Dataset
org_df = pd.read_csv("amr_ds.csv")

#Split dataset between feature and label
label_df = org_df["Not_MDR"]
feat_df = org_df.loc[:, org_df.columns != "Not_MDR"]

#Seperate label_df and feat_df to use 75% for training and 25% for test
train_x, test_x, train_y, test_y = train_test_split(feat_df, label_df, test_size=0.25, random_state=42)

#Create Naive model Bayes
nb_model = GaussianNB()
nb_model.fit(train_x, train_y)
pred_y = nb_model.predict(test_x)

# Calculate accuracy according to test data
accuracy_test = accuracy_score(test_y, pred_y)
print(accuracy_test)

# Calculate number of records
amp_pen = getNumber(org_df, 'Ampicillin', 'Penicillin')
amp_nmdr = getNumber(org_df, 'Ampicillin', 'Not_MDR')
pen_nmdr = getNumber(org_df, 'Penicillin', 'Not_MDR')

#Create states
states = ["Ampicillin","Penicillin","Not_MDR"]

#Create Transition Matrix
transition_matrix = [
    [0, amp_pen/(amp_pen+amp_nmdr), amp_nmdr/(amp_pen+amp_nmdr)],
    [amp_pen/(pen_nmdr+amp_pen), 0, pen_nmdr/(amp_pen+pen_nmdr)],
    [amp_nmdr/(amp_nmdr+pen_nmdr), pen_nmdr/(pen_nmdr+amp_nmdr), 0]]

#Create Markov Chain
mc = MarkovChain(transition_matrix, states)
print(mc)

#Show stationary state
print(mc.steady_states)

# Visualize results
plt.ion()
plt.figure(figsize=(10, 10))
plot_graph(mc)
plot_redistributions(mc, 100, plot_type='projection', initial_status='Ampicillin')
plt.show()

# Hidden Markov
hidden_states = ["Ampicillin", "Penicillin", "Not_MDR"]
observation_symbols = ['Infection', 'No Infection']
emission_matrix = [[0.4, 0.6], # AMP
                   [0.3, 0.7], # PEN
                   [0.8, 0.2]] # NMDR

# Create Hidden Markov Model
hmm = HiddenMarkovModel(transition_matrix, emission_matrix, hidden_states, observation_symbols)

# Visualize results
plt.ion()
plt.figure(figsize=(10, 10))
plot_graph(hmm)
plot_sequence(hmm, steps=10, plot_type='matrix')

# Predict hidden states
lp, most_probable_states = hmm.predict(prediction_type='viterbi', symbols=['Infection', 'No Infection', 'Infection'])
print(most_probable_states)




