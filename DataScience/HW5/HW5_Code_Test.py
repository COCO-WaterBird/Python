import pandas as pd
from mlxtend.frequent_patterns import fpgrowth,apriori, association_rules
import matplotlib.pyplot as plt

# Hyperpramater for min_sup, min_conf, min_lift
min_sups = [0.05, 0.1, 0.4]
min_confs = [0.70, 0.85, 0.95]
min_lifts = [1.1, 1.5, 4]

#Input data_frame
data_frame = pd.read_csv('amr_horse_ds.csv')

#Smooth Age for binning
data_frame['Age_binned'] = pd.qcut(data_frame['Age'], 2)
# data_frame['Age'] = pd.Series([interval.mid for interval in data_frame['Age_binned']], index=data_frame.index)
data_frame.drop(columns=['Age'], inplace=True)
# print(data_frame)

#Encoding data_frame
data_frame = pd.get_dummies(data_frame)

#Extract association rule
k = 0
best_min_sup = 0
best_min_conf = 0
best_min_lift = 0
num_rows = {}
best_num_rows = 0
i = 0
for min_sup in min_sups:
    for min_conf in min_confs:
        for min_lift in min_lifts:
            frequent_patterns_df = apriori(data_frame, min_support=min_sup, use_colnames=True)
            rules_df = association_rules(frequent_patterns_df, metric='confidence', min_threshold=min_conf)
            lift_rules_df = rules_df[rules_df['lift'] > min_lift]
            num_rows[k] = len(lift_rules_df)
            if num_rows[k] > 20 and num_rows[k] < 50:
                best_min_lift, best_min_sup, best_min_conf = min_lift, min_sup, min_conf
                best_num_rows = num_rows[k]
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(lift_rules_df['support'], lift_rules_df['confidence'], lift_rules_df['lift'], marker='o')
                ax.set_xlabel('Support')
                ax.set_ylabel('Confidence')
                ax.set_zlabel('Lift')
                plt.show()
                lift_rules_df.to_csv('lift_rules_df.csv')
                i = i + 1

print('best_min_sup =',best_min_sup, '\n'
      'best_min_conf =',best_min_conf, '\n'
      'best_min_lift =',best_min_lift, '\n'
      'best_num_rows =',best_num_rows
      )
    