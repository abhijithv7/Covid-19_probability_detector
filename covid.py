import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import pickle


covid_data = pd.read_csv("data.csv")

train_data, test_data = train_test_split(covid_data, test_size=0.3, random_state=42)

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3, random_state = 42)
for train_index, test_index in split.split(covid_data, covid_data['bodyPain']):
    strat_train_data = covid_data.loc[train_index]
    strat_test_data = covid_data.loc[test_index]

x_train=strat_train_data[['fever','bodyPain','age','runnyNose','diffBreathe']].to_numpy()
x_test=strat_test_data[['fever','bodyPain','age','runnyNose','diffBreathe']].to_numpy()
y_train=strat_train_data[['infectionProb']].to_numpy().reshape(2136,)
y_test=strat_test_data[['infectionProb']].to_numpy().reshape(916,)

clf = LogisticRegression()
clf.fit(x_train,y_train)

file = open('model.pkl','wb')
pickle.dump(clf, file)

file.close()