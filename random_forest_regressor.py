

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

combined_data = pd.read_csv('D:/cleaned_data.csv')

combined_data.head()

X = combined_data.iloc[:, :5].values
y = combined_data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

"""## RANDOM FOREST REGRESSOR"""

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train, y_train)




# Saving model to disk
pickle.dump(forest, open('randomForestRegressor666.pkl','wb'))
