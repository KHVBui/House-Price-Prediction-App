import pandas as pd
from sklearn import linear_model
import pickle

df = pd.read_csv('ml-model/prices.csv')

y = df['Value'] # dependent variable
x = df[['Rooms', 'Distance']] # independent variables

# Not focusing on training and validating the model results for now
lin_model = linear_model.LinearRegression()
lin_model.fit(x, y)

lin_model.predict([[15, 61]])

# Save the model using pickle
pickle.dump(lin_model, open('model.pkl', 'wb'))
