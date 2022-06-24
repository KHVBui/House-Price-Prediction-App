import pandas as pd
from sklearn import linear_model
import pickle

df = pd.read_csv('ml-model/prices.csv')

y = df['Value'] # dependent variable
x = df[['Rooms', 'Distance']] # independent variables

# Not focusing on training and validating the model results for now
lin_model = linear_model.LinearRegression()
lin_model.fit(x, y)

# Test the output of the model
input_rooms = 15
input_distance = 61
print(f'rooms: {input_rooms}')
print(f'distance (km): {input_distance}')
print(lin_model.predict([[input_rooms, input_distance]])) 
print(f'score: {lin_model.score(x, y)}')

# Save the model using pickle
pickle.dump(lin_model, open('model.pkl', 'wb')) # save the model in a pickle file
