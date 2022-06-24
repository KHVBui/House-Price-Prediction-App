from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) # load the model

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Grabs the input values and uses them to make predictions"""
    rooms = int(request.form["rooms"])
    distance = int(request.form["distance"])
    
    prediction = model.predict([[rooms, distance]]) # returns a list
    output = round(prediction[0], 2) # pick only first element from list

    return render_template('index.html',
            prediction_text = f'A house with {rooms} rooms and located '
                              f'{distance} km from the city center has a '
                              f'value of ${output}K')

if __name__ == "__main__":
    app.run()
