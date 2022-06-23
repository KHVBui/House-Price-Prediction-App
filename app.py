from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) # load the model

@app.route("/")
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
