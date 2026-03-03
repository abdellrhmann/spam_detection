from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return "Spam Detection Model is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    data_vec = vectorizer.transform([data])
    prediction = model.predict(data_vec)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)