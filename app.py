
from flask import Flask, jsonify, request
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Create a simple linear regression model
model = LinearRegression()

# Training dataset (for instance, X = hours of study, Y = scores)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  # Study hours
Y = np.array([1, 3, 5, 7, 10, 10, 10, 10])  # Scores

# Train the model
model.fit(X, Y)

# Save the model using joblib
joblib.dump(model, 'modelo.pkl')

@app.route('/')
def hello_world():
    return "Hello World! This AI is ready to make predictions."

@app.route('/predict', methods=['GET'])
def predict():
    # Obtener el valor de entrada desde la URL
    horas_estudio = request.args.get('horas', type=float)

    if horas_estudio is None:
        return jsonify({"error": "Please provide a valid value for 'hours'."}), 400

    # Load the model
    model = joblib.load('modelo.pkl')

    # Make the prediction
    prediccion = model.predict([[horas_estudio]])

    return jsonify({"Hours of study": horas_estudio, "Predicted score": prediccion[0]})

if __name__ == '__main__':
    app.run(debug=True)
