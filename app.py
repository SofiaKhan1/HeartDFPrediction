from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Heart Disease and Failure Prediction App"


@app.route('/predict', methods=['POST'])
# 'cp','thalach','slope','restecg','chol','trestbps','fbs','oldpeak'
# [0,108,1,0,250,160,1,1.5] 0
# [3,150,0,0,233,145,1,2.3] 1
def predict():
    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')

    input_query = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    result = model.predict(input_query)[0]

    return jsonify({'hearth_disease': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
