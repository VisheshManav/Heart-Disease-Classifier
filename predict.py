import pickle
from flask import Flask, render_template, request

model_file = 'model_lr.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('heartD')

@app.route('/')
def home_form():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def predict():

    patient = {
        "age":      request.form['age'],
        "sex":      request.form['sex'], 
        "cp":       request.form['cp'],
        "trestbps": request.form['trestbps'],
        "chol":     request.form['chol'],
        "fbs":      request.form['fbs'],
        "restecg":  request.form['restecg'],
        "thalach":  request.form['thalach'],
        "exang":    request.form['exang'],
        "oldpeak":  request.form['oldpeak'],
        "slope":    request.form['slope'],
        "ca":       request.form['ca'],
        "thal":     request.form['thal']
    }

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    disease_status = y_pred > 0.5

    return render_template('prediction.html', pred=y_pred, disease=disease_status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)