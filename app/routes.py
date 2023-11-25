from flask import render_template, request
from app import app
from app.ml_model import train_and_evaluate

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    classifier_name = request.form['classifier']
    param1 = float(request.form['param1'])
    param2 = float(request.form['param2'])
    param3 = float(request.form['param3'])

    results = train_and_evaluate(classifier_name, param1, param2, param3)

    return render_template('result.html', results=results)


