from flask import Flask, render_template, request, jsonify, g, session
from app.sms_spam_detector import SpamDetector
import random

app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    train_percentage = int(request.form['train_percentage'])/100
    feature_extraction = request.form['feature_extraction']

    corpus = open('./data/smsspamcollection/SMSSpamCollection', 'r').readlines()
    spam_detector = SpamDetector()
    spam_detector.preprocess(corpus)
    spam_detector.train_test_split(train_percentage)
    spam_detector.auto_train(feature_extraction)

    test_results = spam_detector.test()
    session['test_results'] = test_results

    return jsonify({'message': 'Training completed successfully'})

@app.route('/evaluateSamples', methods=['POST', 'GET'])
def evaluateSamples():
    if request.method == 'POST':
        # Handle POST request
        evaluation_results = session.get('test_results', [])
        print("Evaluation results (POST):", evaluation_results)

        return jsonify({'evaluation_results': evaluation_results})

    else:
        # Handle GET request
        evaluation_results = session.get('test_results', [])
        print("Evaluation results (GET):", evaluation_results)

        return jsonify({'evaluation_results': evaluation_results})

if __name__ == '__main__':
    app.run(debug=True)