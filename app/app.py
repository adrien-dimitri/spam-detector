from flask import Flask, render_template, request, jsonify
from app.sms_spam_detector import SpamDetector
import os

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

    home_dir = os.path.expanduser('~')
    print(os.path.expanduser("~"))

    corpus = open(f'{home_dir}/spam-detector/data/smsspamcollection/SMSSpamCollection', 'r').readlines()
    spam_detector = SpamDetector()
    spam_detector.preprocess(corpus)
    spam_detector.train_test_split(train_percentage)
    spam_detector.auto_train(feature_extraction)

    test_results = spam_detector.test()

    return jsonify({'test_results': test_results})

@app.route('/evaluateSamples', methods=['POST', 'GET'])
def evaluateSamples():
    if request.method == 'POST':

        return jsonify({'message': 'evaluation complete'})
if __name__ == '__main__':
    app.run(debug=False)