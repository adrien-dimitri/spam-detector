from flask import Flask, render_template, request, jsonify, g, session
from app.sms_spam_detector import SpamDetector
import random

app = Flask(__name__)
app.secret_key = 'secret_key'

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
    test_samples = random.sample(corpus, 6)
    test_samples = [sample.split('\t')[1] for sample in test_samples]
    session['test_samples'] = test_samples

    return jsonify({'message': 'Training completed successfully'})

@app.route('/classify', methods=['GET'])
def classify():
    test_samples = session.get('test_samples', [])
    test_sample_message = test_samples.pop()

    return jsonify({'sample_text': test_sample_message})

# get a random message to display on the web page
@app.route('/get_random_text', methods=['GET'])
def get_random_text():
    test_samples = session.get('test_samples', [])
    if test_samples:
        test_sample_message = test_samples.pop()
        session['test_samples'] = test_samples
        return jsonify({'sample_text': test_sample_message})
    else:
        return jsonify({'error': 'No more samples'})
    

if __name__ == '__main__':
    app.run(debug=True)