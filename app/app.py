from flask import Flask, render_template, request, jsonify
from app.sms_spam_detector import SpamDetector

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    train_percentage = int(request.form['train_percentage'])
    feature_extraction = request.form['feature_extraction']

    corpus = open('./data/smsspamcollection/SMSSpamCollection', 'r').readlines()
    spam_detector = SpamDetector()
    spam_detector.preprocess(corpus)
    spam_detector.train_test_split(train_percentage)
    spam_detector.auto_train(feature_extraction)

    return jsonify({'message': 'Training completed successfully'})

if __name__ == '__main__':
    app.run(debug=True)