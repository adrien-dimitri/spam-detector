Spam detector
=====

Spam Detector is a web-based application that allows users to train a machine learning model to detect spam messages. It supports two feature extraction methods: Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).

## Table of Contents
* [General info](#general-info)
* [Deployment on PythonAnywhere](#deployment-on-pythonanywhere)
* [Data](#data)
* [How it works](#how-it-works)
* [Technologies](#technologies)
* [Requirements](#requirements)
* [Setup](#setup)
* [Deployment locally](#deployment)
* [Contributing](#contributing)

## General Info
This project is an SMS spam detector implemented as a Flask web application. It provides a user-friendly interface to train and evaluate a machine learning model for spam detection.

## Deployment on PythonAnywhere

This application is deployed on [PythonAnywhere](https://www.pythonanywhere.com/), a cloud-based Python hosting service. You can access the live application at [adriendimitri.pythonanywhere.com](https://adriendimitri.pythonanywhere.com/).

## Data
The available data is the SMS Spam Collection Dataset from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

The SMS Spam Collection v.1 (hereafter the corpus) is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, all classified as either *ham* (legitimate) or *spam*. The first word of each text is the classification, followed by a space, and the rest is the SMS itself.

The SMS Spam Collection v.1 (text file: smsspamcollection) has a total of 4,827 SMS legitimate messages (86.6%) and a total of 747 (13.4%) spam messages.

## How it Works
For features extraction, you may choose either **Bag-of-Words** (BoW) or **Term Frequency - Inverse Document Frequency** (TF-IDF), with both giving good results. 

A **Naive Bayes Classifier** is used as the core during training which saves all the parameters of spam and ham sms messages based on the features extracted prior.

## Technologies
* Python 3.10.12

## Requirements
* [numpy](https://numpy.org/) 1.26.3
* [pandas](https://pandas.pydata.org/) 2.2.0
* [Flask](https://flask.palletsprojects.com/) (latest version)

# Setup
1. Clone the repository
2. Navigate to the project directory:

    ```bash
    $ cd spam-detector
    ```

3. Create a virtual environment (optional but recommended):
   
    ```bash
    $ python -m venv .venv
    ```

4. Activate the virtual environment:

    - On Windows:

    ```bash
    $ .venv\Scripts\activate
    ```

    - On macOS and Linux:

    ```bash
    $ source .venv/bin/activate
    ```

5. Install the required packages using the command below:

    ```bash
    $ pip install -r requirements.txt
    ```

6. Run the Flask application:

    ```bash
    $ python run.py
    ```

**Note:** It's recommended to use a virtual environment to isolate the project dependencies. If you choose not to use a virtual environment, make sure to adapt the installation command (`pip install -r requirements.txt`) accordingly.

## Deployment
Access the application by navigating to [http://localhost:5000](http://localhost:5000) in your web browser.

## Contributing
If you'd like to contribute to the development of the SMS Spam Detector, please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).