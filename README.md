SMS spam detector
=====

SMS spam detector using Python from scratch

## Table of Contents
* [General info](#general-info)
* [Data](#data)
* [How it works](#how-it-works)
* [Technologies](#technologies)
* [Requirements](#requirements)
* [Setup](#setup)

## General info
This project is an sms spam detector implemented using Python from scratch. 

It includes all necessary steps to prepare the data for training and testing such as preprocessing and feature extraction.

## Data
The available data is the SMS Spam Collection Dataset from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

The SMS Spam Collection v.1 (hereafter the corpus) is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, all classified as either *ham* (legitimate) or *spam*. The first word of each text is the classification, followed by a space, and the rest is the SMS itself.

The SMS Spam Collection v.1 (text file: smsspamcollection) has a total of 4,827 SMS legitimate messages (86.6%) and a total of 747 (13.4%) spam messages.

## How it works
For features extraction, you may choose either **Bag-of-Words** or **Term Frequency - Inverse Document Frequency**, with both giving good results. 

A **Naive Bayes Classifier** is used as the core during training which saves all the parameters of spam and ham sms messages based on the features extracted prior.

## Technologies
* Python 3.10.6

## Requirements
* [numpy](https://numpy.org/) 1.24.3 
* [pandas](https://pandas.pydata.org/) 2.0.1 
* [matplotlib](https://matplotlib.org/) 3.7.1 
* [seaborn](https://seaborn.pydata.org/) 0.12.2 

## Setup
1. Clone repository
2. Use the command below to install the packages according to the configuration file requirements.txt:

    `$ pip install -r requirements.txt`

3. Run [main.py](main.py)
