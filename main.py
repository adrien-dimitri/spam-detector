import sms_spam_detector


corpus = open('/data/smsspamcollection/SMSSpamCollection', 'r').readlines()

spam_dectector = sms_spam_detector.SpamDetector()

spam_dectector.preprocess(corpus)

spam_dectector.train_test_split(0.8)

spam_dectector.auto_train()

spam_dectector.test()