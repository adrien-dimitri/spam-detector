import string
from random import sample
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import math

# an example of how to use the class is given in main.py

class SpamDetector():
    def __init__(self):
        self.preprocessed = None
        self.p_spam = None
        self.p_ham = None
        self.vocabulary = None
        self.features = None
        self.results = None


    def preprocess(self, corpus):
        # the stopwords were chosen specifically for an sms spam detection task 
        stop_words = [
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'you', 'your', 'u', 'ur', 'r', 'm', 'im', 'd', 'dont', 'cant', 'wont', '2', '4', 'b', 'c', 'd', 
        'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'n', 'o', 'p', 'q', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'pls', 'plz', 'thx', 'thank', 
        'thanks', 'ok', 'okay', 'lol', 'gt', 'br', 'amp'
        ]

        labels_list = []
        proprocessed_sms = []
        
        for sms in corpus:

            label, sms = sms.split('\t')

            labels_list.append(label)

            sms = sms.lower()

            remove_punc = str.maketrans("", "", string.punctuation)

            sms = sms.translate(remove_punc)

            clean_sms = sms.split()
            
            clean_sms = [word for word in clean_sms if word not in stop_words]
            clean_sms = ['0_short_number' if word.isnumeric() and len(word)<5 else '1_long_number' if word.isnumeric() and len(word)>=5 else word for word in clean_sms ] # change all integer values to 0
            

            proprocessed_sms.append(clean_sms)


            sms_df = pd.DataFrame({
                'LABEL': labels_list,
                'SMS': proprocessed_sms
            })


        self.preprocessed = sms_df

        return sms_df
    
    # show the percentage of spam and ham in the dataset
    def show_valuecounts(self):
        print(self.preprocessed['LABEL'].value_counts(normalize=True))
    
    
    def train_test_split(self, ratio):
        # Randomize the dataset
        data_randomized = self.preprocessed.sample(frac=1)

        # Calculate index for split
        training_test_index = round(len(data_randomized) * ratio)

        # Split into training and test sets
        self.training_set = data_randomized[:training_test_index].reset_index(drop=True)
        self.testing_set = data_randomized[training_test_index:].reset_index(drop=True)

        return self.training_set, self.testing_set
    

    def build_vocabulary(self):
        vocab = set()
        for _, row in self.training_set.iterrows():
            vocab.update(row['SMS'])

        self.vocabulary = vocab
        return vocab
    
    # 2 different methods have been implements to perform feature extraction
    # 1. Bag-of-Words, 2. Term Frequency-Inverse Document Frequency
    # choose bow for 1. and tfidf for 2.
    def extract_features(self, mode="bow"):
        
        sms_count = len(self.training_set)

        if mode=='bow':
            word_counts_per_sms = {unique_word: [0] * len(self.training_set['SMS']) for unique_word in self.vocabulary}

            for index, sms in enumerate(self.training_set['SMS']):
                for word in sms:
                    word_counts_per_sms[word][index] += 1


                if index % 200 == 0:
                    progress = int((index/sms_count)*100)
                    print(f"[{(progress//5*'|')}{((20-progress//5)*' ')}]", end="")
                    print(f"    {progress}% completed :- {index} of {sms_count} sms scanned...")

            print('SCAN COMPLETE')
            
            bag_of_words_df = pd.DataFrame(word_counts_per_sms)
            bag_of_words_df = pd.concat((self.training_set['LABEL'], self.training_set['SMS'], bag_of_words_df), axis=1)

            self.features = bag_of_words_df
            return bag_of_words_df

        if mode=='tfidf':

            tf_idf_matrix = {unique_word: [0] * len(self.training_set['SMS']) for unique_word in self.vocabulary}

            for index, sms in enumerate(self.training_set['SMS']):
                sms_length = len(sms)
                for word in sms:
                    # tf
                    word_count = sms.count(word)
                    tf_idf_matrix[word][index] = word_count/sms_length

                    # tfidf
                    num_sms_containing_word = sum(1 for sms in self.training_set['SMS'] if word in sms)
                    tf_idf_matrix[word][index] *= math.log(sms_count/(num_sms_containing_word+1))
                    
                
                if index % 200 == 0:
                    progress = int((index/sms_count)*100)
                    print(f"[{(progress//5*'|')}{((20-progress//5)*' ')}]", end="")
                    print(f"    {progress}% completed :- {index} of {sms_count} sms scanned...")

            print('SCAN COMPLETE')

            tf_idf_dataframe = pd.DataFrame(tf_idf_matrix)
            tf_idf_dataframe = pd.concat((self.training_set['LABEL'], self.training_set['SMS'], tf_idf_dataframe), axis=1)

            
            self.features = tf_idf_dataframe
            return tf_idf_dataframe
        
        else:
            print('Choose between "bow" and "tfidf".')
            return None
        
    def train(self):
        spam_sms = self.features[self.features['LABEL'] == 'spam']
        ham_sms = self.features[self.features['LABEL'] == 'ham']

        p_spam = len(spam_sms) / len(self.features)
        p_ham = len(ham_sms) / len(self.features)

        n_words_per_spam_message = spam_sms['SMS'].apply(len)
        n_spam = n_words_per_spam_message.sum()

        n_words_per_ham_message = ham_sms['SMS'].apply(len)
        n_ham = n_words_per_ham_message.sum()

        n_vocabulary = len(self.vocabulary)

        # Laplace smoothing
        alpha = 1

        parameters_spam = {unique_word:0 for unique_word in self.vocabulary}
        parameters_ham = {unique_word:0 for unique_word in self.vocabulary}

        for word in self.vocabulary:
            n_word_given_spam = spam_sms[word].sum() 
            p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + n_vocabulary)
            parameters_spam[word] = p_word_given_spam

            n_word_given_ham = ham_sms[word].sum() 
            p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + n_vocabulary)
            parameters_ham[word] = p_word_given_ham


        self.p_ham = p_ham
        self.p_spam = p_spam
        self.parameters_ham = parameters_ham
        self.parameters_spam = parameters_spam
        
        return p_spam, p_ham, parameters_spam, parameters_ham

    # use auto train to perform all necessary steps for training automatically
    def auto_train(self, mode="bow"):
        self.build_vocabulary()
        self.extract_features(mode)
        self.train()

    
    def evaluate(self):
        correct = 0
        total = self.test_results.shape[0]

        for row in self.test_results.iterrows():
            row = row[1]
            if row['LABEL'] == row['PREDICTED']:
                correct += 1

        print('Correct:', correct)
        print('Incorrect:', total - correct)
        print('Accuracy:', f"{correct/total*100:1.2f}%")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        correct_spam = 0
        for i, row in self.test_results.iterrows():
            if row['LABEL'] == row['PREDICTED'] == 'spam':
                correct_spam += 1

        total_pred_spam = (self.test_results['PREDICTED'] == 'spam').sum()

        precision = correct_spam/total_pred_spam

        print('Total Spam predictions:', total_pred_spam)
        print('Correct Spam predictions:', correct_spam)
        print("Precision: ", f"{precision*100:1.2f}%")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        correct_spam = 0
        for i, row in self.test_results.iterrows():
            if row['LABEL'] == row['PREDICTED'] == 'spam':
                correct_spam += 1

        total_spam = (self.test_results['LABEL'] == 'spam').sum()

        recall = correct_spam/total_spam

        print('Total true Spam: ', total_spam)
        print('Correct Spam predictions:', correct_spam)
        print("Recall: ", f"{recall*100:1.2f}%")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        f1 = 2 * (precision * recall) / (precision + recall)
        print('F1 score: ', f1)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # display the confusion matrix
    def display_confusionmat(self):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i, row in self.test_results.iterrows():
            if row['LABEL'] == row['PREDICTED'] == 'spam':
                TP += 1
            if row['LABEL'] == row['PREDICTED'] == 'ham':
                TN += 1
            if row['PREDICTED'] == 'spam' and row['LABEL'] == 'ham':
                FP += 1
            if row['PREDICTED'] == 'ham' and row['LABEL'] == 'spam':
                FN += 1


        cm = [[TP, FP],
            [FN, TN]]

        df_cm = pd.DataFrame(cm, index=['pred spam',  'pred ham'], columns=['true spam', 'true ham'])

        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size

        plt.show()
    
    # testing will automatically calculate accuracy, precision, recall, and F1 score
    def test(self):
        y_pred = []
        for sms in self.testing_set['SMS']:
            spam_prob = np.log(self.p_spam)
            ham_prob = np.log(self.p_ham)
            for _, word in enumerate(self.vocabulary):
                if word in sms:
                    spam_prob += np.log(self.parameters_spam[word])
                    ham_prob += np.log(self.parameters_ham[word])

            if spam_prob > ham_prob:
                y_pred.append('spam')
            else:
                y_pred.append('ham')

        test_df = pd.concat((pd.Series(y_pred).rename('PREDICTED'), self.testing_set), axis=1)

        self.test_results = test_df

        self.evaluate()
        
        return test_df