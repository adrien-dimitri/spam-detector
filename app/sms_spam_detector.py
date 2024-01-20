import string
from collections import Counter
import pandas as pd
import numpy as np
import math

class SpamDetector():
    def __init__(self):
        self.preprocessed = None
        self.p_spam = None
        self.p_ham = None
        self.vocabulary = None
        self.features = None

    def preprocess(self, corpus):
        stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'you', 'your', 'u', 'ur', 'r', 'm', 'im', 'd', 'dont', 'cant', 'wont', '2', '4', 'b', 'c', 'd', 
            'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'n', 'o', 'p', 'q', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'pls', 'plz', 'thx', 'thank', 
            'thanks', 'ok', 'okay', 'lol', 'gt', 'br', 'amp'
        ])

        labels_list = []
        proprocessed_sms = []

        for sms in corpus:
            label, sms_text = sms.split('\t')
            labels_list.append(label)

            sms_text = sms_text.lower()
            remove_punc = str.maketrans("", "", string.punctuation)
            sms_text = sms_text.translate(remove_punc)

            clean_sms = [
                '0_short_number' if word.isnumeric() and len(word) < 5 else
                '1_long_number' if word.isnumeric() and len(word) >= 5 else
                word for word in sms_text.split() if word not in stop_words
            ]

            proprocessed_sms.append(clean_sms)

        sms_df = pd.DataFrame({'LABEL': labels_list, 'SMS': proprocessed_sms})
        self.preprocessed = sms_df
        return sms_df

    def train_test_split(self, ratio):
        data_randomized = self.preprocessed.sample(frac=1)
        training_test_index = round(len(data_randomized) * ratio)
        self.training_set = data_randomized[:training_test_index].reset_index(drop=True)
        self.testing_set = data_randomized[training_test_index:].reset_index(drop=True)
        return self.training_set, self.testing_set

    def build_vocabulary(self):
        vocab = set()
        for _, row in self.training_set.iterrows():
            vocab.update(row['SMS'])
        self.vocabulary = vocab
        return vocab

    def extract_features(self, mode="bow"):
        sms_count = len(self.training_set)

        if mode == 'bow':
            word_counts_per_sms = {unique_word: [0] * len(self.training_set['SMS']) for unique_word in self.vocabulary}

            for index, sms in enumerate(self.training_set['SMS']):
                word_counts = Counter(sms)
                for word, count in word_counts.items():
                    word_counts_per_sms[word][index] += count

            bag_of_words_df = pd.DataFrame(word_counts_per_sms)
            self.features = pd.concat((self.training_set['LABEL'], self.training_set['SMS'], bag_of_words_df), axis=1)
            return bag_of_words_df

        elif mode == 'tfidf':
            tf_idf_matrix = {unique_word: [0] * len(self.training_set['SMS']) for unique_word in self.vocabulary}
            idf_dict = {word: 0 for word in self.vocabulary}

            for index, sms in enumerate(self.training_set['SMS']):
                sms_length = len(sms)
                word_counts = Counter(sms)

                for word, count in word_counts.items():
                    # tf
                    tf = count / sms_length
                    tf_idf_matrix[word][index] = tf

                    # idf
                    if idf_dict[word] == 0:
                        num_sms_containing_word = sum(1 for sms in self.training_set['SMS'] if word in sms)
                        idf_dict[word] = math.log(sms_count / (num_sms_containing_word + 1))

                    # tfidf
                    tf_idf_matrix[word][index] *= idf_dict[word]

            tf_idf_dataframe = pd.DataFrame(tf_idf_matrix)
            self.features = pd.concat((self.training_set['LABEL'], self.training_set['SMS'], tf_idf_dataframe), axis=1)

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

        alpha = 1  # Laplace smoothing

        parameters_spam = {unique_word: 0 for unique_word in self.vocabulary}
        parameters_ham = {unique_word: 0 for unique_word in self.vocabulary}

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

    def auto_train(self, mode="bow"):
        self.build_vocabulary()
        self.extract_features(mode)
        self.train()

    def evaluate(self):
        correct = (self.test_results['LABEL'] == self.test_results['PREDICTED']).sum()
        total = len(self.test_results)

        print('Correct:', correct)
        print('Incorrect:', total - correct)
        print('Accuracy:', f"{correct / total * 100:1.2f}%")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        correct_spam = ((self.test_results['LABEL'] == self.test_results['PREDICTED']) & (self.test_results['PREDICTED'] == 'spam')).sum()
        total_pred_spam = (self.test_results['PREDICTED'] == 'spam').sum()
        precision = correct_spam / total_pred_spam

        print('Total Spam predictions:', total_pred_spam)
        print('Correct Spam predictions:', correct_spam)
        print("Precision: ", f"{precision * 100:1.2f}%")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        correct_spam = ((self.test_results['LABEL'] == self.test_results['PREDICTED']) & (self.test_results['PREDICTED'] == 'spam')).sum()
        total_spam = (self.test_results['LABEL'] == 'spam').sum()
        recall = correct_spam / total_spam

        print('Total true Spam: ', total_spam)
        print('Correct Spam predictions:', correct_spam)
        print("Recall: ", f"{recall * 100:1.2f}%")
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        f1 = 2 * (precision * recall) / (precision + recall)
        print('F1 score: ', f1)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        return {
            'accuracy': correct / total * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1
        }

    def predict(self, sms):
        spam_prob = np.log(self.p_spam)
        ham_prob = np.log(self.p_ham)

        for word in sms:
            if word in self.vocabulary:
                spam_prob += np.log(self.parameters_spam[word])
                ham_prob += np.log(self.parameters_ham[word])

        return 'spam' if spam_prob > ham_prob else 'ham'


    def test(self):
        y_pred = [self.predict(sms) for sms in self.testing_set['SMS']]
        test_df = pd.concat((pd.Series(y_pred).rename('PREDICTED'), self.testing_set), axis=1)
        print(test_df.head())
        self.test_results = test_df
        results = self.evaluate()
        return results
