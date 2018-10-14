"""
    Our class containing our model and method to fit it and evaluate it
"""

import numpy as np
import matplotlib.pyplot as plt

class linear_regression_model():
    def __init__(self,):
        pass

    def fit(self, x_train, y_train):
        # Computing matrix M such as y = M * x
        transpose = np.transpose(x_train)
        self.matrix = np.nan_to_num(np.dot(transpose, x_train))
        self.matrix = np.nan_to_num(np.linalg.inv(self.matrix))
        self.matrix = np.nan_to_num(np.dot(self.matrix, transpose))
        self.matrix = np.nan_to_num(np.dot(self.matrix, y_train))
        print('Our model is Y = M*X with ')
        print('M = '+str(self.matrix))
        print(' ')

    def predict(self, X):
        # Computing prediction
        return np.dot(X, self.matrix)

    def compute_TPR_FPR(self, Y_true, Y_predicted):
        # Compute True Positive Rate and False Positive Rate
        treshold = [i*0.001 for i in range(1, 1000, 1)] # All tested treshold
        true_positive_rate = []
        false_positive_rate = []
        for temp_treshold in treshold:
            # temp_class_prediction will be our prediction for a given treshold
            temp_class_prediction = np.array(Y_predicted)
            indexes_1 = np.where(temp_class_prediction >= temp_treshold)[0]
            indexes_0 = np.where(temp_class_prediction < temp_treshold)[0]

            temp_class_prediction[indexes_1] = 1
            temp_class_prediction[indexes_0] = 0

            true_positive = len([1 for j, k in enumerate(temp_class_prediction) if (Y_true[j] == 1 and k == 1)])
            false_positive = len([1 for j, k in enumerate(temp_class_prediction) if (Y_true[j] == 0 and k == 1)])
            true_negative = len([1 for j, k in enumerate(temp_class_prediction) if (Y_true[j] == 0 and k == 0)])
            false_negative = len([1 for j, k in enumerate(temp_class_prediction) if (Y_true[j] == 1 and k == 0)])

            true_positive_rate.append(true_positive/(true_positive+false_negative))
            false_positive_rate.append(false_positive/(false_positive+true_negative))

        return true_positive_rate, false_positive_rate, treshold

    def show_roc(self, Y_test_true, Y_test_predicted, Y_train_true, Y_train_predicted):
        # Show ROC Curb, can be use to choose treshold
        True_Positive_Rate_train, False_Positive_Rate_train, treshold = self.compute_TPR_FPR(Y_train_true, Y_train_predicted)
        True_Positive_Rate_test, False_Positive_Rate_test, treshold = self.compute_TPR_FPR(Y_test_true, Y_test_predicted)
        y_x_line = [0.1* i for i in range(6)]
        fig, ax = plt.subplots()
        ax.plot(False_Positive_Rate_train, True_Positive_Rate_train)
        ax.plot(False_Positive_Rate_test, True_Positive_Rate_test)
        ax.plot(y_x_line, y_x_line)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(['ROC Curb Train', 'ROC Curb Test', 'y=x'])
        plt.show()

    def compute_tresholds(self, Y_test_true, Y_test_predicted, Y_train_true, Y_train_predicted, verbose=False):
        True_Positive_Rate_train, False_Positive_Rate_train, treshold = self.compute_TPR_FPR(Y_train_true, Y_train_predicted)
        True_Positive_Rate_test, False_Positive_Rate_test, treshold = self.compute_TPR_FPR(Y_test_true, Y_test_predicted)

        # Compute Youden J Stats (see https://en.wikipedia.org/wiki/Youden%27s_J_statistic) for all treshold.
        Youden_J_train = []
        for i in range(len(True_Positive_Rate_train)):
            Youden_J_train.append(True_Positive_Rate_train[i] - False_Positive_Rate_train[i])
        Youden_J_test = []
        for i in range(len(True_Positive_Rate_test)):
            Youden_J_test.append(True_Positive_Rate_test[i] - False_Positive_Rate_test[i])

        if verbose:
            fig, ax = plt.subplots()
            ax.plot(treshold, Youden_J_train)
            ax.plot(treshold, Youden_J_test)
            ax.set_xlabel('Treeshold')
            ax.set_ylabel('Youden J stat')
            ax.legend(['Train', 'Test'])
            plt.show()

        # Youden J Stats goes from 0 to 1 (1 being the best and meaning 0 true negative and false positive)
        # We select the treshold maximising Youden J Stats
        max_Youden_J_train = np.argmax(np.asarray(Youden_J_train))
        treshold_max_Youden_J_train = treshold[max_Youden_J_train]

        # Now looking for theeshold corresponding to False Positive Rate < 0.001
        # False_Positive_Rate_train is sort decreasingly
        index = 0
        for i, j in enumerate(False_Positive_Rate_train):
            if j < 0.001:
                index = i
                break

        return treshold_max_Youden_J_train, treshold[index]

    def evaluate_result(self, Y_true, Y_predicted, treshold=0.5, verbose=False):
        # Evalute our model : MSE, Precision, Recall for spam and message, F1 score

        print('Using %0.2f as treshold' % treshold)

        Y_predicted_class = np.array(Y_predicted)
        indexes_1 = np.where(Y_predicted >= treshold)[0]
        indexes_0 = np.where(Y_predicted < treshold)[0]
        Y_predicted_class[indexes_1] = 1
        Y_predicted_class[indexes_0] = 0

        # Compute Precision, Recall and F1 score
        precision = 0
        for i, j in enumerate(Y_predicted_class):
            if Y_true[i] == j:
                precision += 1
        precision = 100*precision/len(Y_predicted_class)

        recal_spam = 0
        for i, j in enumerate(Y_predicted_class):
            if Y_true[i] == j and j == 1:
                recal_spam += 1
        recal_spam = 100*recal_spam/len([i for i in Y_true if i == 1])

        recal_msg = 0
        for i, j in enumerate(Y_predicted_class):
            if Y_true[i] == j and j == 0:
                recal_msg += 1
        recal_msg = 100*recal_msg/len([i for i in Y_true if i == 0])

        # Show our result
        if verbose:
            print('MSE = '+str(round(np.mean(Y_true - Y_predicted), 4)))
        print('Precision is %.2f%%' % precision)
        print('Spamm Recall is %.2f%%' % recal_spam)
        if verbose:
            print('Message Recall is %.2f%%' % recal_msg)
        print('F1 score of spam is %.2f%%' % (2*precision*recal_spam/(recal_spam+precision)))
        if verbose:
            print('F1 score of Message is %.2f%%' % (2*precision*recal_msg/(recal_msg+precision)))
        print(' ')

        return [precision, recal_spam, recal_msg, (2*precision*recal_spam/(recal_spam+precision))]
