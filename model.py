"""
    Our class containing our model and method to fit it and evaluate it
"""

import numpy as np
import matplotlib.pyplot as plt

class ML_model():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.X_train = x_train
        self.X_test = x_test
        self.Y_train = y_train
        self.Y_test = y_test

    def fit_linear_regression(self):
        # Computing matrix M such as y = M * x
        transpose = np.transpose(self.X_train)
        self.matrix = np.nan_to_num(np.dot(transpose, self.X_train))
        self.matrix = np.nan_to_num(np.linalg.inv(self.matrix))
        self.matrix = np.nan_to_num(np.dot(self.matrix, transpose))
        self.matrix = np.nan_to_num(np.dot(self.matrix, self.Y_train))
        print('Our model is Y = M*X with ')
        print('M = '+str(self.matrix))
        print(' ')

        # Computing prediction for train set and test set
        self.Y_train_predicted = np.dot(self.X_train, self.matrix)
        self.Y_test_predicted = np.dot(self.X_test, self.matrix)

    def Compute_TPR_FPR_train(self):
        # Compute True Positive Rate and False Positive Rate in order to choose our treeshold thanks to ROC curb
        treeshold = [i*0.001 for i in range(1, 1000, 1)] # The range is big to have a complete ROC curb
        true_positive_rate = []
        false_positive_rate = []
        for temp_treeshold in treeshold:
            # temp_class_prediction will be our prediction for a given treeshold
            temp_class_prediction = np.array(self.Y_train_predicted)
            indexes_1 = np.where(temp_class_prediction >= temp_treeshold)[0]
            indexes_0 = np.where(temp_class_prediction < temp_treeshold)[0]

            temp_class_prediction[indexes_1] = 1
            temp_class_prediction[indexes_0] = 0

            true_positive = len([1 for j, k in enumerate(temp_class_prediction) if (self.Y_train[j] == 1 and k == 1)])
            false_positive = len([1 for j, k in enumerate(temp_class_prediction) if (self.Y_train[j] == 0 and k == 1)])
            true_negative = len([1 for j, k in enumerate(temp_class_prediction) if (self.Y_train[j] == 0 and k == 0)])
            false_negative = len([1 for j, k in enumerate(temp_class_prediction) if (self.Y_train[j] == 1 and k == 0)])

            true_positive_rate.append(true_positive/(true_positive+false_negative))
            false_positive_rate.append(false_positive/(false_positive+true_negative))

        return true_positive_rate, false_positive_rate, treeshold

# TO DO : only one Compute_TPR_FPR
    def Compute_TPR_FPR_test(self):
        # Compute True Positive Rate and False Positive Rate in order to verify chosen treeshold on test set
        treeshold = [i*0.001 for i in range(1, 1000, 1)] # The range is big to have a complete ROC curb
        true_positive_rate = []
        false_positive_rate = []
        for temp_treeshold in treeshold:
            # temp_class_prediction will be our prediction for a given treeshold
            temp_class_prediction = np.array(self.Y_test_predicted)
            indexes_1 = np.where(temp_class_prediction >= temp_treeshold)[0]
            indexes_0 = np.where(temp_class_prediction < temp_treeshold)[0]

            temp_class_prediction[indexes_1] = 1
            temp_class_prediction[indexes_0] = 0

            true_positive = len([1 for j, k in enumerate(temp_class_prediction) if (self.Y_test[j] == 1 and k == 1)])
            false_positive = len([1 for j, k in enumerate(temp_class_prediction) if (self.Y_test[j] == 0 and k == 1)])
            true_negative = len([1 for j, k in enumerate(temp_class_prediction) if (self.Y_test[j] == 0 and k == 0)])
            false_negative = len([1 for j, k in enumerate(temp_class_prediction) if (self.Y_test[j] == 1 and k == 0)])

            true_positive_rate.append(true_positive/(true_positive+false_negative))
            false_positive_rate.append(false_positive/(false_positive+true_negative))

        return true_positive_rate, false_positive_rate, treeshold

    def show_roc(self):
        # Show ROC Curb, can be use to choose treeshold
        True_Positive_Rate_train, False_Positive_Rate_train, treeshold = self.Compute_TPR_FPR_train()
        True_Positive_Rate_test, False_Positive_Rate_test, treeshold = self.Compute_TPR_FPR_test()
        y_x_line = [0.1* i for i in range(6)]
        fig, ax = plt.subplots()
        ax.plot(False_Positive_Rate_train, True_Positive_Rate_train)
        ax.plot(False_Positive_Rate_test, True_Positive_Rate_test)
        ax.plot(y_x_line, y_x_line)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(['ROC Curb Train', 'ROC Curb Test', 'y=x'])
        plt.show()

    def compute_Youden_J_stats(self, verbose=False):
        # Compute Youden J Stats (see https://en.wikipedia.org/wiki/Youden%27s_J_statistic) for all treeshold.
        True_Positive_Rate_train, False_Positive_Rate_train, treeshold = self.Compute_TPR_FPR_train()
        True_Positive_Rate_test, False_Positive_Rate_test, treeshold = self.Compute_TPR_FPR_test()
        Youden_J_train = []
        for i in range(len(True_Positive_Rate_train)):
            Youden_J_train.append(True_Positive_Rate_train[i] - False_Positive_Rate_train[i])
        Youden_J_test = []
        for i in range(len(True_Positive_Rate_test)):
            Youden_J_test.append(True_Positive_Rate_test[i] - False_Positive_Rate_test[i])

        if verbose:
            fig, ax = plt.subplots()
            ax.plot(treeshold, Youden_J_train)
            ax.plot(treeshold, Youden_J_test)
            ax.set_xlabel('Treeshold')
            ax.set_ylabel('Youden J stat')
            ax.legend(['Train', 'Test'])
            plt.show()

        # Youden J Stats goes from 0 to 1 (1 being the best and meaning 0 true negative and false positive)
        # We select the treeshold maximising Youden J Stats
        max_Youden_J_train = np.argmax(np.asarray(Youden_J_train))
        treeshold_max_Youden_J_train = treeshold[max_Youden_J_train]

        return treeshold_max_Youden_J_train

    def evaluate_result(self, treeshold=0.5, verbose=False):
        # Evalute our model : MSE, Precision, Recall for spam and message, F1 score
        # Need self.Y_train_predicted and self.Y_test_predicted already set

        print('Using %0.2f as treeshold' % treeshold)

        self.Y_train_predicted_class = np.array(self.Y_train_predicted)
        indexes_1 = np.where(self.Y_train_predicted >= treeshold)[0]
        indexes_0 = np.where(self.Y_train_predicted < treeshold)[0]
        self.Y_train_predicted_class[indexes_1] = 1
        self.Y_train_predicted_class[indexes_0] = 0

        self.Y_test_predicted_class = np.array(self.Y_test_predicted)
        indexes_1 = np.where(self.Y_test_predicted >= treeshold)[0]
        indexes_0 = np.where(self.Y_test_predicted < treeshold)[0]
        self.Y_test_predicted_class[indexes_1] = 1
        self.Y_test_predicted_class[indexes_0] = 0

        # Compute Precision, Recall and F1 score
        precision_train = 0
        for i, j in enumerate(self.Y_train_predicted_class):
            if self.Y_train[i] == j:
                precision_train += 1
        precision_train = 100*precision_train/len(self.Y_train_predicted_class)

        precision_test = 0
        for i, j in enumerate(self.Y_test_predicted_class):
            if self.Y_test[i] == j:
                precision_test += 1
        precision_test = 100*precision_test/len(self.Y_test)

        recal_spamm_train = 0
        for i, j in enumerate(self.Y_train_predicted_class):
            if self.Y_train[i] == j and j == 1:
                recal_spamm_train += 1
        recal_spamm_train = 100*recal_spamm_train/len([i for i in self.Y_train if i == 1])

        recal_msg_train = 0
        for i, j in enumerate(self.Y_train_predicted_class):
            if self.Y_train[i] == j and j == 0:
                recal_msg_train += 1
        recal_msg_train = 100*recal_msg_train/len([i for i in self.Y_train if i == 0])

        recal_spamm_test = 0
        for i, j in enumerate(self.Y_test_predicted_class):
            if self.Y_test[i] == j and j == 1:
                recal_spamm_test += 1
        recal_spamm_test = 100*recal_spamm_test/len([i for i in self.Y_test if i == 1])

        recal_msg_test = 0
        for i, j in enumerate(self.Y_test_predicted_class):
            if self.Y_test[i] == j and j == 0:
                recal_msg_test += 1
        recal_msg_test = 100*recal_msg_test/len([i for i in self.Y_test if i == 0])

        # Show our result
        if verbose:
            print('MSE train = '+str(round(np.mean(self.Y_test - self.Y_test_predicted), 4)))
            print('MSE test = '+str(round(np.mean(self.Y_train - self.Y_train_predicted), 4)))
        print('Precision on train set is %.1f %%' % precision_train)
        print('Precision on test set is %.1f %%' % precision_test)
        print('Spamm Recall on train set is %.1f %%' % recal_spamm_train)
        print('Spamm Recall on test set is %.1f %%' % recal_spamm_test)
        if verbose:
            print('Message Recall on train set is %.1f %%' % recal_msg_train)
            print('Message Recall on test set is %.1f %%' % recal_msg_test)
        print('F1 score on train set is %.1f %%' % (2*precision_train*recal_spamm_train/(recal_spamm_train+precision_train)))
        print('F1 score on test set is %.1f %%' %  (2*precision_test*recal_spamm_test/(recal_spamm_test+precision_test)))
        print(' ')
