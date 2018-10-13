import numpy as np
import pandas as pd

import model
import function

##############  Parameter  ##############
# Set as true to see all info such as dictionnary example, stratification detail Roc Curb, Youden J stats curve, result with an other treshold
VERBOSE = False
NB_FOLD = 10
##############    Script    ##############
# Read raw data
raw_data = pd.read_csv('data.csv', sep=',', dtype={'v1':str, 'v2':str}, encoding='latin-1')

# Clean message: getting rid of stop word and special caractere, split sentence in a list of word
clean_data = function.cleanning_data(raw_data)

if VERBOSE:
    function.study_selected_features(clean_data)

# Compute features from our data
clean_data = function.select_features(clean_data)

# Get our folds with split train/test and keeping same spam/normal message ratio
stratified_folds = function.get_stratified_folds(clean_data, stratify_on='label', nb_fold=NB_FOLD, verbose=VERBOSE)

result = []
fold = 0
for train, test in stratified_folds:
    print('Fold n°%d' % fold)
    print(' ')
    fold += 1

    # Add to our data the number of occurence of commun spam word
    train, test = function.add_feature_occurence_spam_word(train, test)

    # Selectionning our model's input and output
    x_train, y_train = train[['occurence_spam_word', 'nb_digits', 'nb_uppercase_letter']].values, train['label'].values
    x_test, y_test = test[['occurence_spam_word', 'nb_digits', 'nb_uppercase_letter']].values, test['label'].values

    # Initialize and fitting our model
    classifier = model.linear_regression_model()
    classifier.fit(x_train, y_train)

    y_train_predicted = classifier.predict(x_train)
    y_test_predicted = classifier.predict(x_test)

    if VERBOSE:
        classifier.show_roc(y_test, y_test_predicted, y_train, y_train_predicted)

    # Choosing our threeshold
    optimal_treshold, treshold_FPR_0_001 = classifier.compute_tresholds(y_test, y_test_predicted, y_train, y_train_predicted, verbose=VERBOSE)
    print('Optimal treshold is %.2f.'% optimal_treshold)
    print('Treeshold for False Positive Rate of 0.1%% is %.2f.'% treshold_FPR_0_001)
    print(' ')

    # Getting our final result
    print('On TRAIN with Optimal treshold: ')
    classifier.evaluate_result(y_train, y_train_predicted, treshold=optimal_treshold, verbose=VERBOSE)

    print('On TEST with Optimal treshold: ')
    result.append(classifier.evaluate_result(y_test, y_test_predicted, treshold=optimal_treshold, verbose=VERBOSE))

    if VERBOSE:
    # if we want to see result with treshold = treshold_FPR_0_001
        print('On TRAIN with Treeshold for False Positive Rate of 0.1%%: ')
        classifier.evaluate_result(y_train, y_train_predicted, treshold=treshold_FPR_0_001, verbose=VERBOSE)

        print('On TEST with Treeshold for False Positive Rate of 0.1%%: ')
        classifier.evaluate_result(y_test, y_test_predicted, treshold=treshold_FPR_0_001, verbose=VERBOSE)

result = np.asarray(result)
result_name = ['Precision', 'Recal Spam', 'Recal message', 'F1 score Spam']
for i in range(4):
    mean = np.mean(result, axis=0)
    print(result_name[i]+' => %.2f' % mean[i])
