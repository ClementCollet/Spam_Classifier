import pandas as pd

import model
import function

##############  Parameter  ##############
# Set as true to see all info such as dictionnary example, stratification detail Roc Curb, Youden J stats curb, result with a non-optimal treeshold
VERBOSE = True
##############    Script    ##############
# Read raw data
raw_data = pd.read_csv('data.csv', sep=',', dtype={'v1':str, 'v2':str}, encoding='latin-1')

# Clean message: getting rid of stop word and special caractere, split sentence in a list of word
clean_data = function.cleanning_data(raw_data)

if VERBOSE:
    function.describe_dico(clean_data, column='clean_msg')

# Compute features from our data
clean_data = function.select_features(clean_data)

# Get our folds with split train/test and keeping same spam/normal message ratio
stratified_folds = function.get_stratified_folds(clean_data, stratify_on='label', nb_fold=4, verbose=VERBOSE)

fold = 1
for train, test in stratified_folds:
    print('Fold n°%d' % fold)
    print(' ')
    fold += 1

    # Add to our data the number of occurence of commun spam word
    train, test = function.add_feature_occurence_spam_word(train, test)

    # Selectionning our model's input and output
    x_train, y_train = train[['occurence_spam_word', 'occurence_money', 'nb_digits', 'nb_special', 'nb_upper_letter']].values, train['label'].values
    x_test, y_test = test[['occurence_spam_word', 'occurence_money', 'nb_digits', 'nb_special', 'nb_upper_letter']].values, test['label'].values

    # Initialize and fitting our model
    classifier = model.linear_regression_model()
    classifier.fit(x_train, y_train)

    y_train_predicted = classifier.predict(x_train)
    y_test_predicted = classifier.predict(x_test)

    if VERBOSE:
        classifier.show_roc(y_test, y_test_predicted, y_train, y_train_predicted)

    # Choosing our threeshold thanks to Youden J stats
    optimum_treeshold, treeshold_FPR_0_001 = classifier.compute_optimal_treeshold(y_test, y_test_predicted, y_train, y_train_predicted, verbose=VERBOSE)
    print('Optimal treeshold is %.2f.'%optimum_treeshold)
    print('Treeshold for False Positive Rate of 0.1%% is %.2f.'%optimum_treeshold)
    print(' ')

    # Getting our final result
    print('On TRAIN with Optimal treeshold: ')
    classifier.evaluate_result(y_train, y_train_predicted, treeshold=optimum_treeshold, verbose=VERBOSE)

    print('On TEST with Optimal treeshold: ')
    classifier.evaluate_result(y_test, y_test_predicted,treeshold=optimum_treeshold, verbose=VERBOSE)

    if VERBOSE:
    # if we want to see result with treeshold = 0.5
        print('On TRAIN with Treeshold for False Positive Rate of 0.1%%: ')
        classifier.evaluate_result(y_train, y_train_predicted, treeshold=treeshold_FPR_0_001, verbose=VERBOSE)

        print('On TEST with Treeshold for False Positive Rate of 0.1%%: ')
        classifier.evaluate_result(y_test, y_test_predicted,treeshold=treeshold_FPR_0_001, verbose=VERBOSE)
