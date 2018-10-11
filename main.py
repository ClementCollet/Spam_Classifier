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
    classifier = model.ML_model(x_train, y_train, x_test, y_test)
    classifier.fit_linear_regression()

    if VERBOSE:
        classifier.show_roc()

    # Choosing our threeshold thanks to Youden J stats
    optimum_treeshold = classifier.compute_Youden_J_stats(verbose=VERBOSE)
    print('Optimal treeshold is %.2f.'%optimum_treeshold)
    print(' ')

    # Getting our final result
    classifier.evaluate_result(treeshold=optimum_treeshold, verbose=VERBOSE)

    if VERBOSE:
    # if we want to see result with treeshold = 0.5
        classifier.evaluate_result(treeshold=0.5, verbose=VERBOSE)
