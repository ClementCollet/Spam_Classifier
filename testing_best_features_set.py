import numpy as np
import pandas as pd

import model
import function

##############  Parameter  ##############
# Set as true to see all info such as dictionnary example, stratification detail Roc Curb, Youden J stats curve, result with an other treshold
VERBOSE = False
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
stratified_folds = function.get_stratified_folds(clean_data, stratify_on='label', nb_fold=4, verbose=VERBOSE)

result = []
select_features_to_test = [['occurence_spam_word', 'nb_digits', 'nb_uppercase_letter'],
                            ['occurence_spam_word', 'nb_digits', 'nb_uppercase_letter','len_msg'],
                            ['occurence_spam_word', 'nb_digits', 'nb_special', 'nb_uppercase_letter','len_msg']]

for set_features in select_features_to_test:
    fold = 1
    result_set_features = []
    for train, test in stratified_folds:
        print('Fold n°%d' % fold)
        print(' ')
        fold += 1

        # Add to our data the number of occurence of commun spam word
        train, test = function.add_feature_occurence_spam_word(train, test)

        # Selectionning our model's input and output
        x_train, y_train = train[set_features].values, train['label'].values
        x_test, y_test = test[set_features].values, test['label'].values

        # Initialize and fitting our model
        classifier = model.linear_regression_model()
        classifier.fit(x_train, y_train)

        y_train_predicted = classifier.predict(x_train)
        y_test_predicted = classifier.predict(x_test)

        optimal_treshold, optimal_FPR_0_001 = classifier.compute_tresholds(y_test, y_test_predicted, y_train, y_train_predicted, verbose=VERBOSE)

        print('On TEST with Optimal treshold: ')
        result_set_features.append(classifier.evaluate_result(y_test, y_test_predicted,treshold=optimal_treshold, verbose=VERBOSE))

    result.append(result_set_features)

result = np.asarray(result)

result_name = ['Precision', 'Recal Spam', 'Recal message', 'F1 score']
for i in range(3):
    print('Testing model with '+str(select_features_to_test[i]))
    print('Result on test is :')
    mean = np.mean(result[i], axis=0)
    for j in range(4):
        print(result_name[j]+' => %.3f ' % mean[j])
