import re
import collections
import numpy as np
import pandas as pd
from string import digits

from nltk.corpus import stopwords

# Creating a new table with organized and cleanned data
def cleanning_data(raw_table):
    clean_data = pd.DataFrame()

    # Getting our label in integer to be model output
    label = []
    for i in raw_table['v1']:
        if i == 'spam':
            label.append(1)
        elif i == 'ham':
            label.append(0)
    clean_data['label'] = label

    # We keep raw data for some features
    clean_data['raw_msg'] = raw_table['v2']

    # We clean message by geting rid of special letters & stop word and putting all letter in lower case
    content_clean = []
    for i in raw_table['v2']:
        content_clean.append(clean_sentence(i, only_word=True, stop_word=True))
    clean_data['clean_msg'] = content_clean

    return clean_data

# Get rid of special letters & stop word, put in lower case
def clean_sentence(sentence, only_word=True, stop_word=True):
    if stop_word:
        stop = set(stopwords.words('english'))

    # We put space around special symbol because we will want the text split to isolize word
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " \( ", sentence)
    sentence = re.sub(r"\)", " \) ", sentence)
    sentence = re.sub(r"\?", " \? ", sentence)
    sentence = re.sub(r"\'s", " \'s", sentence)
    sentence = re.sub(r"\'ve", " \'ve", sentence)
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " \'re", sentence)
    sentence = re.sub(r"\'ll", " \'ll", sentence)
    sentence = re.sub(r"  ", " ", sentence)

    output = []
    # For each word
    for word in sentence.split():
        if only_word:
            # Taking out special symbol and put in lower case
            word = re.sub(r"[^A-Za-z]", "", word)
            word = word.lower()

        if word != '':
            if stop_word:
                # Taking out stop word like 'a', 'to', 'the' ...
                if word not in stop:
                    output.append(word.strip())
            else:
                output.append(word.strip())

    return output

# Shuffle rows in a panda Dataframe and return it
def shuffle_table(table):
    temp = np.arange(len(table))
    np.random.shuffle(temp)

    return table.loc[temp]

# Get our folds with a stratification on a table column, here do it on label (spamm or not)
def get_stratified_folds(table, stratify_on="label", nb_fold=4, verbose=False):
    shuffled_data = shuffle_table(table)

    values_to_stratif = np.unique(shuffled_data[stratify_on].values)
    shuffled_data_classA = shuffled_data[shuffled_data[stratify_on] == values_to_stratif[0]]
    shuffled_data_classB = shuffled_data[shuffled_data[stratify_on] == values_to_stratif[1]]

    nb_rows_per_stratif_classA = int(len(shuffled_data[shuffled_data[stratify_on] == values_to_stratif[0]])/nb_fold)
    nb_rows_per_stratif_classB = int(len(shuffled_data[shuffled_data[stratify_on] == values_to_stratif[1]])/nb_fold)

    shuffled_data_classA = shuffled_data[shuffled_data[stratify_on] == values_to_stratif[0]]
    shuffled_data_classB = shuffled_data[shuffled_data[stratify_on] == values_to_stratif[1]]

    output = []
    for i in range(nb_fold):
        train_A = shuffled_data_classA[nb_rows_per_stratif_classA*i:nb_rows_per_stratif_classA*(i+1)]
        train_B = shuffled_data_classB[nb_rows_per_stratif_classB*i:nb_rows_per_stratif_classB*(i+1)]
        train = pd.concat([train_A, train_B]).reset_index(drop=True)
        test_A_left = shuffled_data_classA[:nb_rows_per_stratif_classA*i]
        test_A_right = shuffled_data_classA[nb_rows_per_stratif_classA*(i+1):]
        test_B_left = shuffled_data_classB[:nb_rows_per_stratif_classB*i]
        test_B_right = shuffled_data_classB[nb_rows_per_stratif_classB*(i+1):]
        test = pd.concat([test_A_left, test_A_right, test_B_left, test_B_right]).reset_index(drop=True)
        output.append([shuffle_table(train), shuffle_table(test)])

    if verbose:
        # Detail fold repartition
        print('There is '+str(len(shuffled_data))+' mails, repartition on label normal message/spam is '+str([len(shuffled_data_classA), len(shuffled_data_classB)]))
        print('Each of '+str(nb_fold)+' fold will have a [train,test] size of '+str([len(train), len(test)])+', with repartition on label ham/spam = '+str([[nb_rows_per_stratif_classA, nb_rows_per_stratif_classB],[nb_rows_per_stratif_classA*(nb_fold-1), (nb_fold-1)*nb_rows_per_stratif_classB]]))


    return output

# Show most used word in both sub dataset
def study_selected_features(table):
    special = ['!', ';', '?', '<', '>', '#', '$', '€', '£']
    for indx, row in table.iterrows():
        table.at[indx, 'nb_digits'] = len([i for i in row['raw_msg'] if i in digits])
        table.at[indx, 'nb_special'] = len([i for i in row['raw_msg'] if i in special])
        table.at[indx, 'nb_uppercase_letter'] = len([i for i in row['raw_msg'] if i.isupper()])
        table.at[indx, 'len_msg'] = len(row['raw_msg'])

    temp_spam = table[table.label == 1]
    temp_normal_msg = table[table.label == 0]

    print('Concerning most used word in spam')
    spam_dictionary = temp_spam['clean_msg'].values
    spam_dictionary = [word for sentence in spam_dictionary for word in sentence]
    counter_spam = collections.Counter(spam_dictionary)
    print('There is '+str(len(spam_dictionary))+ ' words in '+str(len(temp_spam))+' spams, the dictionary containt '+str(len(list(counter_spam)))+' differents words.')
    print('Here is the top 20 of most commun word and the number of times there are used: ')
    print(counter_spam.most_common(20))
    print(' ')

    spam_dictionary = temp_normal_msg['clean_msg'].values
    spam_dictionary = [word for sentence in spam_dictionary for word in sentence]
    counter_normal_msg = collections.Counter(spam_dictionary)
    print('There is '+str(len(spam_dictionary))+' words in '+str(len(temp_normal_msg))+' normal message, the dictionary containt '+str(len(list(counter_normal_msg)))+' differents words.')
    print('Here is the top 20 of most commun word and the number of times there are used : ')
    print(counter_normal_msg.most_common(20))
    print(' ')

    print('Concerning number of digits used :')
    print('for spam : mean %.1f +/- %.1f ' % (np.mean(temp_spam.nb_digits.values), np.std(temp_spam.nb_digits.values)))
    print('for normal message : mean %.1f +/- %.1f ' % (np.mean(temp_normal_msg.nb_digits.values), np.std(temp_normal_msg.nb_digits.values)))
    print(' ')

    print('Concerning number of special used :')
    print('for spam : mean %.1f +/- %.1f ' % (np.mean(temp_spam.nb_special.values), np.std(temp_spam.nb_special.values)))
    print('for normal message : mean %.1f +/- %.1f ' % (np.mean(temp_normal_msg.nb_special.values), np.std(temp_normal_msg.nb_special.values)))
    print(' ')

    print('Concerning number uppercase used :')
    print('for spam : mean %.1f +/- %.1f ' % (np.mean(temp_spam.nb_uppercase_letter.values), np.std(temp_spam.nb_uppercase_letter.values)))
    print('for normal message : mean %.1f +/- %.1f ' % (np.mean(temp_normal_msg.nb_uppercase_letter.values), np.std(temp_normal_msg.nb_uppercase_letter.values)))
    print(' ')

    print('Concerning length of message :')
    print('for spam : mean %.1f +/- %.1f ' % (np.mean(temp_spam.len_msg.values), np.std(temp_spam.len_msg.values)))
    print('for normal message : mean %.1f +/- %.1f ' % (np.mean(temp_normal_msg.len_msg.values), np.std(temp_normal_msg.len_msg.values)))
    print(' ')

# Add features to our dataset
def select_features(table):
    special = ['!', ';', '?', '<', '>', '#', '$', '€', '£']
    for indx, row in table.iterrows():
        table.at[indx, 'nb_digits'] = len([i for i in row['raw_msg'] if i in digits])
        table.at[indx, 'nb_special'] = len([i for i in row['raw_msg'] if i in special])
        table.at[indx, 'nb_uppercase_letter'] = len([i for i in row['raw_msg'] if i.isupper()])
        table.at[indx, 'len_msg'] = len(row['raw_msg'])
    table['nb_digits'] = table['nb_digits'].values/np.max(table['nb_digits'].values)
    table['nb_special'] = table['nb_special'].values/np.max(table['nb_special'].values)
    table['nb_uppercase_letter'] = table['nb_uppercase_letter'].values/np.max(table['nb_uppercase_letter'].values)
    table['len_msg'] = table['len_msg'].values/np.max(table['len_msg'].values)

    return table

# Add feature on recuurent spam word use, using only trainning data
def add_feature_occurence_spam_word(train, test):
    temp_spam = train[train.label == 1]
    spam_dictionary = temp_spam['clean_msg'].values
    spam_dictionary = [word for sentence in spam_dictionary for word in sentence]
    counter_spam = collections.Counter(spam_dictionary)
    commun_spam_word = counter_spam.most_common(20)
    commun_spam_word = [i[0] for i in commun_spam_word]

    temp_normal_msg = train[train.label == 0]
    normal_msg_dictionary = temp_normal_msg['clean_msg'].values
    normal_msg_dictionary = [word for sentence in normal_msg_dictionary for word in sentence]
    counter_normal_msg = collections.Counter(normal_msg_dictionary)
    commun_normal_msg = counter_normal_msg.most_common(50)
    commun_normal_msg = [i[0] for i in commun_normal_msg]

    only_in_spam_word = []
    for i in commun_spam_word:
        if i not in commun_normal_msg:
            only_in_spam_word.append(i)

    for indx, row in train.iterrows():
        train.at[indx, 'occurence_spam_word'] = len([i for i in row['clean_msg'] if i in only_in_spam_word])
    train['occurence_spam_word'] = train['occurence_spam_word'].values/np.max(train['occurence_spam_word'].values)

    for indx, row in test.iterrows():
        test.at[indx, 'occurence_spam_word'] = len([i for i in row['clean_msg'] if i in only_in_spam_word])
    test['occurence_spam_word'] = test['occurence_spam_word'].values/np.max(test['occurence_spam_word'].values)

    return train, test
