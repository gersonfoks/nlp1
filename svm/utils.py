import json
from collections import defaultdict, Counter

import numpy as np
from scipy.sparse import csc_matrix
from tqdm import tqdm

from scipy.sparse import hstack, vstack


def load_reviews():
    with open("reviews.json", mode="r", encoding="utf-8") as f:
        reviews = json.load(f)
    return reviews


def calculate_results(predictions, targets):
    '''
        prediction: predictions of a model
        targets: the ground truth
        returns an array containing a 0 if the prediction is incorrect and 1 otherwise
    '''
    return np.array(predictions) == np.array(targets)


def calculate_accuracy(predictions, targets):
    '''
        Calculates the accuracy of the predictions
    '''
    return np.sum(calculate_results(predictions, targets)) / len(targets)


def create_fold_indexes(reviews, n_folds):
    '''
    Creates folds using round-robin splitting
    '''
    indexes = []

    for (index, review) in enumerate(reviews):
        i = review['cv'] % n_folds
        indexes.append(i)
    return indexes


def create_voc_of_features(X):
    '''
        Creates a vocabulary of the features that we encounter in the features
    :param X:
    :return: a vocabulary of the features
    '''
    voc = list(set([feature for features in X for feature in features]))

    return voc


def create_index_mapping(voc):
    '''
    Creates an index representation of the vocabulary
    First item will have index 1, second 2 etc.. Will reserve index 0 for unseen features.
    :param voc: The vocabulary to create index representation
    :return: a defaultdict containing a mapping from word to index
    '''
    one_hot_mapping = defaultdict(float)
    for i in tqdm(range(1, len(voc) + 1)):
        word = voc[i - 1]
        one_hot_mapping[word] = i
    return one_hot_mapping


def prepare_dataset(reviews, feature_function):
    '''
    Prepares the dataset, in which each review get prepared using the feature_function
    :param reviews:
    :param feature_map:
    :return:
    '''
    targets = []
    X = []

    ### Loop over the reviews, get the target, and make 1 sentence of the content.
    for review in reviews:
        targets.append(int(review['sentiment'] == 'POS'))

        whole_review = feature_function(review)
        X.append(whole_review)
    return X, targets


def review_to_sentence(review):
    '''
        Creates a sentence (list) of the content of a review
        content: [(word: string, POS: string)]
        returns: [word: string]
    '''
    whole_review = []
    for sentence in review['content']:
        whole_review += [word.lower() for word in np.array(sentence)[:, 0]]

    return whole_review

def review_to_words_pos(review):
    '''
        Creates a sentence (list) of the content of a review
        content: [(word: string, POS: string)]
        returns: [word: string]
    '''
    result = []
    for sentence in review['content']:
        result += [(word.lower(), pos) for word, pos in np.array(sentence)]

    return result

def review_to_words_pos_filtered(review):
    '''
        Creates a sentence (list) of the content of a review
        content: [(word: string, POS: string)]
        returns: [word: string]
    '''
    to_keep =  [
        'NN',
        'VB',
        'JJ',
        'RB',
    ]
    result = []
    for sentence in review['content']:
        result += [(word.lower(), pos) for word, pos in np.array(sentence) if pos in to_keep]

    return result


def features_to_indexes(features, index_mapping):
    '''
    Maps the features to their index
    :param features:
    :param index_mapping:
    :return:
    '''
    return [index_mapping[feature] for feature in features]


def counter_to_features(counter, n_features):
    '''
    Creates a vector containing the count of each index
    (Remember that each index represents a feature)
    :param counter: a Counter containing indexes and there counts
    :param n_features: the number of features that are used.
    :return: A numpy array containing the count of each feature in the corresponing index.
    '''
    result = np.zeros(n_features)
    for key, val in counter.items():
        result[key] = val
    return result


def create_feature_matrix(X, index_mapping):
    '''
    Create a sparse representation of the features using the given index map
    :param X:
    :param index_mapping:
    :return:
    '''
    result = []
    n_features = len(index_mapping) + 1
    for i in tqdm(range(len(X))):
        result.append(
            counter_to_features(
                Counter(features_to_indexes(X[i], index_mapping)),
                n_features
            )
        )
    return result


def train_on_fold_svm(model, x_folds, y_folds, held_out_index):
    '''
        Train the given model on the folds. Use held_out_index to check what the held out fold is.
    '''
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(x_folds)):
        if i != held_out_index:
            train_x += x_folds[i]
            train_y += y_folds[i]
        else:
            test_x = x_folds[i]
            test_y += y_folds[i]

    train_x = csc_matrix(train_x)
    test_x = csc_matrix(test_x)

    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    results = calculate_results(predictions, test_y)
    accuracy = calculate_accuracy(predictions, test_y)
    return accuracy, predictions, results


def train_on_folds_svm(create_model, X, Y, fold_indexes):
    '''
        Trains a model on the given folds.
        Create_model: a function that returns a model that we want to train
        x_folds: list of folds with the input features
        y_folds: list of folds with the target class.
    '''

    ### Create the actual folds.
    x_folds = defaultdict(list)
    y_folds = defaultdict(list)
    for index, x,y  in zip(fold_indexes, X, Y):
        x_folds[index] += [x]
        y_folds[index] += [y]
    print(len(x_folds))
    models = [create_model() for i in range(len(x_folds))]

    accuracies = []
    predictions = []
    results = []

    ### Loop over the folds.
    for i in tqdm(range(len(x_folds))):

        model = models[i]
        accuracy, prediction, result = train_on_fold_svm(model, x_folds, y_folds, i)
        accuracies.append(accuracy)
        predictions.append(predictions)
        results.append(result)
    result_dict = {
        "models": models,
        'accuracies': accuracies,
        'predictions': predictions,
        'results': results
    }
    return result_dict


def compare_models(result_dicts_model_1, result_dict_model_2):
    pass