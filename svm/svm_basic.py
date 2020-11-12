from svm.utils import *
from sklearn import svm


reviews = load_reviews()



X, Y = prepare_dataset(reviews, review_to_sentence)


voc = create_voc_of_features(X)

index_mapping = create_index_mapping(voc)

X = create_feature_matrix(X, index_mapping)


fold_indexes = create_fold_indexes(reviews, n_folds=10)


result_dict = train_on_folds_svm(svm.SVC, X,Y, fold_indexes)
print(result_dict['predictions'])
print(result_dict['accuracies'])








