"""
    Classifies children' audio files in different emotional expressions groups using MESD (100%), Draw&Talk (20%)
    and Combined (MESD 100% + Draw&Talk 20%) datasets.
    In addition, this script use three classifiers: Multilayer Perceptron (MLP), Support Vector Machine (SVM)
    with different kernels, K Nearest Neighbors (KNN) and Logistic Regression (LR).

    The selected classes are the following ones: Positive, Negative and Neutral emotional expressions.
"""

import json
import os
from multiprocessing import cpu_count
import numpy as np
# from imblearn.over_sampling import SMOTE
from sklearn import svm
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score, auc
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from processing_audio_data_module.extracting_audio_features.audio_features_extractor import get_audio_features
from util.helper import get_path, get_logger

default_classifier_rfe = dict()
default_classifier_rfe['svc'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['nn'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['knn'] = None
default_classifier_rfe['lr'] = LogisticRegression(solver='liblinear')

default_classifiers = dict()
default_classifiers['svc'] = svm.SVC()
default_classifiers['nn'] = MLPClassifier()
default_classifiers['knn'] = KNeighborsClassifier()
default_classifiers['lr'] = LogisticRegression()


default_parameters = dict()
default_parameters['svc'] = {'C': [0.5, 1], 'kernel': ['poly'],
                             'gamma': ['scale'], 'tol': [1e-2], 'probability': [True], 'cache_size': [1024 * 4]}
default_parameters['nn'] = {'hidden_layer_sizes': [20, (20, 20)],
                            'activation': ['identity', 'relu', 'tanh', 'relu'], 'solver': ['adam', 'sgd', 'lbfgs'],
                            'alpha': [1, 0.1, 0.01, 0.001], 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                            'max_iter': [3000, 4000, 5000, 10000, 1000000], 'n_iter_no_change': [10, 15, 20],
                            'early_stopping': [True]}
default_parameters['knn'] = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'weights': ['uniform', 'distance'],
                             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                             'leaf_size': [10, 20, 30, 40, 50], 'p': [1, 2],
                             'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}
default_parameters['lr'] = {'penalty': ['l2', 'l1'], 'tol': [1e-2, 1e-3, 1e-4, 1e-5], 'solver': ['liblinear'],
                            'max_iter': [100, 50, 200]}

num_folds = 10
val_size = 0.2
grade = 2.5

# For logging output
console_log = False
log = get_logger("train_model", console_log, __file__, "training_results", "logs")


def children_audio_emotions_classifier(model_type, number_of_emotions, dataset_name=None, multiple_dataset=False):
    if dataset_name:
        log.info("Classifying " + dataset_name + " audio files by " + str(number_of_emotions) + " emotions")
    else:
        log.info("Classifying both corpus audio files by " + str(number_of_emotions) + " emotions")
    classifier = default_classifier_rfe[model_type]
    clf = default_classifiers[model_type]
    params = default_parameters[model_type]

    # Uncomment only when need two dataset with different proportions
    dataset2 = {}

    if multiple_dataset:
        dataset = get_audio_features(number_of_emotions=number_of_emotions, multiple_dataset=multiple_dataset)
    else:
        dataset = get_audio_features(dataset_name, number_of_emotions)
        # Uncomment only when need two dataset with different proportions
        dataset2 = get_audio_features('draw_talk', number_of_emotions)

    data = dataset['data']
    target = dataset['target']
    features = dataset['features']

    # Uncomment only when need two dataset with different proportions
    skf = StratifiedKFold(n_splits=5)
    first_fold_train_indices, first_fold_test_indices = next(skf.split(dataset2['data'], dataset2['target']))
    first_fold_train_data = [dataset2['data'][i] for i in first_fold_test_indices]
    first_fold_train_target = [dataset2['target'][i] for i in first_fold_test_indices]

    # Uncomment only when need two dataset with different proportions
    data2 = first_fold_train_data
    target2 = first_fold_train_target

    # Uncomment only when need two dataset with different proportions
    data = np.concatenate((data, data2))
    target = np.concatenate((target, target2))
    print(target)

    # Create the scaler object between 0 and 1
    scaler = MinMaxScaler()

    # Fit and transform the data
    data_scaled = scaler.fit_transform(data)
    log.info("Normalizing dataset using MinMaxScaler")

    # Split train and validation dataset
    x, x_val, y, y_val = train_test_split(data_scaled, target, test_size=val_size, random_state=round(grade))
    log.info("Split train and validation dataset (" + str(int((1 - val_size) * 100)) + "% - " + str(
        int(val_size * 100)) + "%)")

    """
    # Oversampling. Applying SMOTE oversampling to training data
    log.info("Applying oversampling with SMOTE")
    sm = SMOTE(random_state=42)
    x, y = sm.fit_resample(x, y)
    """

    if classifier is None:
        x2 = x
        features_names = np.array(features)
        feature_idx = range(len(features_names))
    else:
        # N_jobs for cross validation
        # 10 is for the numbers of folds in CV
        n_jobs = min(10, cpu_count())
        # Get the best variables using RFE
        selector = RFECV(estimator=classifier, step=1, n_jobs=n_jobs, verbose=0, cv=10, min_features_to_select=10)
        selector = selector.fit(x, y)

        feature_idx = selector.get_support(True)
        features_names = np.array(features)[feature_idx]
        x2 = x[:, feature_idx]

    log.info(("Selected features [" + str(len(features_names)) + "] (" + model_type + "): ")
             + ', '.join((str(r) for r in features_names)))

    cv = StratifiedKFold(n_splits=num_folds)

    if model_type == 'nn':
        searcher = RandomizedSearchCV(estimator=clf, param_distributions=params, cv=cv, scoring='accuracy', n_jobs=1,
                                      random_state=round(grade), verbose=1)
    else:
        searcher = GridSearchCV(clf, params, scoring='accuracy', cv=cv, n_jobs=1, verbose=1)
    searcher.fit(x2, y)
    best_model = searcher.best_estimator_
    log.info("Best hyper parameters (" + model_type + "): " + json.dumps(searcher.best_params_))

    x2_val = x_val[:, feature_idx]
    prediction_labels = best_model.predict(x2_val)
    accuracy = accuracy_score(y_val, prediction_labels)
    f1 = f1_score(y_val, prediction_labels, average='weighted')

    # Labels metrics
    log.info("Accuracy (" + model_type + "): " + str(accuracy))
    log.info("F1 Score (" + model_type + "): " + str(f1))

    log.info('Confusion Matrix: (' + model_type + '): \n' + str(confusion_matrix(y_val, prediction_labels)))
    log.info('Classification report: (' + model_type + '): \n' + str(classification_report(y_val, prediction_labels)))

    # Save model_module into directory
    if dataset_name:
        output_name = 'model_' + model_type + '_' + dataset_name + '_' + str(number_of_emotions)
    else:
        output_name = 'model_' + model_type + '_both_' + str(number_of_emotions)
    out_path = os.path.join(get_path("", __file__), 'training_results', 'models', model_type, output_name + '.pkl')
    # Save model_module
    joblib.dump(best_model, out_path)

    if number_of_emotions == 2:
        probabilities = best_model.predict_proba(x2_val)
        # Compute ROC curve and area the curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, probabilities[:, 1])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        log.info('AUC: ' + str(roc_auc))

        # Save values for roc cures
        np.savetxt(
            os.path.join(get_path("", __file__), 'training_results', 'models', model_type, output_name + '_fpr.txt'),
            false_positive_rate)
        np.savetxt(
            os.path.join(get_path("", __file__), 'training_results', 'models', model_type, output_name + '_tpr.txt'),
            true_positive_rate)
    else:
        probabilities = best_model.predict_proba(x2_val)
        # Compute ROC curve and area the curve
        roc_auc = roc_auc_score(y_val, probabilities, multi_class='ovr')
        log.info('AUC: ' + str(roc_auc))


# Uncomment only when need two dataset with different proportions
def get_data2():
    dataset2 = get_audio_features('draw_talk', 3)

    skf = StratifiedKFold(n_splits=5)
    first_fold_train_indices, first_fold_test_indices = next(skf.split(dataset2['data'], dataset2['target']))
    first_fold_test_data = [dataset2['data'][i] for i in first_fold_train_indices]
    first_fold_test_target = [dataset2['target'][i] for i in first_fold_train_indices]
    first_fold_test_files = [dataset2['files'][i] for i in first_fold_train_indices]

    data = first_fold_test_data
    target = first_fold_test_target
    files = first_fold_test_files

    return data, target, files
