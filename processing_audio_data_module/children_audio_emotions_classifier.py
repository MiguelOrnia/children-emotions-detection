""" Sort by Neutral, Positive and Negative using IESC-Child corpus """

import json
from datetime import datetime
import os
import sys
from multiprocessing import cpu_count
import logging as log
import numpy as np
from sklearn import svm
import joblib
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from processing_audio_data_module.extracting_audio_features.audios_processor import get_corpus_data
from util.os_helper import get_path

console_log = False

default_classifier_rfe = dict()
default_classifier_rfe['svc'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['nn'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['dt'] = None

default_classifiers = dict()
default_classifiers['svc'] = svm.SVC()
default_classifiers['nn'] = MLPClassifier()
default_classifiers['dt'] = DecisionTreeClassifier()

default_parameters = dict()
default_parameters['svc'] = {'C': [1], 'kernel': ['rbf'],
                             'gamma': ['scale'], 'tol': [1e-2], 'probability': [True], 'cache_size': [1024 * 4]}
default_parameters['nn'] = {'hidden_layer_sizes': [20, (20, 20)],
                            'activation': ['identity', 'relu', 'tanh', 'relu'], 'solver': ['adam', 'sgd', 'lbfgs'],
                            'alpha': [1, 0.1, 0.01, 0.001], 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                            'max_iter': [3000, 4000, 10000]}
default_parameters['dt'] = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                            'max_depth': [None, 5, 10, 15], 'max_features': ['sqrt'],
                            'class_weight': [None, 'balanced']}

num_folds = 6
val_size = 0.2
grade = 2.5


# Check logger output
if console_log:
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
else:
    # Logger to file
    sufix = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = 'train_model_' + sufix
    log.basicConfig(filename=os.path.join(get_path("", __file__), 'training', 'logs', log_name + '.log'), filemode='w',
                    level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


def children_audio_emotions_classifier(model_type, number_of_emotions, corpus_name):
    log.info("Sorting audio files by " + str(number_of_emotions) + " emotions")
    classifier = default_classifier_rfe[model_type]
    clf = default_classifiers[model_type]
    params = default_parameters[model_type]

    corpus = get_corpus_data(number_of_emotions, corpus_name)
    data = corpus['data']
    target = corpus['target']
    features = corpus['features']

    # Create the scaler object between 0 and 1
    scaler = MinMaxScaler()

    # Fit and transform the data
    data_scaled = scaler.fit_transform(data)
    log.info("Normalizing dataset using MinMaxScaler")

    # Split train and validation dataset
    x, x_val, y, y_val = train_test_split(data_scaled, target, test_size=val_size, random_state=round(grade))
    log.info("Split train and validation dataset (80% - 20%)")

    if classifier is None:
        x2 = x
        features_names = np.array(features)
        feature_idx = range(len(features_names))
    else:
        # N_jobs for cross validation
        # 6 is for the numbers of folds in CV
        n_jobs = min(6, cpu_count())
        # Get best variables using RFE
        selector = RFECV(estimator=classifier, step=1, n_jobs=n_jobs, verbose=0, cv=6, min_features_to_select=10)
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

    # Save model into directory
    output_name = 'model_' + model_type + '_' + corpus_name + '_' + str(number_of_emotions)
    out_path = os.path.join(get_path("", __file__), 'training', 'models', model_type, output_name + '.pkl')
    # Save model
    joblib.dump(best_model, out_path)

    if number_of_emotions == 2:
        probabilities = best_model.predict_proba(x2_val)
        # Compute ROC curve and area the curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, probabilities[:, 1])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        log.info('AUC: ' + str(roc_auc))

        # Save values for roc cures
        np.savetxt(os.path.join(get_path("", __file__), 'training', 'models', model_type, output_name + '_fpr.txt'),
                   false_positive_rate)
        np.savetxt(os.path.join(get_path("", __file__), 'training', 'models', model_type, output_name + '_tpr.txt'),
                   true_positive_rate)

    """
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_scaled, y, train_size=0.80, test_size=0.20
                                                                        , random_state=101)

    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(x_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1).fit(x_train, y_train)

    poly_pred = poly.predict(x_test)
    rbf_pred = rbf.predict(x_test)

    poly_accuracy = accuracy_score(y_test, poly_pred)
    poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
    print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
    """
