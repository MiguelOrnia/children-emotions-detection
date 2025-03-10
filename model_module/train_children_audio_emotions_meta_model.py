"""
Starting from the classifiers obtained in the Python file train_children_audio_emotions_model.py, we
obtain meta models using the ensemble stacking technique.
"""

import os
from multiprocessing import cpu_count
import numpy as np
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score, auc
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from processing_audio_data_module.extracting_audio_features.audio_features_extractor import get_audio_features
from util.helper import get_path, get_logger

default_classifier_rfe = dict()
default_classifier_rfe['lr'] = LogisticRegression(solver='liblinear')

default_parameters = dict()
default_parameters['lr'] = {'penalty': ['l2', 'l1'], 'tol': [1e-2, 1e-3, 1e-4, 1e-5], 'solver': ['liblinear'],
                            'max_iter': [100, 50, 200]}

num_folds = 10
val_size = 0.2
grade = 2.5

# For logging output
console_log = False
log = get_logger("train_model", console_log, __file__, "training_results", "logs")


def children_audio_emotions_meta_classifier(number_of_emotions, dataset_name=None, multiple_dataset=False):
    if dataset_name:
        log.info("Classifying " + dataset_name + " audio files by " + str(number_of_emotions) + " emotions")
    else:
        log.info("Classifying both corpus audio files by " + str(number_of_emotions) + " emotions")

    classifier = default_classifier_rfe['lr']
    params = default_parameters['lr']

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
    print(len(first_fold_train_data))

    # Uncomment only when need two dataset with different proportions
    data2 = first_fold_train_data
    target2 = first_fold_train_target
    print(data2)
    print(target2)

    # Uncomment only when need two dataset with different proportions
    data = data + data2
    target = target + target2
    print(data)
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

    # Load pre-trained models
    with open(os.path.join(get_path("", __file__),
                           "training_results/models/psvm/model_psvm_mesd_3.pkl"), 'rb') as file:
        model1 = joblib.load(file)

    with open(os.path.join(get_path("", __file__),
                           "training_results/models/knn/model_knn_mesd_3.pkl"), 'rb') as file:
        model2 = joblib.load(file)

    with open(os.path.join(get_path("", __file__),
                           "training_results/models/nn/model_nn_mesd_3.pkl"), 'rb') as file:
        model4 = joblib.load(file)

    # Define base models
    base_models = [
        ('model1', model1),
        ('model2', model2),
        ('model4', model4),
    ]

    if classifier is None:
        x2 = x
        features_names = np.array(features)
        feature_idx = range(len(features_names))
    else:
        # N_jobs for cross validation
        # 10 is for the numbers of folds in CV
        n_jobs = min(10, cpu_count())
        # Get best variables using RFE
        selector = RFECV(estimator=classifier, step=1, n_jobs=n_jobs, verbose=0, cv=10, min_features_to_select=10)
        selector = selector.fit(x, y)

        feature_idx = selector.get_support(True)
        features_names = np.array(features)[feature_idx]
        x2 = x[:, feature_idx]

    log.info(("Selected features [" + str(len(features_names)) + "] (meta_model): ")
             + ', '.join((str(r) for r in features_names)))

    cv = StratifiedKFold(n_splits=num_folds)

    # Create and train the stacking classifier
    stacking_clf = StackingClassifier(estimators=base_models, cv=cv)

    stacking_clf.fit(x2, y)

    x2_val = x_val[:, feature_idx]
    prediction_labels = stacking_clf.predict(x2_val)
    accuracy = accuracy_score(y_val, prediction_labels)
    f1 = f1_score(y_val, prediction_labels, average='weighted')

    # Labels metrics
    log.info("Accuracy (meta_model): " + str(accuracy))
    log.info("F1 Score (meta_model): " + str(f1))

    log.info('Confusion Matrix: (meta_model): \n' + str(confusion_matrix(y_val, prediction_labels)))
    log.info('Classification report: (meta_model): \n' + str(classification_report(y_val, prediction_labels)))

    # Save model_module into directory
    if dataset_name:
        output_name = 'model_metamodel_' + dataset_name + '_' + str(number_of_emotions)
    else:
        output_name = 'model_metamodel_both_' + str(number_of_emotions)
    out_path = os.path.join(get_path("", __file__), 'training_results', 'models', 'metamodel', output_name + '.pkl')
    # Save model_module
    joblib.dump(stacking_clf, out_path)

    if number_of_emotions == 2:
        probabilities = stacking_clf.predict_proba(x2_val)
        # Compute ROC curve and area the curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, probabilities[:, 1])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        log.info('AUC: ' + str(roc_auc))

        # Save values for roc cures
        np.savetxt(
            os.path.join(get_path("", __file__), 'training_results', 'models', 'metamodel', output_name + '_fpr.txt'),
            false_positive_rate)
        np.savetxt(
            os.path.join(get_path("", __file__), 'training_results', 'models', 'metamodel', output_name + '_tpr.txt'),
            true_positive_rate)
    else:
        probabilities = stacking_clf.predict_proba(x2_val)
        # Compute ROC curve and area the curve
        roc_auc = roc_auc_score(y_val, probabilities, multi_class='ovr')
        log.info('AUC: ' + str(roc_auc))
