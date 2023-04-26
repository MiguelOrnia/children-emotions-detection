""" In this script we are able to get audio features from children' audio files """

import os
import opensmile
import pandas as pd
import csv
import json
import numpy as np
from util.os_helper import get_path


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


DATASET_MESD_PATH = "../../datasets/mesd_audio_dataset"
DATASET_MESD_RELATIVE_PATH = "datasets/mesd_audio_dataset/"
DATASET_IESC_CHILD_PATH = "../../datasets/iesc_child_audio_dataset"
DATASET_IESC_CHILD_RELATIVE_PATH = "datasets/iesc_child_audio_dataset/"
IESC_CHILD_LABELS_PATH = r"C:\Users\mor-n\Escritorio\Master\Segundo curso\Segundo semestre\TFM\Proyecto" \
                         r"\processing_audio_data_module\extracting_audio_features\Files_labels_iesc_child.xlsx"

emotions = []
emotions_labels = []
data = {}
files = []
features = []
results_functionals = []
results_low_level = []
results_low_level_using_mean = []

mesd_emotions_classification = {3: {0: 'Positive', 1: 'Negative', 2: 'Neutral'}, 5: {'Anger': 0, 'Disgust': 1,
                                                                                     'Fear': 2, 'Happiness': 3,
                                                                                     'Sadness': 4}}
mesd_negative_emotions = ['Anger', 'Disgust', 'Fear', 'Sadness']
mesd_postive_emotions = ['Happiness']

iesc_child_emotions_classification = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
iesc_child_positive_emotions = ['felicidad']
iesc_child_negative_emotions = ['desprecio', 'miedo', 'ninguno', 'sorpresa', 'tristeza']


def get_target(emotion, number_of_emotions, corpus):
    if corpus == 'mesd':
        if number_of_emotions == 5:
            return emotion
        else:
            if emotion in mesd_postive_emotions:
                return mesd_emotions_classification[3][0]
            elif emotion in mesd_negative_emotions:
                return mesd_emotions_classification[3][1]
            else:
                return mesd_emotions_classification[3][2]
    else:
        if number_of_emotions == 2:
            if emotion == 'neutral':
                return 'Neutral'
            else:
                return 'No Neutral'
        else:
            if emotion in iesc_child_positive_emotions:
                return iesc_child_emotions_classification[0]
            elif emotion in iesc_child_negative_emotions:
                return iesc_child_emotions_classification[1]
            else:
                return iesc_child_emotions_classification[2]


def get_target_value(emotion, number_of_emotions, corpus):
    if corpus == 'mesd':
        if number_of_emotions == 5:
            return mesd_emotions_classification[5][emotion]
        else:
            if emotion in mesd_postive_emotions:
                return 0
            elif emotion in mesd_negative_emotions:
                return 1
            else:
                return 2
    else:
        if number_of_emotions == 2:
            if emotion == 'neutral':
                return 0
            else:
                return 1
        else:
            if emotion in iesc_child_positive_emotions:
                return 0
            elif emotion in iesc_child_negative_emotions:
                return 1
            else:
                return 2


def __get_smile_conf(feature_set, feature_level):
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )
    return smile


def __get_results_from_other_configurations(result_low_level, features_low_level, file_number):
    iteration_file_number = 0
    initial_position = len(results_low_level[file_number])
    for i, row in result_low_level[features_low_level].iterrows():
        actual_position = initial_position
        if iteration_file_number == 0 and file_number == 0:
            global features
            features += row.keys().tolist()
        for value in row.values:
            if iteration_file_number == 0:
                results_low_level[file_number].append([])
            feature_position = actual_position
            results_low_level[file_number][feature_position].append(value)
            actual_position += 1
        iteration_file_number += 1


def __get_results_low_level(result_egemaps_low_level, file_number, result_emobase_low_level, emobase_low_level,
                            result_compare_low_level, compare_low_level):
    iteration_file_number = 0
    results_low_level.append([])
    for i, row in result_egemaps_low_level.iterrows():
        feature_number = 0
        if iteration_file_number == 0 and file_number == 0:
            global features
            features = row.keys().tolist()
        for value in row.values:
            if iteration_file_number == 0:
                results_low_level[file_number].append([])
            results_low_level[file_number][feature_number].append(value)
            feature_number += 1
        iteration_file_number += 1
    __get_results_from_other_configurations(result_emobase_low_level, emobase_low_level, file_number)
    __get_results_from_other_configurations(result_compare_low_level, compare_low_level, file_number)


def __get_selected_features_json(filename, conf):
    # Opening JSON file
    features_file = open(filename)

    # returns JSON object as a dictionary
    selected_features = json.load(features_file)

    # Closing file
    features_file.close()

    return selected_features[conf]['functionals'], selected_features[conf]['low-level']


def __export_data_to_json(number_of_emotions, corpus):
    with open("processing_audio_data_module/extracting_audio_features/data_" + str(number_of_emotions) + "_" + corpus
              + ".json", "w") as outfile:
        json.dump(data, outfile, cls=NpEncoder)


def __remove_outliers(arr):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return arr[(arr >= lower_bound) & (arr <= upper_bound)]


def __get_emotion(filename, corpus):
    if corpus == 'mesd':
        return str(filename).split("_")[0]
    else:
        labels = pd.read_excel(IESC_CHILD_LABELS_PATH)
        for i, row in labels.iterrows():
            if row['Filename'] == filename:
                return row['Emotion']


def __calculate_audio_features(number_of_emotions, corpus):
    file_number = 0
    if corpus == "mesd":
        path = DATASET_MESD_PATH
        relative_path = DATASET_MESD_RELATIVE_PATH
    else:
        path = DATASET_IESC_CHILD_PATH
        relative_path = DATASET_IESC_CHILD_RELATIVE_PATH
    for filename in os.listdir(get_path(path, __file__)):
        emotion = __get_emotion(filename, corpus)
        if number_of_emotions != 5 or emotion != 'Neutral':
            emotions.append(get_target_value(emotion, number_of_emotions, corpus))
            emotions_labels.append(get_target(emotion, number_of_emotions, corpus))
            files.append(filename)

            # OpenSmile configurations
            # eGeMAPS (functionals, low level)
            smile_egemaps_functionals = __get_smile_conf(opensmile.FeatureSet.eGeMAPSv02,
                                                         opensmile.FeatureLevel.Functionals)
            smile_egemaps_low_level = __get_smile_conf(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.
                                                       LowLevelDescriptors)
            # emobase (functionals, low level)
            smile_emobase_functionals = __get_smile_conf(opensmile.FeatureSet.emobase,
                                                         opensmile.FeatureLevel.Functionals)
            smile_emobase_low_level = __get_smile_conf(opensmile.FeatureSet.emobase, opensmile.FeatureLevel.
                                                       LowLevelDescriptors)
            # ComParE (functionals, low level)
            smile_compare_functionals = __get_smile_conf(opensmile.FeatureSet.ComParE_2016, opensmile.FeatureLevel.
                                                         Functionals)
            smile_compare_low_level = __get_smile_conf(opensmile.FeatureSet.ComParE_2016, opensmile.FeatureLevel.
                                                       LowLevelDescriptors)

            # Get OpenSmile configurations results for the current wav file
            # emobase
            result_emobase_low_level = smile_emobase_low_level.process_file(relative_path + str(filename))
            result_emobase_functionals = smile_emobase_functionals.process_file(relative_path + str(filename))
            # eGeMAPS
            result_egemaps_low_level = smile_egemaps_low_level.process_file(relative_path + str(filename))
            result_egemaps_functionals = smile_egemaps_functionals.process_file(relative_path + str(filename))
            # ComParE
            result_compare_low_level = smile_compare_low_level.process_file(relative_path + str(filename))
            result_compare_functionals = smile_compare_functionals.process_file(relative_path + str(filename))

            # Selected features from emobase and ComParE configurations
            # emobase
            emobase_functionals = __get_selected_features_json('processing_audio_data_module/extracting_audio_features'
                                                               '/emobase_features.json', 'emobase')[0]
            emobase_low_level = __get_selected_features_json('processing_audio_data_module/extracting_audio_features'
                                                             '/emobase_features.json', 'emobase')[1]
            # ComParE
            compare_functionals = __get_selected_features_json('processing_audio_data_module/extracting_audio_features/'
                                                               'compare_features.json', 'compare')[0]
            compare_low_level = __get_selected_features_json('processing_audio_data_module/extracting_audio_features/'
                                                             'compare_features.json', 'compare')[1]

            # Combining statistics results (functionals configurations)
            result_functional = result_egemaps_functionals.join(result_emobase_functionals[emobase_functionals],
                                                                how="outer").join(
                result_compare_functionals[compare_functionals], how="outer")
            results_functionals.append(result_functional)

            __get_results_low_level(result_egemaps_low_level, file_number, result_emobase_low_level, emobase_low_level,
                                    result_compare_low_level, compare_low_level)
            file_number += 1

    # Generating csv file with statistics (obtained with functionals configurations) of the selected features
    features_functionals = pd.concat(results_functionals)
    features_functionals.to_csv('features_functionals-' + str(number_of_emotions) + '-' + corpus + '.csv',
                                quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)

    # Calculating mean after removing outliers
    global results_low_level
    file_number = 0
    for file in results_low_level:
        results_low_level_using_mean.append([])
        for feature in file:
            results_low_level_without_outliers = __remove_outliers(np.array(feature))
            mean = np.mean(results_low_level_without_outliers)
            results_low_level_using_mean[file_number].append(mean)
        file_number += 1

    # Filling data dictionary
    data['data'] = results_low_level_using_mean
    data['target'] = emotions
    data['target_names'] = emotions_labels
    data['files'] = files
    data['features'] = features

    # Exporting data to a json file
    __export_data_to_json(number_of_emotions, corpus)


def get_corpus_data(number_of_emotions, corpus):
    try:
        filename = "processing_audio_data_module/extracting_audio_features/data_" + str(number_of_emotions) + "_" \
                   + corpus + ".json"
        data_file = open(filename)
        global data
        data = json.load(data_file)
        data_file.close()
    except FileNotFoundError:
        __calculate_audio_features(number_of_emotions, corpus)
    finally:
        return data
