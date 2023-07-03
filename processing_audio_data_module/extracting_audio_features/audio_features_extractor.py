""" In this script we are able to get audio features from children' audio files (wav) """

import os
import opensmile
import pandas as pd
import csv
import json
import numpy as np
from util.helper import get_path, remove_outliers, NpEncoder

MESD_DATASET_NAME = "mesd"
DATASET_MESD_PATH = "../../datasets/mesd_audio_dataset"
DATASET_MESD_RELATIVE_PATH = "datasets/mesd_audio_dataset/"

IESC_CHILD_DATASET_NAME = "iesc_child"
DATASET_IESC_CHILD_PATH = "../../datasets/iesc_child_audio_dataset"
DATASET_IESC_CHILD_RELATIVE_PATH = "datasets/iesc_child_audio_dataset/"
IESC_CHILD_LABELS_PATH = r"processing_audio_data_module/extracting_audio_features/labels_files/" \
                         r"files_labels_iesc_child.xlsx"

DRAW_TALK_DATASET_NAME = "draw_talk"
DATASET_DRAW_TALK_PATH = "../../datasets/draw_talk_full_dataset_cleaned"
DATASET_DRAW_TALK_RELATIVE_PATH = "datasets/draw_talk_full_dataset_cleaned/"

CONF_FEATURES_PATH = "processing_audio_data_module/extracting_audio_features/conf_features"
OUTPUT_FUNCTIONALS_PATH = "processing_audio_data_module/extracting_audio_features/outputs/functionals/"
OUTPUT_LOW_LEVEL_PATH = "processing_audio_data_module/extracting_audio_features/outputs/low_level/"

POSITIVE = 0
NEGATIVE = 1
NEUTRAL = 2

emotions = []
emotions_labels = []
audios_data = {}
files = []
features = []
results_functionals = []
results_low_level = []
results_low_level_using_mean = []
file_number = 0

emotions_classification = {POSITIVE: 'Positive', NEGATIVE: 'Negative', NEUTRAL: 'Neutral'}

mesd_negative_emotions = ['Anger', 'Disgust', 'Fear', 'Sadness']
mesd_positive_emotions = ['Happiness']

iesc_child_negative_emotions = ['desprecio', 'miedo', 'tristeza', 'enojo', 'sorpresa']
iesc_child_positive_emotions = ['felicidad']


def __set_paths(dataset):
    if dataset == MESD_DATASET_NAME:
        path = DATASET_MESD_PATH
        relative_path = DATASET_MESD_RELATIVE_PATH
    else:
        path = DATASET_IESC_CHILD_PATH
        relative_path = DATASET_IESC_CHILD_RELATIVE_PATH
    return path, relative_path


def __get_target(emotion):
    if emotion in iesc_child_positive_emotions or emotion in mesd_positive_emotions:
        return emotions_classification[POSITIVE]
    elif emotion in iesc_child_negative_emotions or emotion in mesd_negative_emotions:
        return emotions_classification[NEGATIVE]
    elif emotion == 'Neutral':
        return emotions_classification[NEUTRAL]
    else:
        raise ValueError("Inappropriate emotion for IESC-Child: Neutral")


def __get_target_value(emotion):
    if emotion in iesc_child_positive_emotions or emotion in mesd_positive_emotions:
        return POSITIVE
    elif emotion in iesc_child_negative_emotions or emotion in mesd_negative_emotions:
        return NEGATIVE
    elif emotion == 'Neutral':
        return NEUTRAL
    else:
        raise ValueError("Inappropriate emotion for IESC-Child: Neutral")


def get_label_by_value(emotion_value):
    return emotions_classification[emotion_value]


def __get_smile_conf(feature_set, feature_level):
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )
    return smile


def __get_results_from_other_configurations(result_low_level, features_low_level):
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


def __get_results_low_level(result_egemaps_low_level, result_emobase_low_level, emobase_low_level,
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
    __get_results_from_other_configurations(result_emobase_low_level, emobase_low_level)
    __get_results_from_other_configurations(result_compare_low_level, compare_low_level)


def __get_selected_features_from_json_file(features_filename, conf):
    # Opening JSON file
    features_file = open(features_filename)

    # returns JSON object as a dictionary
    selected_features = json.load(features_file)

    # Closing file
    features_file.close()

    return selected_features[conf]['functionals'], selected_features[conf]['low-level']


def __export_data_results_to_json(number_of_emotions, dataset):
    if dataset == DRAW_TALK_DATASET_NAME:
        filename = OUTPUT_LOW_LEVEL_PATH + "data_audios_" + dataset + ".json"
    else:
        filename = OUTPUT_LOW_LEVEL_PATH + "data_audios_" + str(number_of_emotions) + "_" + dataset + ".json"
    with open(filename, "w") as outfile:
        json.dump(audios_data, outfile, cls=NpEncoder)


def __calculate_mean_for_audio_features():
    global results_low_level, file_number
    file_number = 0
    for file in results_low_level:
        results_low_level_using_mean.append([])
        for feature in file:
            results_low_level_without_outliers = remove_outliers(np.array(feature))
            mean = np.mean(results_low_level_without_outliers)
            results_low_level_using_mean[file_number].append(mean)
        file_number += 1


def __get_emotion(filename, corpus):
    if corpus == MESD_DATASET_NAME:
        return str(filename).split("_")[0]
    else:
        labels = pd.read_excel(IESC_CHILD_LABELS_PATH)
        for i, row in labels.iterrows():
            if row['Filename'] == filename:
                return row['Emotion']


def __check_emotion(number_of_emotions, emotion):
    return number_of_emotions != 2 or emotion.lower() != 'neutral'


def __extract_audio_features(relative_path, filename):
    global file_number

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
    emobase_functionals = \
        __get_selected_features_from_json_file(CONF_FEATURES_PATH + '/emobase_features.json', 'emobase')[0]
    emobase_low_level = \
        __get_selected_features_from_json_file(CONF_FEATURES_PATH + '/emobase_features.json', 'emobase')[1]
    # ComParE
    compare_functionals = \
        __get_selected_features_from_json_file(CONF_FEATURES_PATH + '/compare_features.json', 'compare')[0]
    compare_low_level = \
        __get_selected_features_from_json_file(CONF_FEATURES_PATH + '/compare_features.json', 'compare')[1]

    # Combining statistics results (functionals configurations)
    result_functional = result_egemaps_functionals.join(result_emobase_functionals[emobase_functionals],
                                                        how="outer").join(
        result_compare_functionals[compare_functionals], how="outer")
    results_functionals.append(result_functional)

    __get_results_low_level(result_egemaps_low_level, result_emobase_low_level, emobase_low_level,
                            result_compare_low_level, compare_low_level)
    file_number += 1


def __export_functional_values(number_of_emotions, dataset):
    features_functionals = pd.concat(results_functionals)
    features_functionals.to_csv(
        OUTPUT_FUNCTIONALS_PATH + 'features_functionals-' + str(number_of_emotions) + '-' + dataset + '.csv',
        quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)


def __calculate_audio_features_from_labeled_dataset(number_of_emotions, dataset):
    global file_number
    file_number = 0
    paths = __set_paths(dataset)
    path = paths[0]
    relative_path = paths[1]
    for filename in os.listdir(get_path(path, __file__)):
        emotion = __get_emotion(filename, dataset)
        if __check_emotion(number_of_emotions, emotion):
            try:
                emotions.append(__get_target_value(emotion))
                emotions_labels.append(__get_target(emotion))
                files.append(filename)

                # Extracting audio features using OpenSmile library
                __extract_audio_features(relative_path, filename)
            except ValueError:
                continue

    # Generating csv file with statistics (obtained with functionals configurations) of the selected features
    __export_functional_values(number_of_emotions, dataset)

    # Calculating mean after removing outliers
    __calculate_mean_for_audio_features()

    # Filling data dictionary
    audios_data['data'] = results_low_level_using_mean
    audios_data['target'] = emotions
    audios_data['target_names'] = emotions_labels
    audios_data['files'] = files
    audios_data['features'] = features

    # Exporting data to a json file
    __export_data_results_to_json(number_of_emotions, dataset)


def __calculate_audio_features_from_unlabeled_dataset(number_of_emotions, dataset):
    global file_number
    file_number = 0
    path = DATASET_DRAW_TALK_PATH
    relative_path = DATASET_DRAW_TALK_RELATIVE_PATH
    for root, dirs, dir_files in os.walk(get_path(path, __file__)):
        for file in dir_files:
            if file.endswith('.wav'):
                files.append(file)
                # Extracting audio features using OpenSmile library
                file_path = relative_path + str(root.split("\\")[len(root.split("\\")) - 1]) + "/"
                __extract_audio_features(file_path, file)

    # Generating csv file with statistics (obtained with functionals configurations) of the selected features
    __export_functional_values(number_of_emotions, dataset)

    # Calculating mean after removing outliers
    __calculate_mean_for_audio_features()

    # Filling data dictionary
    audios_data['data'] = results_low_level_using_mean
    audios_data['files'] = files
    audios_data['features'] = features

    # Exporting data to a json file
    __export_data_results_to_json(number_of_emotions, dataset)


def __get_audio_features_data(audio_data_filename):
    global audios_data
    audio_data_file = open(audio_data_filename)
    audios_data = json.load(audio_data_file)
    audio_data_file.close()


def get_audio_features(dataset=None, number_of_emotions=None, multiple_dataset=False):
    global audios_data
    try:
        if multiple_dataset:
            data_combined = {'data': [], 'target': [], 'target_names': [], 'files': [], 'features': []}

            audio_data_mesd_filename = OUTPUT_LOW_LEVEL_PATH + "data_audios_" + str(number_of_emotions) + "_mesd.json"
            audio_data_iesc_child_filename = OUTPUT_LOW_LEVEL_PATH + "data_audios_" + str(
                number_of_emotions) + "_iesc_child.json"

            __get_audio_features_data(audio_data_mesd_filename)
            data_mesd = audios_data
            __get_audio_features_data(audio_data_iesc_child_filename)
            data_iesc_child = audios_data

            data_combined["data"].extend(data_mesd["data"])
            data_combined["target"].extend(data_mesd["target"])
            data_combined["target_names"].extend(data_mesd["target_names"])
            data_combined["files"].extend(data_mesd["files"])
            data_combined["features"].extend(data_mesd["features"])

            data_combined["data"].extend(data_iesc_child["data"])
            data_combined["target"].extend(data_iesc_child["target"])
            data_combined["target_names"].extend(data_iesc_child["target_names"])
            data_combined["files"].extend(data_iesc_child["files"])

            audios_data = data_combined
        else:
            if dataset == DRAW_TALK_DATASET_NAME:
                audio_data_filename = OUTPUT_LOW_LEVEL_PATH + "data_audios_" + dataset + ".json"
            else:
                audio_data_filename = OUTPUT_LOW_LEVEL_PATH + "data_audios_" + str(number_of_emotions) + "_" \
                                      + dataset + ".json"
            __get_audio_features_data(audio_data_filename)
    except FileNotFoundError:
        if dataset == DRAW_TALK_DATASET_NAME:
            __calculate_audio_features_from_unlabeled_dataset(number_of_emotions, dataset)
        else:
            __calculate_audio_features_from_labeled_dataset(number_of_emotions, dataset)
    finally:
        return audios_data
