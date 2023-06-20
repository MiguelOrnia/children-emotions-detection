""" In this script we classify children' audio files from Draw&Talk dataset.
Additionally, we extract drawings features. """

import csv
import os
from collections import defaultdict
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from processing_audio_data_module.extracting_audio_features.audio_features_extractor import get_audio_features, \
    get_label_by_value
from processing_drawings_data_module.extracting_drawing_features.drawing_features_extractor import get_drawing_features
from util.helper import get_path

OUTPUT_PATH = "model_module/outputs/"


def __get_children_ids(files):
    ids = []
    for file in files:
        ids.append(str(file.split("_")[1].split(".")[0]))
    return ids


def __get_all_colors(colors_with_occurrences):
    # Get a set of all the colors in the dictionaries
    colors = set()
    for drawing in colors_with_occurrences:
        for color in drawing.keys():
            colors.add(color)

    # Initialize the dictionary of color - occurrences with default values of 0.00
    initial_colors = defaultdict(lambda: [0.00] * len(colors_with_occurrences))

    # Fill in the values from the dictionaries
    for i, drawing in enumerate(colors_with_occurrences):
        for color, value in drawing.items():
            initial_colors[color][i] = value

    # Convert the defaultdict to a regular dictionary
    initial_colors = dict(initial_colors)

    return initial_colors


def predict_using_existing_model(name_model, model_features):
    feature_ids = []
    predictions_labels = []
    results = {}
    scaler = MinMaxScaler()
    classifier_type = name_model.split('_')[1]
    corpus = name_model.split('_')[2]
    number_of_emotions = name_model.split('_')[3].split('.')[0]
    path_to_model = os.path.join(get_path("", __file__),
                                 "training_results/models/" + classifier_type + "/" + name_model)
    model = joblib.load(path_to_model)
    audio_data = get_audio_features('draw_talk')
    feature_position = 0
    for feature in audio_data['features']:
        if feature in model_features:
            feature_ids.append(feature_position)
        feature_position += 1
    model_data = np.array(audio_data['data'])
    data_filtered = model_data[:, feature_ids]
    data_scaled = scaler.fit_transform(data_filtered)
    predictions = model.predict(data_scaled)
    for prediction in predictions:
        predictions_labels.append(get_label_by_value(prediction))
    drawing_data = get_drawing_features(audio_data['files'])
    results['Emotion'] = predictions_labels
    results['Emotion_Value'] = predictions.tolist()
    results['Child_Id'] = __get_children_ids(audio_data['files'])
    for color, value in __get_all_colors(drawing_data['colors']).items():
        results[color] = value
    results['Predominant_Color'] = drawing_data['predominant_color']
    results['Number_Of_Colors'] = drawing_data['number_of_colors']
    results['Colored_Surface'] = drawing_data['colored_surface']

    # Find the length of the longest list in the dictionary
    max_len = max([len(val) for val in results.values()])

    # Create the CSV file and write the header row
    with open(
            OUTPUT_PATH + 'output_' + corpus + '_' + classifier_type + '_' + number_of_emotions +
            '_draw_talk_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(results.keys())

        # Write each row of data
        for i in range(max_len):
            row = [results[key][i] if i < len(results[key]) else '' for key in results.keys()]
            writer.writerow(row)
