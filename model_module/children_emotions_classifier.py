""" In this script we classify children' audio files from UniOvi dataset.
Additionally, we extract drawings features. """

import csv
import os

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from processing_audio_data_module.extracting_audio_features.audio_features_extractor import get_audio_features, \
    get_label_by_value
from processing_drawings_data_module.extracting_drawing_features.drawing_features_extractor import get_drawing_features
from util.helper import get_path


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
    audio_data = get_audio_features('uniovi')
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
    results['File'] = audio_data['files']
    results['Predominant_Color'] = drawing_data['predominant_color']
    results['Number_Of_Colors'] = drawing_data['number_of_colors']
    results['Colored_Surface'] = drawing_data['colored_surface']

    # Find the length of the longest list in the dictionary
    max_len = max([len(val) for val in results.values()])

    # Create the CSV file and write the header row
    with open('outputs/output_' + corpus + '_' + classifier_type + '_' + number_of_emotions + '_uniovi_dataset.csv',
              'w',
              newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(results.keys())

        # Write each row of data
        for i in range(max_len):
            row = [results[key][i] if i < len(results[key]) else '' for key in results.keys()]
            writer.writerow(row)
