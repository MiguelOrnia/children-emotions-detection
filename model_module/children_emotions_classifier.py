""" In this script we classify children' audio files from Draw&Talk dataset.
Additionally, we extract drawings features. """

import csv
import os
from collections import defaultdict
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score
from processing_audio_data_module.extracting_audio_features.audio_features_extractor import get_audio_features, \
    get_label_by_value
from processing_drawings_data_module.extracting_drawing_features.drawing_features_extractor import get_drawing_features
from util.helper import get_path
from model_module.train_children_audio_emotions_model import get_data2

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


# Uncomment only when need two dataset with different proportions
def filter_audio_features():
    audio_data = get_audio_features('draw_talk', 3)
    audio_data['data'] = [audio for audio in audio_data['data'] if audio in get_data2()[0]]
    audio_data['target'] = get_data2()[1]
    audio_data['files'] = [file for file in audio_data['files'] if file in get_data2()[2]]
    print(get_data2()[1])
    return audio_data


def predict_using_existing_model(name_model, model_features):
    feature_ids = []
    predictions_labels = []
    predictions_probs = []
    results = {}
    scaler = MinMaxScaler()
    classifier_type = name_model.split('_')[1]
    corpus = name_model.split('_')[2]
    # number_of_emotions = name_model.split('_')[3].split('.')[0]
    number_of_emotions = 3
    path_to_model = os.path.join(get_path("", __file__),
                                 "training_results/models/" + classifier_type + "/" + name_model)
    model = joblib.load(path_to_model)
    audio_data = filter_audio_features()
    feature_position = 0
    for feature in audio_data['features']:
        if feature in model_features:
            feature_ids.append(feature_position)
        feature_position += 1
    model_data = np.array(audio_data['data'])
    data_filtered = model_data[:, feature_ids]
    target_filtered = audio_data['target']
    data_scaled = scaler.fit_transform(data_filtered)
    probability_predictions = model.predict_proba(data_scaled)
    predictions = model.predict(data_scaled)
    for i in range(len(data_scaled)):
        predictions_labels.append(get_label_by_value(predictions[i]))
        predictions_probs.append(probability_predictions[i])

    drawing_data = get_drawing_features(audio_data['files'])
    results['Child_Id'] = __get_children_ids(audio_data['files'])
    results['Emotion_Value'] = predictions.tolist()
    print(results['Emotion_Value'])
    results['Emotion'] = predictions_labels
    results['Negative_Probability'] = [probs[1] for probs in predictions_probs]
    results['Positive_Probability'] = [probs[0] for probs in predictions_probs]
    if int(number_of_emotions) == 3:
        results['Neutral_Probability'] = [probs[2] for probs in predictions_probs]

    for color, value in __get_all_colors(drawing_data['colors']).items():
        results[color] = value
    results['Number_Of_Colors'] = drawing_data['number_of_colors']
    results['Colored_Surface'] = drawing_data['colored_surface']

    # Find the length of the longest list in the dictionary
    max_len = max([len(val) for val in results.values()])

    # Create the CSV file and write the header row
    with open(
            OUTPUT_PATH + 'output_' + corpus + '_' + classifier_type + '_' + str(number_of_emotions) +
            '_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(results.keys())

        # Write each row of data
        for i in range(max_len):
            row = [results[key][i] if i < len(results[key]) else '' for key in results.keys()]
            writer.writerow(row)

    # Calculate metrics
    accuracy = accuracy_score(target_filtered, results['Emotion_Value'])
    recall = recall_score(target_filtered, results['Emotion_Value'], average='micro')
    f1 = f1_score(target_filtered, results['Emotion_Value'], average='micro')

    # Confusion matrix
    cm = confusion_matrix(target_filtered, results['Emotion_Value'])

    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Classes
    classes = ['Positive', 'Negative', 'Neutral']

    # Create a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)

    # Adding titles and tags
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.title('Confusion matrix normalized')

    # Show the graph
    plt.show()

    # Show results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
