""" In this script we are able to get drawing features from children' drawing files (png) """

import json
import extcolors
import numpy as np
from util.helper import color_to_df, get_path, hex_to_color_name, remove_outliers, NpEncoder

DATASET_UNIOVI_PATH = "../../datasets/uniovi_full_dataset_cleaned"
OUTPUT_PATH = "processing_drawings_data_module/extracting_drawings_features/outputs/"
UNIOVI_DATASET_NAME = "uniovi"

WHITE_RGB = (255, 255, 255)

drawings_data = {}


def __get_white_hex():
    return '#' + ''.join(hex(c)[2:].zfill(2) for c in WHITE_RGB).upper()


def __get_color_with_most_occurrences(colors_hex_occurrence):
    white_hex = __get_white_hex()
    exclude_white = white_hex
    max_hex_color = None
    max_color_frequency = float('-inf')

    for color_hex, color_frequency in colors_hex_occurrence.items():
        if color_hex == exclude_white:
            continue
        if color_frequency > max_color_frequency:
            max_color_frequency = color_frequency
            max_hex_color = color_hex

    return hex_to_color_name(max_hex_color)


def __get_colors(colors_hex_occurrence, removing_outliers=False):
    colors = [occurrence for color_hex, occurrence in colors_hex_occurrence.items() if
              color_hex != __get_white_hex() and occurrence != 0.0]
    if removing_outliers:
        filtered_colors = remove_outliers(np.array(colors))
    else:
        filtered_colors = np.array(colors)
    return filtered_colors


def __export_data_results_to_json(dataset):
    with open(OUTPUT_PATH + "data_drawings_" + dataset + ".json", "w") as outfile:
        json.dump(drawings_data, outfile, cls=NpEncoder)


def __calculate_drawing_features(drawing_file_path):
    colors_x = extcolors.extract_from_path(get_path(drawing_file_path, __file__), tolerance=12, limit=13)
    df_color = color_to_df(colors_x)
    colors = list(df_color['c_code'])
    colors_occurences = [int(i) for i in list(df_color['occurence'])]
    colors_hex_occurence = {color_hex: round(occurrence * 100 / sum(colors_occurences), 1) for color_hex, occurrence in
                            zip(colors, colors_occurences)}

    drawings_data['predominant_color'].append(__get_color_with_most_occurrences(colors_hex_occurence))
    drawings_data['number_of_colors'].append(len(__get_colors(colors_hex_occurence, True)))
    drawings_data['colored_surface'].append(round(sum(__get_colors(colors_hex_occurence)), 2))

    __export_data_results_to_json(UNIOVI_DATASET_NAME)


def get_drawing_features(drawings_files):
    global drawings_data
    drawings_data = {"predominant_color": [], "number_of_colors": [], "colored_surface": []}
    try:
        drawings_data_filename = OUTPUT_PATH + "data_drawings_" + UNIOVI_DATASET_NAME + ".json"
        drawings_data_file = open(drawings_data_filename)
        drawings_data = json.load(drawings_data_file)
        drawings_data_file.close()
    except FileNotFoundError:
        for drawing_file in drawings_files:
            drawing_file_name = drawing_file.replace("audio", "drawing")
            drawing_file_name_with_extension = drawing_file_name.replace(".wav", ".png")
            drawing_file_path = DATASET_UNIOVI_PATH + "/" + drawing_file.split("_")[1].split(".")[
                0] + "/" + drawing_file_name_with_extension
            __calculate_drawing_features(drawing_file_path)
    finally:
        return drawings_data
