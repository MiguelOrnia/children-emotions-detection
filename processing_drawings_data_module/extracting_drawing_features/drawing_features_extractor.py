""" In this script we are able to get drawing features from children' drawing files (png) """

import json
import extcolors
import numpy as np
import colorspacious
from skimage.color import deltaE_ciede2000
from webcolors import hex_to_rgb

from util.helper import color_to_df, get_path, hex_to_color_name, NpEncoder

DATASET_DRAW_TALK_PATH = "../../datasets/draw_talk_full_dataset_cleaned"
OUTPUT_PATH = "processing_drawings_data_module/extracting_drawing_features/outputs/"
DRAW_TALK_DATASET_NAME = "draw_talk"

WHITE_RGB = (255, 255, 255)

drawings_data = {}
deleted_colors = []
actual_colors = []


def __merge_similar_colors(color_dict, threshold=5.0):
    global deleted_colors
    merged_dict = {}
    deleted_colors = []
    color_list = list(color_dict.keys())
    n_colors = len(color_list)
    similarities = np.zeros((n_colors, n_colors))
    for i, color1 in enumerate(color_list):
        for j, color2 in enumerate(color_list):
            if j <= i:
                continue
            similarity = __color_similarity(color1, color2)
            similarities[i, j] = similarity
            similarities[j, i] = similarity
    for i, color in enumerate(color_list):
        for j in range(i + 1, n_colors):
            if threshold > similarities[i, j] > 0.0:
                __merge_colors(color, color_list[j], merged_dict, color_dict)
    for i, color in enumerate(color_list):
        if color not in deleted_colors and color not in merged_dict:
            merged_dict[color] = color_dict[color]
    return merged_dict


def __max_occurrence_color(color1, color2, color_dict):
    if color_dict[color1] > color_dict[color2]:
        deleted_colors.append(color2)
        return color1
    else:
        deleted_colors.append(color1)
        return color2


def __merge_colors(color1, color2, merged_dict, color_dict):
    merged_color = __max_occurrence_color(color1, color2, color_dict)
    if merged_color not in deleted_colors:
        if merged_color in merged_dict:
            merged_dict[merged_color] += color_dict[color2]
        else:
            merged_dict[merged_color] = color_dict[color1]
            merged_dict[merged_color] += color_dict[color2]


def __rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(*[int(round(x)) for x in rgb_color])


def __check_colors_similarities(color, threshold=5.0):
    global actual_colors
    similar_color = False
    target_color = ''
    if len(actual_colors) == 0:
        actual_colors.append(color)
    else:
        for actual_color in actual_colors:
            if __color_similarity(actual_color, color) < threshold:
                similar_color = True
                target_color = actual_color
    if similar_color:
        return target_color
    elif color not in actual_colors:
        actual_colors.append(color)
    return color


def __iterate_colors_checking_similarities(colors_occurrences):
    color_occurrences_aux = {}
    for color, occurrence in colors_occurrences.items():
        color_checked = __check_colors_similarities(color)
        color_occurrences_aux[color_checked] = colors_occurrences[color]
    return color_occurrences_aux


def __color_similarity(color1, color2):
    # Convert hex colors to RGB tuples
    color1_rgb = hex_to_rgb(color1)
    color2_rgb = hex_to_rgb(color2)

    # Convert RGB tuples to LabColor objects
    color1_lab = colorspacious.cspace_convert(color1_rgb, "sRGB255", "CAM02-UCS")
    color2_lab = colorspacious.cspace_convert(color2_rgb, "sRGB255", "CAM02-UCS")

    # Compute CIEDE2000 color difference
    delta_e_value = deltaE_ciede2000(color1_lab, color2_lab)

    return delta_e_value


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


def __get_colors(colors_hex_occurrence):
    colors = [occurrence for color_hex, occurrence in colors_hex_occurrence.items()]
    return np.array(colors)


def __export_data_results_to_json(dataset):
    with open(OUTPUT_PATH + "data_drawings_" + dataset + ".json", "w") as outfile:
        json.dump(drawings_data, outfile, cls=NpEncoder)


def __calculate_drawing_features(drawing_file_path):
    colors_x = extcolors.extract_from_path(get_path(drawing_file_path, __file__), tolerance=15)
    df_color = color_to_df(colors_x)
    colors = list(df_color['c_code'])
    colors_occurrences = [int(i) for i in list(df_color['occurence'])]
    colors_hex_occurrence = {color_hex: round(occurrence / sum(colors_occurrences), 3) for color_hex, occurrence in
                             zip(colors, colors_occurrences)}
    color_hex_occurrence_filtered = {color_name: occurrence for color_name, occurrence in colors_hex_occurrence.items()
                                     if color_name.lower() != __get_white_hex().lower() and occurrence > 0.004}

    drawings_data['colors'].append(
        __iterate_colors_checking_similarities(__merge_similar_colors(color_hex_occurrence_filtered)))
    drawings_data['predominant_color'].append(__get_color_with_most_occurrences(color_hex_occurrence_filtered))
    drawings_data['number_of_colors'].append(len(__get_colors(color_hex_occurrence_filtered)))
    drawings_data['colored_surface'].append(round(sum(__get_colors(color_hex_occurrence_filtered)), 3))


def get_drawing_features(drawings_files):
    global drawings_data
    drawings_data = {"colors": [], "predominant_color": [], "number_of_colors": [], "colored_surface": []}
    try:
        drawings_data_filename = OUTPUT_PATH + "data_drawings_" + DRAW_TALK_DATASET_NAME + ".json"
        drawings_data_file = open(drawings_data_filename)
        drawings_data = json.load(drawings_data_file)
        drawings_data_file.close()
    except FileNotFoundError:
        number_of_files = 0
        for drawing_file in drawings_files:
            drawing_file_name = drawing_file.replace("audio", "drawing")
            drawing_file_name_with_extension = drawing_file_name.replace(".wav", ".png")
            drawing_file_path = DATASET_DRAW_TALK_PATH + "/" + drawing_file.split("_")[1].split(".")[
                0] + "/" + drawing_file_name_with_extension
            __calculate_drawing_features(drawing_file_path)
            number_of_files += 1
            if len(drawings_files) == number_of_files:
                __export_data_results_to_json(DRAW_TALK_DATASET_NAME)
    finally:
        return drawings_data
