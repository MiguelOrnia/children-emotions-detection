import json
import os
import logging as log
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from colormap import rgb2hex
from webcolors import hex_to_rgb, CSS3_HEX_TO_NAMES

""" Os functions """


def create_cleaned_data_directory(child_id, path_target, file):
    child_path = get_path(path_target, file) + "/" + child_id
    if not os.path.exists(child_path):
        os.makedirs(child_path)
    return child_path


def get_path(target_path, file):
    absolute_path = os.path.dirname(file)
    relative_path = target_path
    full_path = os.path.join(absolute_path, relative_path)
    return full_path


""" Drawing functions """


def color_to_df(input_value):
    colors_pre_list = str(input_value).replace('([(', '').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')', '') for i in colors_pre_list]

    # convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(", "")),
                           int(i.split(", ")[1]),
                           int(i.split(", ")[2].replace(")", ""))) for i in df_rgb]

    df = pd.DataFrame(zip(df_color_up, df_percent), columns=['c_code', 'occurence'])
    return df


def hex_to_color_name(hex_code):
    r, g, b = hex_to_rgb(hex_code)
    min_colors = {}
    for key, name in CSS3_HEX_TO_NAMES.items():
        cr, cg, cb = hex_to_rgb(key)
        color_diff = (abs(r - cr) + abs(g - cg) + abs(b - cb))
        min_colors[color_diff] = name
    return min_colors[min(min_colors.keys())]


""" Outliers function """


def remove_outliers(feature_data):
    q1 = np.percentile(feature_data, 25)
    q3 = np.percentile(feature_data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return feature_data[(feature_data >= lower_bound) & (feature_data <= upper_bound)]


""" Logger function """


def get_logger(name, console_log, file, subdirectory1, subdirectory2):
    # Check logger output
    if console_log:
        log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        # Logger to file
        sufix = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = name + "_" + sufix
        log.basicConfig(filename=os.path.join(get_path("", file), subdirectory1, subdirectory2, log_name + '.log'),
                        filemode='w',
                        level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    return log


""" Encoder class for exporting data to JSON file """


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
