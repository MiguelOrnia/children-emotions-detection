""" In this script we clean our children' drawing files from UniOvi dataset """

import os
import extcolors
from util.helper import get_path, create_cleaned_data_directory, color_to_df
import shutil

DIRECTORY_UNIOVI_DATA_SOURCE = "../datasets/uniovi_full_dataset"
DIRECTORY_UNIOVI_DATA_TARGET = "../datasets/uniovi_full_dataset_cleaned"
DRAWING_EXTENSION = ".png"


def drawing_cleaner():
    print("-- Starting with drawing files cleaning process -- \n")
    for root, dirs, files in os.walk(get_path(DIRECTORY_UNIOVI_DATA_SOURCE, __file__)):
        drawings = []
        count_files = 0
        file_to_export = ''
        child_number = 0
        for file in files:
            count_files += 1
            if file.endswith(DRAWING_EXTENSION):
                colors_x = extcolors.extract_from_path(os.path.join(root, file), tolerance=12, limit=13)
                df_color = color_to_df(colors_x)
                colors = list(df_color['c_code'])
                split_path = os.path.join(root, file).split("\\")
                child_number = split_path[len(split_path) - 2]
                if len(colors) > 0:
                    drawings.append(file)
                    file_to_export = file
                if len(drawings) > 1 and count_files == len(files):
                    max_file_number = -1
                    for drawing in drawings:
                        file_number = int(str(drawing).split('-')[1].split('.')[0])
                        if file_number > max_file_number:
                            max_file_number = file_number
                    file_to_export = [drawing for drawing in drawings if str(max_file_number) in drawing][0]
                    print("--- CHILD: ", child_number, ". Selecting the last drawing done by the child (",
                          str(file_to_export), ") --- \n")
            if file_to_export != '' and count_files == len(files):
                child_path = create_cleaned_data_directory(child_number, DIRECTORY_UNIOVI_DATA_TARGET, __file__)
                file_name = "drawing_" + str(child_number) + DRAWING_EXTENSION
                print("--- CHILD: ", child_number, ". Exporting drawing file: ", file_name, " --- \n")
                shutil.copy(os.path.join(root, file_to_export), os.path.join(child_path, file_name))
    print("-- Drawing files cleaning process done -- \n")
