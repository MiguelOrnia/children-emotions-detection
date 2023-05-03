""" In this script we clean our children' audio files from UniOvi dataset """

import wave
import os
from pydub import AudioSegment
from util.helper import get_path, create_cleaned_data_directory

DIRECTORY_UNIOVI_DATA_SOURCE = "../datasets/uniovi_full_dataset"
DIRECTORY_UNIOVI_DATA_TARGET = "../datasets/uniovi_full_dataset_cleaned"
AUDIO_EXTENSION = ".wav"


def audio_cleaner():
    print("-- Starting with audio files cleaning process -- \n")
    for root, dirs, files in os.walk(get_path(DIRECTORY_UNIOVI_DATA_SOURCE, __file__)):
        audio_files = 0
        combined_audio = {}
        child_number = 0
        for file in files:
            if file.endswith(AUDIO_EXTENSION):
                current_audio = wave.open(os.path.join(root, file), "rb")
                t_audio = current_audio.getnframes() / current_audio.getframerate()
                if t_audio >= 1.0:
                    audio_files += 1
                    split_path = os.path.join(root, file).split("\\")
                    child_number = split_path[len(split_path) - 2]
                    if audio_files >= 1:
                        if combined_audio == {}:
                            combined_audio = AudioSegment.from_wav(os.path.join(root, file))
                        else:
                            print("--- CHILD: ", child_number, ". Combining multiple audio files (",
                                  audio_files, ") --- \n")
                            combined_audio += AudioSegment.from_wav(os.path.join(root, file))
        if combined_audio != {}:
            child_path = create_cleaned_data_directory(child_number, DIRECTORY_UNIOVI_DATA_TARGET, __file__)
            file_name = "audio_" + str(child_number) + AUDIO_EXTENSION
            print("--- CHILD: ", child_number, ". Exporting audio file: ", file_name, " --- \n")
            combined_audio.export(os.path.join(child_path, file_name), format="wav")
    print("-- Audio files cleaning process done -- \n")
