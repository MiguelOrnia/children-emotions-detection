""" In this script we clean our children' audio files from Draw&Talk dataset """

import wave
import os
from pydub import AudioSegment
from util.helper import get_path, create_cleaned_data_directory, get_logger

DIRECTORY_DRAW_TALK_DATA_SOURCE = "../datasets/draw_talk_full_dataset"
DIRECTORY_DRAW_TALK_DATA_TARGET = "../datasets/draw_talk_full_dataset_cleaned"
AUDIO_EXTENSION = ".wav"

# For logging output
console_log = False
log = get_logger("cleaning_results", console_log, __file__, "cleaning_results", "logs")
total_seconds = 0


def audio_cleaner():
    global total_seconds
    log.info("-- Starting with audio files cleaning process --")
    for root, dirs, files in os.walk(get_path(DIRECTORY_DRAW_TALK_DATA_SOURCE, __file__)):
        audio_files = 0
        combined_audio = {}
        child_number = 0
        for file in files:
            if file.endswith(AUDIO_EXTENSION):
                current_audio = wave.open(os.path.join(root, file), "rb")
                t_audio = current_audio.getnframes() / current_audio.getframerate()
                if t_audio >= 1.0:
                    total_seconds = total_seconds + t_audio
                    audio_files += 1
                    split_path = os.path.join(root, file).split("\\")
                    child_number = split_path[len(split_path) - 2]
                    if audio_files >= 1:
                        if combined_audio == {}:
                            combined_audio = AudioSegment.from_wav(os.path.join(root, file))
                        else:
                            log.info("--- CHILD: " + str(child_number) + ". Combining multiple audio files (" +
                                     str(audio_files) + ") ---")
                            combined_audio += AudioSegment.from_wav(os.path.join(root, file))
        if combined_audio != {}:
            child_path = create_cleaned_data_directory(child_number, DIRECTORY_DRAW_TALK_DATA_TARGET, __file__)
            file_name = "audio_" + str(child_number) + AUDIO_EXTENSION
            log.info("--- CHILD: " + str(child_number) + ". Exporting audio file: " + str(file_name) + " ---")
            combined_audio.export(os.path.join(child_path, file_name), format=AUDIO_EXTENSION.split(".")[1])
    log.info("-- Audio files cleaning process done --")
    log.info("-- Total seconds: " + str(total_seconds) + " --")
    log.info("-----------------------------------------")
