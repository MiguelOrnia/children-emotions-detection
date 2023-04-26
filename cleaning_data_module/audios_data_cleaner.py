import wave
import os
from pydub import AudioSegment
from util.os_helper import get_path


DIRECTORY_UNIOVI_DATA_SOURCE = "../datasets/uniovi_full_dataset"
DIRECTORY_UNIOVI_DATA_TARGET = "../datasets/uniovi_full_dataset_cleaned"
AUDIO_EXTENSION = ".wav"


def __create_cleaned_data_directory(child_id):
    child_path = get_path(DIRECTORY_UNIOVI_DATA_TARGET, __file__) + "/" + child_id
    if not os.path.exists(child_path):
        os.makedirs(child_path)
    return child_path


def audio_cleaner():
    uniovi_dataset = {}
    print("-- Starting with audio files cleaning process -- \n")
    for root, dirs, files in os.walk(get_path(DIRECTORY_UNIOVI_DATA_SOURCE)):
        audio_files = 0
        combined_audio = {}
        current_audio = {}
        child_number = 0
        for file in files:
            if file.endswith('.wav'):
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
            child_path = __create_cleaned_data_directory(child_number)
            file_name = "audio_" + str(child_number) + ".wav"
            print("--- CHILD: ", child_number, ". Exporting audio file: ", file_name, " --- \n")
            combined_audio.export(os.path.join(child_path, file_name), format="wav")
            current_audio = wave.open(os.path.join(child_path, file_name), "rb")
        uniovi_dataset[child_number] = current_audio
    print("-- Audio files cleaning process done -- \n")
