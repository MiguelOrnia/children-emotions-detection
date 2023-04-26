# from cleaning_data_module.audios_data_cleaner import audio_cleaner
from processing_audio_data_module.children_audio_emotions_classifier \
    import children_audio_emotions_classifier

if __name__ == '__main__':
    # Cleaning audio data from uniovi corpus
    """ audio_cleaner() """  # Only uncomment first time, no needed after the first time

    # Sorting by Neutral, Positive and Negative using MESD corpus
    # children_audio_emotions_classifier('svc', 3, 'mesd')
    # children_audio_emotions_classifier('nn', 3, 'mesd')
    # children_audio_emotions_classifier('dt', 3, 'mesd')

    # Sorting by Anger, Disgust, Fear, Happiness and Sadness using MESD corpus
    # children_audio_emotions_classifier('svc', 5, 'mesd')
    # children_audio_emotions_classifier('nn', 5, 'mesd')
    # children_audio_emotions_classifier('dt', 5, 'mesd')

    children_audio_emotions_classifier('svc', 3, 'iesc_child')
