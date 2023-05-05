from model_module.children_emotions_classifier import predict_using_existing_model
from model_module.train_children_audio_emotions_model import children_audio_emotions_classifier

if __name__ == '__main__':
    # Cleaning audio data from uniovi corpus
    """ audio_cleaner() """  # Only uncomment first time, no needed after the first time
    # Cleaning drawing data from uniovi corpus
    """ drawing_cleaner() """  # Only uncomment first time, no needed after the first time

    # Sorting by Positive and Negative using MESD corpus
    # children_audio_emotions_classifier('svc', 2, 'mesd')
    # children_audio_emotions_classifier('nn', 2, 'mesd')
    # children_audio_emotions_classifier('dt', 2, 'mesd')

    # Sorting by Neutral, Positive and Negative using MESD corpus
    # children_audio_emotions_classifier('svc', 3, 'mesd')
    # children_audio_emotions_classifier('nn', 3, 'mesd')
    # children_audio_emotions_classifier('dt', 3, 'mesd')

    # Sorting by Neutral, Positive and Negative using IESC-Child corpus
    children_audio_emotions_classifier('svc', 3, 'iesc_child')
    # children_audio_emotions_classifier('nn', 3, 'iesc_child')
    # children_audio_emotions_classifier('dt', 3, 'iesc_child')

    # Using an existing model_module to tag an untagged corpus

    """
    mesd_2_emotions_features = ["alphaRatio_sma3", "hammarbergIndex_sma3", "slope500-1500_sma3", "spectralFlux_sma3",
                                "mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3", "shimmerLocaldB_sma3nz", "HNRdBACF_sma3nz",
                                "logRelF0-H1-H2_sma3nz", "F1amplitudeLogRelF0_sma3nz", "F2amplitudeLogRelF0_sma3nz",
                                "F3amplitudeLogRelF0_sma3nz", "pcm_RMSenergy_sma"]

    mesd_3_emotions_features_svm = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "slope500-1500_sma3",
                                    "spectralFlux_sma3", "mfcc1_sma3", "mfcc2_sma3", "mfcc4_sma3",
                                    "shimmerLocaldB_sma3nz", "HNRdBACF_sma3nz", "logRelF0-H1-H2_sma3nz",
                                    "F1frequency_sma3nz", "F1bandwidth_sma3nz", "F2amplitudeLogRelF0_sma3nz",
                                    "F3amplitudeLogRelF0_sma3nz", "pcm_RMSenergy_sma"]

    mesd_3_emotions_features = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "slope0-500_sma3",
                                "slope500-1500_sma3", "spectralFlux_sma3", "mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3",
                                "mfcc4_sma3", "jitterLocal_sma3nz", "shimmerLocaldB_sma3nz", "HNRdBACF_sma3nz",
                                "logRelF0-H1-H2_sma3nz", "logRelF0-H1-A3_sma3nz", "F1frequency_sma3nz",
                                "F1bandwidth_sma3nz", "F1amplitudeLogRelF0_sma3nz", "F2bandwidth_sma3nz",
                                "F2amplitudeLogRelF0_sma3nz", "F3frequency_sma3nz", "F3amplitudeLogRelF0_sma3nz",
                                "pcm_RMSenergy_sma"]

    mesd_3_emotions_features_nn_60_40 = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3",
                                         "slope500-1500_sma3", "spectralFlux_sma3", "mfcc1_sma3", "mfcc2_sma3",
                                         "mfcc3_sma3", "mfcc4_sma3", "HNRdBACF_sma3nz", "logRelF0-H1-A3_sma3nz",
                                         "F1frequency_sma3nz", "F1bandwidth_sma3nz", "F2amplitudeLogRelF0_sma3nz",
                                         "F3frequency_sma3nz", "F3amplitudeLogRelF0_sma3nz", "pcm_RMSenergy_sma"]

    mesd_emotions_dt = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "slope0-500_sma3",
                        "slope500-1500_sma3", "spectralFlux_sma3", "mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3",
                        "mfcc4_sma3", "F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz", "shimmerLocaldB_sma3nz",
                        "HNRdBACF_sma3nz", "logRelF0-H1-H2_sma3nz", "logRelF0-H1-A3_sma3nz", "F1frequency_sma3nz",
                        "F1bandwidth_sma3nz", "F1amplitudeLogRelF0_sma3nz", "F2frequency_sma3nz",
                        "F2bandwidth_sma3nz", "F2amplitudeLogRelF0_sma3nz", "F3frequency_sma3nz",
                        "F3bandwidth_sma3nz", "F3amplitudeLogRelF0_sma3nz", "pcm_zcr_sma", "pcm_RMSenergy_sma",
                        "pcm_fftMag_spectralCentroid_sma", "pcm_fftMag_spectralRollOff25.0_sma",
                        "pcm_fftMag_spectralRollOff50.0_sma", "pcm_fftMag_spectralRollOff75.0_sma",
                        "pcm_fftMag_spectralRollOff90.0_sma"]

    iesc_child_2_emotions = ["Loudness_sma3", "hammarbergIndex_sma3", "slope0-500_sma3", "slope500-1500_sma3",
                             "spectralFlux_sma3", "mfcc1_sma3", "mfcc3_sma3", "mfcc4_sma3",
                             "F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz", "HNRdBACF_sma3nz",
                             "F1frequency_sma3nz", "F1bandwidth_sma3nz", "F2frequency_sma3nz", "F3frequency_sma3nz",
                             "F3amplitudeLogRelF0_sma3nz", "pcm_fftMag_spectralCentroid_sma",
                             "pcm_fftMag_spectralRollOff25.0_sma", "pcm_fftMag_spectralRollOff50.0_sma",
                             "pcm_fftMag_spectralRollOff75.0_sma"]

    predict_using_existing_model('model_nn_mesd_3.pkl', mesd_3_emotions_features)
    """
