""" Main Python file for the execution of the different scripts. """

from cleaning_data_module.audios_data_cleaner import audio_cleaner
from model_module.children_emotions_classifier import predict_using_existing_model
from model_module.train_children_audio_emotions_model import children_audio_emotions_classifier
from model_module.train_children_audio_emotions_meta_model import children_audio_emotions_meta_classifier

if __name__ == '__main__':
    # Cleaning audio data from draw&talk corpus
    """ audio_cleaner() """  # Only uncomment first time, no needed after the first time
    # Cleaning drawing data from draw&talk corpus
    """ drawing_cleaner() """  # Only uncomment first time, no needed after the first time

    # Classifying by Neutral, Positive and Negative using MESD corpus
    # children_audio_emotions_classifier('svc', 3, 'mesd')
    # children_audio_emotions_classifier('nn', 3, 'mesd')
    # children_audio_emotions_classifier('knn', 3, 'mesd')
    # children_audio_emotions_classifier('lr', 3, 'mesd')

    # Meta model using ensambling learning (stacking)
    # children_audio_emotions_meta_classifier(3, 'mesd')

    # Using an existing model to label an unlabelled corpus
    lsvm_features = ["Loudness_sma3",
                     "alphaRatio_sma3",
                     "hammarbergIndex_sma3",
                     "slope500-1500_sma3",
                     "spectralFlux_sma3",
                     "mfcc1_sma3",
                     "mfcc2_sma3",
                     "mfcc3_sma3",
                     "mfcc4_sma3",
                     "F0semitoneFrom27.5Hz_sma3nz",
                     "HNRdBACF_sma3nz",
                     "logRelF0-H1-H2_sma3nz",
                     "logRelF0-H1-A3_sma3nz",
                     "F1frequency_sma3nz",
                     "F1bandwidth_sma3nz",
                     "F1amplitudeLogRelF0_sma3nz",
                     "F2bandwidth_sma3nz",
                     "F2amplitudeLogRelF0_sma3nz",
                     "F3frequency_sma3nz",
                     "F3amplitudeLogRelF0_sma3nz",
                     "pcm_RMSenergy_sma",
                     "pcm_fftMag_spectralRollOff90.0_sma"]

    rsvm_features = ["Loudness_sma3",
                     "alphaRatio_sma3",
                     "hammarbergIndex_sma3",
                     "slope500-1500_sma3",
                     "spectralFlux_sma3",
                     "mfcc1_sma3",
                     "mfcc2_sma3",
                     "mfcc3_sma3",
                     "mfcc4_sma3",
                     "F0semitoneFrom27.5Hz_sma3nz",
                     "HNRdBACF_sma3nz",
                     "logRelF0-H1-H2_sma3nz",
                     "logRelF0-H1-A3_sma3nz",
                     "F1frequency_sma3nz",
                     "F1bandwidth_sma3nz",
                     "F1amplitudeLogRelF0_sma3nz",
                     "F2bandwidth_sma3nz",
                     "F2amplitudeLogRelF0_sma3nz",
                     "F3frequency_sma3nz",
                     "F3amplitudeLogRelF0_sma3nz",
                     "pcm_RMSenergy_sma",
                     "pcm_fftMag_spectralRollOff90.0_sma"]

    mlp_features = ["Loudness_sma3",
                    "alphaRatio_sma3",
                    "hammarbergIndex_sma3",
                    "slope500-1500_sma3",
                    "spectralFlux_sma3",
                    "mfcc1_sma3",
                    "mfcc2_sma3",
                    "mfcc3_sma3",
                    "mfcc4_sma3",
                    "F0semitoneFrom27.5Hz_sma3nz",
                    "HNRdBACF_sma3nz",
                    "logRelF0-H1-H2_sma3nz",
                    "logRelF0-H1-A3_sma3nz",
                    "F1frequency_sma3nz",
                    "F1bandwidth_sma3nz",
                    "F1amplitudeLogRelF0_sma3nz",
                    "F2bandwidth_sma3nz",
                    "F2amplitudeLogRelF0_sma3nz",
                    "F3frequency_sma3nz",
                    "F3amplitudeLogRelF0_sma3nz",
                    "pcm_RMSenergy_sma",
                    "pcm_fftMag_spectralRollOff90.0_sma"]

    knn_features = ["Loudness_sma3",
                    "alphaRatio_sma3",
                    "hammarbergIndex_sma3",
                    "slope0-500_sma3",
                    "slope500-1500_sma3",
                    "spectralFlux_sma3",
                    "mfcc1_sma3",
                    "mfcc2_sma3",
                    "mfcc3_sma3",
                    "mfcc4_sma3",
                    "F0semitoneFrom27.5Hz_sma3nz",
                    "jitterLocal_sma3nz",
                    "shimmerLocaldB_sma3nz",
                    "HNRdBACF_sma3nz",
                    "logRelF0-H1-H2_sma3nz",
                    "logRelF0-H1-A3_sma3nz",
                    "F1frequency_sma3nz",
                    "F1bandwidth_sma3nz",
                    "F1amplitudeLogRelF0_sma3nz",
                    "F2frequency_sma3nz",
                    "F2bandwidth_sma3nz",
                    "F2amplitudeLogRelF0_sma3nz",
                    "F3frequency_sma3nz",
                    "F3bandwidth_sma3nz",
                    "F3amplitudeLogRelF0_sma3nz",
                    "pcm_zcr_sma",
                    "pcm_RMSenergy_sma",
                    "pcm_fftMag_spectralCentroid_sma",
                    "pcm_fftMag_spectralRollOff25.0_sma",
                    "pcm_fftMag_spectralRollOff50.0_sma",
                    "pcm_fftMag_spectralRollOff75.0_sma",
                    "pcm_fftMag_spectralRollOff90.0_sma"]

    lr_features = ["Loudness_sma3",
                   "alphaRatio_sma3",
                   "hammarbergIndex_sma3",
                   "slope500-1500_sma3",
                   "spectralFlux_sma3",
                   "mfcc1_sma3",
                   "mfcc2_sma3",
                   "mfcc3_sma3",
                   "mfcc4_sma3",
                   "F0semitoneFrom27.5Hz_sma3nz",
                   "HNRdBACF_sma3nz",
                   "logRelF0-H1-H2_sma3nz",
                   "logRelF0-H1-A3_sma3nz",
                   "F1frequency_sma3nz",
                   "F1bandwidth_sma3nz",
                   "F1amplitudeLogRelF0_sma3nz",
                   "F2bandwidth_sma3nz",
                   "F2amplitudeLogRelF0_sma3nz",
                   "F3frequency_sma3nz",
                   "F3amplitudeLogRelF0_sma3nz",
                   "pcm_RMSenergy_sma",
                   "pcm_fftMag_spectralRollOff90.0_sma"]

    psvm_features = ["Loudness_sma3",
                     "alphaRatio_sma3",
                     "hammarbergIndex_sma3",
                     "slope500-1500_sma3",
                     "spectralFlux_sma3",
                     "mfcc1_sma3",
                     "mfcc2_sma3",
                     "mfcc3_sma3",
                     "mfcc4_sma3",
                     "F0semitoneFrom27.5Hz_sma3nz",
                     "HNRdBACF_sma3nz",
                     "logRelF0-H1-H2_sma3nz",
                     "logRelF0-H1-A3_sma3nz",
                     "F1frequency_sma3nz",
                     "F1bandwidth_sma3nz",
                     "F1amplitudeLogRelF0_sma3nz",
                     "F2bandwidth_sma3nz",
                     "F2amplitudeLogRelF0_sma3nz",
                     "F3frequency_sma3nz",
                     "F3amplitudeLogRelF0_sma3nz",
                     "pcm_RMSenergy_sma",
                     "pcm_fftMag_spectralRollOff90.0_sma"]

    meta_model_features = ["Loudness_sma3",
                           "alphaRatio_sma3",
                           "hammarbergIndex_sma3",
                           "slope500-1500_sma3",
                           "spectralFlux_sma3",
                           "mfcc1_sma3",
                           "mfcc2_sma3",
                           "mfcc3_sma3",
                           "mfcc4_sma3",
                           "F0semitoneFrom27.5Hz_sma3nz",
                           "HNRdBACF_sma3nz",
                           "logRelF0-H1-H2_sma3nz",
                           "logRelF0-H1-A3_sma3nz",
                           "F1frequency_sma3nz",
                           "F1bandwidth_sma3nz",
                           "F1amplitudeLogRelF0_sma3nz",
                           "F2bandwidth_sma3nz",
                           "F2amplitudeLogRelF0_sma3nz",
                           "F3frequency_sma3nz",
                           "F3amplitudeLogRelF0_sma3nz",
                           "pcm_RMSenergy_sma",
                           "pcm_fftMag_spectralRollOff90.0_sma"]

    meta_model_mesd_features = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3",
                                "slope500-1500_sma3", "spectralFlux_sma3", "mfcc1_sma3",
                                "mfcc2_sma3", "mfcc4_sma3", "shimmerLocaldB_sma3nz",
                                "HNRdBACF_sma3nz", "logRelF0-H1-H2_sma3nz", "F1frequency_sma3nz",
                                "F1bandwidth_sma3nz", "F2amplitudeLogRelF0_sma3nz",
                                "F3amplitudeLogRelF0_sma3nz", "pcm_RMSenergy_sma"]

    # predict_using_existing_model('model_lsvm_mesd_3.pkl', lsvm_features)
    # predict_using_existing_model('model_rsvm_mesd_3.pkl', rsvm_features)
    # predict_using_existing_model('model_psvm_mesd_3.pkl', psvm_features)
    # predict_using_existing_model('model_nn_mesd_3.pkl', mlp_features)
    # predict_using_existing_model('model_knn_mesd_3.pkl', knn_features)
    # predict_using_existing_model('model_lr_mesd_3.pkl', lr_features)
    # predict_using_existing_model('model_metamodel_mesd_3.pkl', meta_model_mesd_features)
    # predict_using_existing_model('model_metamodel_mesd+h&d_3.pkl', meta_model_features)
    # predict_using_existing_model('model_metamodel_h&d_3.pkl', meta_model_features)
