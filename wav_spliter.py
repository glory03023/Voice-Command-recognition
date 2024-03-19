import os
import sys
import numpy as np
from voice_util import (audio_bic_feat,
                         div_gauss,
                         vad_audio_segment,
                         trim_audio_with_sox,
                         trim_audio_ffmpeg,
                         audio_merge_with_sox)


def split_audio_file(audio_file, time_list):
    file_list = []
    split_dir = "split"
    if os.path.exists(split_dir):
        if 'win32' in sys.platform:
            os.system("rmdir {} /s /q".format(split_dir))
        else:
            os.system('rm -rf \"{}\"'.format(split_dir))
    os.mkdir(split_dir)
    for idx, [seg_st, seg_end] in enumerate(time_list):
        split_file = "split_{}.wav".format(idx)
        split_file = os.path.join(split_dir, split_file)
        # split_file_name = trim_audio_with_sox(audio_file, 16000, seg_st, seg_end, split_file)
        split_file_name = trim_audio_ffmpeg(audio_file, seg_st, seg_end, split_file)
        file_list.append("\"{}\"".format(split_file_name))
    return file_list


def get_enroll_audio(audio_file, enroll_file):
    speech_segment = vad_audio_segment(audio_file)
    split_list = split_audio_file(audio_file, speech_segment)
    # target_file = "merge.wav"
    return audio_merge_with_sox(enroll_file, split_list)


def get_verify_audio(audio_file, enroll_file):
    speech_segment = vad_audio_segment(audio_file, concate_gap=0.45)
    split_list = split_audio_file(audio_file, speech_segment)
    # target_file = "merge.wav"
    return audio_merge_with_sox(enroll_file, split_list)


def audio_segmentation(audio_file):
    energy, mfb, mfcc = audio_bic_feat(audio_file)
    win_size = 200
    frame_step = 0.01  # (s)
    segment = div_gauss(mfcc, win=win_size)
    borders = np.array(segment)*frame_step
    segmentation_result = []
    for i in range(len(borders) - 1):
        segmentation_result.append([borders[i], borders[i+1]])

    return segmentation_result


def multiple_verify(audio_file):
    from speaker_verification import SpeakerVerification
    import emo_recognizer as emr
    verifier = SpeakerVerification()
    em_verifier = emr.load_model()
    # audio_segments = audio_segmentation(audio_file)
    audio_segments = vad_audio_segment(audio_file, concate_gap=0.6)
    caller_list = []
    unknown_speaker_num = 0
    for seg_st, seg_end in audio_segments:
        if seg_end - seg_st > 50:
            seg_end -= 1
            seg_st += 1
        print("interval: {}".format([seg_st, seg_end]))
        # seg_file = trim_audio_with_sox(audio_file, 16000, seg_st, seg_end, out_file="merge.wav")
        seg_file = trim_audio_ffmpeg(audio_file, seg_st, seg_end, "merge.wav")
        # res_file  = get_enroll_audio(seg_file, "enroll.wav")
        called_name, sc = verifier.verify(seg_file)
        emotion = emr.emotion_recognizer(em_verifier, seg_file)
        if called_name != "Unknown":
            if len(caller_list) > 0 and caller_list[-1][0] == called_name:
                caller_list[-1][1] = (caller_list[-1][1] + sc) / 2
                caller_list[-1][3] = seg_end
            else:

                caller_list.append([called_name, sc, seg_st, seg_end, emotion[0]])
        else:
            unknown_speaker_num += 1
            caller_list.append(["{}-{}".format(called_name, unknown_speaker_num), sc, seg_st, seg_end, emotion[0]])

            # verify_speaker(verifier, seg_file)
    return caller_list


if __name__ == "__main__":
    wav_dir = "wav"
    audio_file = "Voice 164.wav"
    # audio_file = "Joe_training16.wav"
    # audio_file = "Ashline_training_16.wav"
    audio_file = os.path.join(wav_dir, audio_file)
    enroll_file = "merge.wav"
    # audio_segmentation(audio_file)
    multiple_verify(audio_file)
    # vad_audio_segment(audio_file)
    # get_enroll_audio(audio_file, enroll_file)
