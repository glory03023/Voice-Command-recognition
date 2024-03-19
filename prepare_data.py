import random
import os
import shutil

from voice_util import detect_audio_segment, read_voice_file, total_energy_list, convert2wav, trim_audio_with_sox


def convert_wavs(wav_dir, out_dir):
    for wav_file in os.listdir(wav_dir):
        if wav_file.find(".wav") == -1:
            continue

        wav_in_path = os.path.join(wav_dir, wav_file)
        wav_out_path = os.path.join(out_dir, wav_file)
        convert2wav(wav_in_path, wav_out_path)


def vad_audio_segments(audio_file, sample_rate=16000):
    #sample_rate = 16000
    frame_win = sample_rate*0.05
    frame_step = sample_rate * 0.05
    tr = 0.0001

    wav_data = read_voice_file(audio_file, sample_rate=sample_rate)
    engery_list = total_energy_list(wav_data, sample_rate, frame_win, frame_step)
    audio_segs, sil_segs = detect_audio_segment(engery_list, 0.05, tr)
    return audio_segs


def vad_stream_segments(audio_data, sample_rate):
    #sample_rate = 16000
    frame_win = sample_rate*0.05
    frame_step = sample_rate * 0.05
    tr = 0.0001

    engery_list = total_energy_list(audio_data, sample_rate, frame_win, frame_step)
    audio_segs, sil_segs = detect_audio_segment(engery_list, 0.05, tr)
    return audio_segs


def make_training_data(mono_dir, train_dir):
    sample_rate = 8000
    for wav_file in os.listdir(mono_dir):
        if wav_file.find(".wav") == -1:
            continue
        cmd_name = wav_file.split()[0]
        cmd_dir = os.path.join(train_dir, cmd_name)
        if os.path.exists(cmd_dir):
            shutil.rmtree(cmd_dir)

        os.makedirs(cmd_dir)

        # read and extract speaking time
        in_wav_file = os.path.join(mono_dir, wav_file)
        speech_segs = vad_audio_segments(in_wav_file)
        split_segs = []
        split_segs += speech_segs
        # get split segs
        for i in range(5):
            for seg_st, seg_end in speech_segs:
                seg_st -= random.random() / 3
                seg_st = max(0, seg_st)
                seg_end += random.random() / 3
                split_segs.append([seg_st, seg_end])

        for idx, audio_seg in enumerate(split_segs):
            split_file = "{}{:02}.wav".format(cmd_name, idx)
            split_file = os.path.join(cmd_dir, split_file)
            trim_audio_with_sox(in_wav_file, sample_rate, audio_seg[0], audio_seg[1], split_file)
            aa = 0


if __name__ == "__main__":
    stereo_dir = "data/wavs"
    mono_dir = "data/split_audio"
    train_dir = "data/train"
    #convert_wavs(stereo_dir, mono_dir)

    audio_file = "data/split_audio/abrir command .wav"
    #vad_audio_segments(audio_file)
    make_training_data(mono_dir, train_dir)
