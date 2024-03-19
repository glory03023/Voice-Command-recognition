import os
from dtw import *
import librosa
import numpy as np
from numpy.linalg import norm
#from prepare_data import vad_audio_segments

from voice_util import detect_audio_segment, read_voice_file, total_energy_list, convert2wav, trim_audio_with_sox



def vad_audio_segments(audio_file, sample_rate=16000):
    #sample_rate = 16000
    frame_win = sample_rate*0.05
    frame_step = sample_rate * 0.05
    tr = 0.0001

    wav_data = read_voice_file(audio_file, sample_rate=sample_rate)
    engery_list = total_energy_list(wav_data, sample_rate, frame_win, frame_step)
    audio_segs, sil_segs = detect_audio_segment(engery_list, 0.05, tr)
    return audio_segs


def mfcc_from_audio(audio_data, sample_rate):
    #frame_len = 400
    #frame_step = 160
    frame_len = int(0.025*sample_rate)
    frame_step = int(0.01*sample_rate)
    #y, sr = librosa.load(audio_file, sr=sample_rate)
    y = audio_data
    sr = sample_rate
    assert sr == sample_rate
    y = normalize_sig(y)
    energy = librosa.feature.rms(y, frame_length=frame_len, hop_length=frame_step)
    S, phase = librosa.magphase(librosa.stft(y, win_length=frame_len, hop_length=frame_step))
    rms = librosa.feature.rms(S=S)
    mfcc = librosa.feature.mfcc(y, sr, n_fft=frame_len, hop_length=frame_step, n_mels=40, n_mfcc=19)
    mfcc = mfcc[1:, :]

    return mfcc, energy, rms


def mfcc_from_file(audio_file, sample_rate):
    #frame_len = 400
    #frame_step = 160
    frame_len = int(0.025*sample_rate)
    frame_step = int(0.01*sample_rate)
    y, sr = librosa.load(audio_file, sr=sample_rate)
    assert sr == sample_rate
    y = normalize_sig(y)
    energy = librosa.feature.rms(y, frame_length=frame_len, hop_length=frame_step)
    S, phase = librosa.magphase(librosa.stft(y, win_length=frame_len, hop_length=frame_step))
    rms = librosa.feature.rms(S=S)
    mfcc = librosa.feature.mfcc(y, sr, n_fft=frame_len, hop_length=frame_step, n_mels=40, n_mfcc=19)
    mfcc = mfcc[1:, :]

    return mfcc, energy, rms


def normalize_sig(wav_data):
    signal = np.double(wav_data)
    # signal = signal / (2.0 ** 15)
    assert isinstance(signal, np.ndarray)
    sig_mean = signal.mean()
    #sig_std = signal.std()
    sig_max = (np.abs(signal)).max()
    #signal = (signal - sig_mean) / (sig_std + 0.0000000001)
    #signal = (signal - sig_mean)
    signal = (signal - sig_mean) / (sig_max + 0.0000000001)
    return signal


def dtw_dist(audio_file1, audio_file2):
    mfcc1, energy1, rms1 = mfcc_from_file(audio_file1, sample_rate=8000)
    mfcc2, energy2, rms2 = mfcc_from_file(audio_file2, sample_rate=8000)
    speech_segs = vad_audio_segments(audio_file1, sample_rate=8000)
    if speech_segs:
        seg_st = int(speech_segs[0][0] / 0.01)
        seg_end = int(speech_segs[-1][1] / 0.01)
        mfcc1 = mfcc1[:, seg_st:seg_end]

    speech_segs = vad_audio_segments(audio_file2, sample_rate=8000)
    if speech_segs:
        seg_st = int(speech_segs[0][0] / 0.01)
        seg_end = int(speech_segs[-1][1] / 0.01)
        mfcc2 = mfcc2[:, seg_st:seg_end]

    alignmentOBE = dtw(mfcc1.T, mfcc2.T)
    return alignmentOBE, mfcc1, mfcc2


def get_similarity(audio_file1, audio_file2):
    alignmentOBE, mfcc1, mfcc2 = dtw_dist(audio_file1, audio_file2)
    # print ('Normalized distance between the two sounds: {}'.format(dist))
    similarity = []
    for idx1, idx2 in zip(alignmentOBE.index1.tolist(), alignmentOBE.index2.tolist()):
        #print(f"idex1: {idx1}, index2: {idx2}")
        similarity.append(mfcc1.T[idx1] - mfcc2.T[idx2])
    return similarity


def similarity_mfcc(mfcc1, mfcc2):
    alignmentOBE = dtw(mfcc1.T, mfcc2.T)
    # print ('Normalized distance between the two sounds: {}'.format(dist))
    similarity = []
    for idx1, idx2 in zip(alignmentOBE.index1.tolist(), alignmentOBE.index2.tolist()):
        #print(f"idex1: {idx1}, index2: {idx2}")
        similarity.append(mfcc1.T[idx1] - mfcc2.T[idx2])
    return similarity


def train_model(train_dir):
    total_file_map = {}
    for cmd_name in os.listdir(train_dir):
        cmd_dir = os.path.join(train_dir, cmd_name)
        if not os.path.isdir(cmd_dir):
            continue
        total_file_map[cmd_name] = []
        for wav_file in os.listdir(cmd_dir):
            if wav_file.find(".wav") == -1:
                continue
            wav_file = os.path.join(cmd_dir, wav_file)
            total_file_map[cmd_name].append(wav_file)

    # same commands
    same_commands = []
    all_cmd_names = []
    for cmd_name in total_file_map:
        all_cmd_names.append(cmd_name)
        cmd_file_list = total_file_map[cmd_name]
        for i in range(0, len(cmd_file_list)-1, 1):
            for j in range(1, len(cmd_file_list), 1):
                same_commands.append([cmd_file_list[i], cmd_file_list[j]])

    # different commands
    diff_commands = []
    for i in range(0, len(all_cmd_names) - 1):
        prev_file_list = total_file_map[all_cmd_names[i]]
        np.random.shuffle(prev_file_list)
        prev_file_list = prev_file_list[:int(len(prev_file_list)/3)]
        for j in range(1, len(all_cmd_names)):
            next_file_list = total_file_map[all_cmd_names[j]]
            np.random.shuffle(next_file_list)
            next_file_list = next_file_list[:int(len(next_file_list)/4)]
            for perv_file in prev_file_list:
                for next_file in next_file_list:
                    diff_commands.append([perv_file, next_file])

    print("same command pairs: {}".format(len(same_commands)))
    print("different command pairs: {}".format(len(diff_commands)))

    # train same command
    train_same_data = []
    for audio_file1, audio_file2 in same_commands:
        train_same_data += get_similarity(audio_file1, audio_file2)

    print("train same samples: {}".format(len(train_same_data)))
    np.savez("voice_command.npz", mfcc=np.array(train_same_data))
    train_same_data = []

    # train different command
    train_diff_data = []
    for audio_file1, audio_file2 in diff_commands:
        train_diff_data += get_similarity(audio_file1, audio_file2)
    print("train different samples: {}".format(len(train_diff_data)))
    np.savez("diff_command.npz", mfcc=np.array(train_diff_data))


if __name__ == "__main__":
    train_dir = "data/train"
    train_model(train_dir)
