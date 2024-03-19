import os
import numpy as np
from gmm_model import load_gmm_from_npz
from train_model import mfcc_from_file, similarity_mfcc
from prepare_data import vad_audio_segments
from voice_util import audio_merge_with_sox


def recognize_cmd(cmd_data, same_model, diff_model, audio_file):
    sample_rate = 8000
    mfcc_data, _, _ = mfcc_from_file(audio_file, sample_rate=sample_rate)
    speech_segs = vad_audio_segments(audio_file, sample_rate=sample_rate)
    if speech_segs:
        seg_st = int(speech_segs[0][0] / 0.01)
        seg_end = int(speech_segs[-1][1] / 0.01)
        mfcc_data = mfcc_data[:, seg_st:seg_end]

    recog_cmd = ""
    pre_score = 0
    for cmd_name in cmd_data:
        cmd_feat = cmd_data[cmd_name][1]
        similarity = similarity_mfcc(cmd_feat, mfcc_data)
        similarity = np.array(similarity)
        cmd_score = same_model.score(similarity) - diff_model.score(similarity)
        similarity = similarity_mfcc(mfcc_data, cmd_feat)
        similarity = np.array(similarity)
        cmd_score2 = same_model.score(similarity) - diff_model.score(similarity)
        #if cmd_score > 0.2 and cmd_score2 > 0.2:
        if cmd_score > 0.1:
            print(f"score1: {cmd_score}, score2: {cmd_score2}")
            if pre_score < cmd_score + cmd_score2:
                pre_score = cmd_score + cmd_score2
            recog_cmd = cmd_name
            break

    return recog_cmd


def recognize_command(train_dir):
    same_model = load_gmm_from_npz("models/same_model.npz")
    diff_model = load_gmm_from_npz("models/diff_model.npz")
    sample_rate = 8000

    same_cmd = {}
    diff_cmd = {}

    for cmd_name in os.listdir(train_dir):
        cmd_dir = os.path.join(train_dir, cmd_name)
        if not os.path.isdir(cmd_dir):
            continue

        file_idx = 0
        for wav_file in os.listdir(cmd_dir):
            if wav_file.find(".wav") == -1:
                continue
            wav_path = os.path.join(cmd_dir, wav_file)
            if file_idx == 0:
                # add file name
                same_cmd[cmd_name] = [wav_path]
                diff_cmd[cmd_name] = []
            else:
                diff_cmd[cmd_name].append(wav_path)
            file_idx += 1

    # calculate feature for command model
    for cmd_name in same_cmd:
        mfcc_feat, _, _ = mfcc_from_file(same_cmd[cmd_name][0], sample_rate=sample_rate)
        speech_segs = vad_audio_segments(same_cmd[cmd_name][0], sample_rate=sample_rate)
        assert speech_segs

        seg_st = int(speech_segs[0][0] / 0.01)
        seg_end = int(speech_segs[-1][1] / 0.01)
        mfcc_feat = mfcc_feat[:, seg_st:seg_end]
        # add acoustic feature
        same_cmd[cmd_name].append(mfcc_feat)
        # add segment time
        same_cmd[cmd_name].append(seg_end - seg_st)

    # recognize cmd
    total_files = 0
    recog_files = 0
    for cmd_name in diff_cmd:
        for wav_file in diff_cmd[cmd_name]:
            recog_cmd = recognize_cmd(same_cmd, same_model, diff_model, wav_file)
            print("cmd: {} --> result: {}".format(cmd_name, recog_cmd))
            total_files += 1
            if cmd_name == recog_cmd:
                recog_files += 1

    print("recognized {}/{}".format(recog_files, total_files))
    print("{} percents".format(recog_files *100.0/total_files))



def recognize_test(train_dir, test_dir):
    same_model = load_gmm_from_npz("models/same_model.npz")
    diff_model = load_gmm_from_npz("models/diff_model.npz")
    sample_rate = 8000

    same_cmd = {}
    diff_cmd = {}

    for cmd_name in os.listdir(train_dir):
        cmd_dir = os.path.join(train_dir, cmd_name)
        if not os.path.isdir(cmd_dir):
            continue

        file_idx = 0
        for wav_file in os.listdir(cmd_dir):
            if wav_file.find(".wav") == -1:
                continue
            wav_path = os.path.join(cmd_dir, wav_file)
            if file_idx == 0:
                # add file name
                same_cmd[cmd_name] = [wav_path]
            file_idx += 1
            if file_idx == 1:
                break

    # calculate feature for command model
    for cmd_name in same_cmd:
        mfcc_feat, _, _ = mfcc_from_file(same_cmd[cmd_name][0], sample_rate=sample_rate)
        speech_segs = vad_audio_segments(same_cmd[cmd_name][0], sample_rate=sample_rate)
        assert speech_segs

        seg_st = int(speech_segs[0][0] / 0.01)
        seg_end = int(speech_segs[-1][1] / 0.01)
        mfcc_feat = mfcc_feat[:, seg_st:seg_end]
        # add acoustic feature
        same_cmd[cmd_name].append(mfcc_feat)
        # add segment time
        same_cmd[cmd_name].append(seg_end - seg_st)

    # recognize cmd
    total_files = 0

    for wav_file in os.listdir(test_dir):
        if wav_file.find(".wav") == -1:
            continue
        wav_file = os.path.join(test_dir, wav_file)
        recog_cmd = recognize_cmd(same_cmd, same_model, diff_model, wav_file)
        print("file: {} --> result: {}".format(wav_file, recog_cmd))
        total_files += 1


def proc_vad(train_dir):
    for cmd_name in os.listdir(train_dir):
        cmd_dir = os.path.join(train_dir, cmd_name)
        if not os.path.isdir(cmd_dir):
            continue
        print("------------proc cmd: {}-----------".format(cmd_name))
        for wav_file in os.listdir(cmd_dir):
            if wav_file.find(".wav") == -1:
                continue
            speech_seg = vad_audio_segments(os.path.join(cmd_dir, wav_file), sample_rate=8000)
            print("{} segment: {}".format(wav_file, speech_seg))


def make_test_audio_file(train_dir):
    test_files = []
    for cmd_name in os.listdir(train_dir):
        cmd_dir = os.path.join(train_dir, cmd_name)
        if not os.path.isdir(cmd_dir):
            continue

        wav_files = os.listdir(cmd_dir)
        np.random.shuffle(wav_files)
        wav_files = wav_files[:3]
        for wav_file in wav_files:
            test_files.append([cmd_name, os.path.join(cmd_dir, wav_file)])

    np.random.shuffle(test_files)
    file_list = [file_item[1] for file_item in test_files]
    cmd_list = [file_item[0] for file_item in test_files]
    cmd_list = [f"{idx}th: {cmd_name}" for idx, cmd_name in enumerate(cmd_list)]
    with open("cmd_list.txt", "w") as w_f:
        w_f.write("\n".join(cmd_list))

    audio_merge_with_sox("cmd_list.wav", file_list)


if __name__ == "__main__":
    train_dir = "data/train"
    train_dir = "data/debug"
    #make_test_audio_file(train_dir)
    test_dir = "./data/test"
    #test_dir = "./data/debug/encender_luz_cocina"
    recognize_test(train_dir, test_dir)
    #recognize_command(train_dir)
    #proc_vad(train_dir)
