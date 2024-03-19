
import os
import json
import subprocess
import numpy as np
import array
import time
from gmm_model import load_gmm_from_npz
from prepare_data import vad_stream_segments, vad_audio_segments
from train_model import mfcc_from_file, similarity_mfcc, mfcc_from_audio


def get_floatdata(frame):
    short_array = array.array('h', frame)
    float_array = []

    for sample in short_array:
        if sample < 0:
            float_array.append(float(sample/32768.0))
        else:
            float_array.append(float(sample/32767.0))

    return float_array


class VoiceCmdEngine():
    def __init__(self, voice_cmds, same_model, diff_model):
        self.vad_data = []
        self.silence_time = 0
        self.speech_time = 0
        self.cur_time = 0
        self.max_time = 0
        self.cmd_num = 0
        self.vad_segment = []
        self.proc_seg = []
        self.voice_cmds = voice_cmds
        self.same_model = same_model
        self.diff_model = diff_model

    def find_cmd(self, speech_data, seg_st, seg_end, sample_rate):
        recog_cmd = ""
        #sample_rate = 8000
        max_cmd_time = 0
        for cmd_name in self.voice_cmds:
            for idx in range(len(self.voice_cmds[cmd_name])):
                cmd_feat = self.voice_cmds[cmd_name][idx][1]
                cmd_len = self.voice_cmds[cmd_name][idx][2]
                diff_time = abs(cmd_len - (seg_end-seg_st))
                if max_cmd_time < cmd_len:
                    max_cmd_time = cmd_len

                if diff_time < 0.2:
                    seg_st_idx = int(seg_st/0.01)
                    seg_end_idx = int(seg_end/0.01)
                    seg_feat, _, _ = mfcc_from_audio(speech_data, sample_rate)
                    seg_feat = seg_feat[:, seg_st_idx:seg_end_idx]
                    similarity = similarity_mfcc(cmd_feat, seg_feat)
                    similarity = np.array(similarity)
                    cmd_score1 = self.same_model.score(similarity) - self.diff_model.score(similarity)
                    cmd_score2 = 0

                    if cmd_score1 > 0.3:
                        similarity = similarity_mfcc(seg_feat, cmd_feat)
                        similarity = np.array(similarity)
                        cmd_score2 = self.same_model.score(similarity) - self.diff_model.score(similarity)

                    if cmd_score1 > 0.3 and cmd_score2 > 0.3:
                    #if cmd_score1 > 0.1:
                        recog_cmd = cmd_name
                        return recog_cmd, max_cmd_time
                        #recog_cmd = cmd_name
                        #break

        return recog_cmd, max_cmd_time

    def get_command(self, frame):
        # get float data
        float_data = get_floatdata(frame)
        sample_rate = 8000

        # update VAD information
        self.update_vad(float_data)

        if self.vad_segment:
            seg_st = self.vad_segment[0][0]
            seg_end = self.vad_segment[-1][1]

            # print transcribe data
            seg_st = seg_st
            seg_end = seg_end
            # find cmd
            if [seg_st, seg_end] not in self.proc_seg:
                vad_time = time.time()
                cmd_name, self.max_time = self.find_cmd(self.vad_data, seg_st, seg_end, sample_rate)
                print("cmd time: {}".format(time.time() - vad_time))
                if cmd_name:
                    print(f"{self.cmd_num}th {cmd_name}")
                    self.cmd_num += 1
                    seg_end_idx = int(seg_end*sample_rate)
                    self.vad_data = self.vad_data[seg_end_idx:]
                    self.cur_time = len(self.vad_data)*1.0/sample_rate
                    self.max_time = 0
                    self.vad_segment = []

            # update proc time
            self.proc_seg = [[seg_st, seg_end]]

            cur_vad_time = self.cur_time - seg_st
            # check maximum data
            if cur_vad_time > self.max_time + 0.3 and self.vad_segment:
                first_seg_end = self.vad_segment[0][1]
                self.vad_segment = self.vad_segment[1:]
                seg_end_idx = int(first_seg_end * sample_rate)
                self.vad_data = self.vad_data[seg_end_idx:]
                self.cur_time = len(self.vad_data) * 1.0 / sample_rate

            #if seg_end + 0.15 < self.cur_time:
            #    print("seg time: [{}(s), {}(s)]".format(seg_st, seg_end))
        elif self.cur_time > 1.5:
            self.cur_time = 0
            self.vad_data = []
            self.max_time = 0

        stt_res = {"partial" : ""}

        return json.dumps(stt_res)

    def update_vad(self, float_data):
        # update audio_data
        self.vad_data += float_data
        sample_rate = 8000.0
        assert len(float_data) < sample_rate
        # get frame time length
        frame_time = len(float_data) / sample_rate
        self.cur_time += frame_time
        #print("cur time: {}".format(self.cur_time))

        if self.cur_time >= 1.5:
            self.vad_segment = vad_stream_segments(self.vad_data, sample_rate)


def detect_cmd(wav_file, voice_cmds, same_model, diff_model):
    sample_rate = 8000
    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                wav_file,
                                '-ar', str(sample_rate), '-ac', '1', '-f', 's16le', '-'],
                               stdout=subprocess.PIPE)

    #whisper = WhisperEngine("en", model_path)
    cmdEngine = VoiceCmdEngine(voice_cmds, same_model, diff_model)

    while True:
        data = process.stdout.read(400)
        if len(data) == 0:
            break

        cmdEngine.get_command(data)


def voice_cmd_files(train_dir):
    sample_rate = 8000
    same_cmd = {}

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
                same_cmd[cmd_name] = [[wav_path]]
                #break
            else:
                same_cmd[cmd_name].append([wav_path])

            file_idx += 1
            if file_idx == 3:
                break

    # calculate feature for command model
    for cmd_name in same_cmd:
        for idx in range(len(same_cmd[cmd_name])):
            mfcc_feat, _, _ = mfcc_from_file(same_cmd[cmd_name][idx][0], sample_rate=sample_rate)
            speech_segs = vad_audio_segments(same_cmd[cmd_name][idx][0], sample_rate=sample_rate)
            assert speech_segs

            seg_st = int(speech_segs[0][0] / 0.01)
            seg_end = int(speech_segs[-1][1] / 0.01)
            mfcc_feat = mfcc_feat[:, seg_st:seg_end]
            # add acoustic feature
            same_cmd[cmd_name][idx].append(mfcc_feat)
            # add segment time
            same_cmd[cmd_name][idx].append(speech_segs[-1][1] - speech_segs[0][0])
    return same_cmd


if __name__ == "__main__":
    train_dir = "data/train"
    train_dir = "data/register"
    same_model = load_gmm_from_npz("models/same_model.npz")
    diff_model = load_gmm_from_npz("models/diff_model.npz")
    cur_time = time.time()
    #train_dir = "./data/debug/"
    voice_cmds = voice_cmd_files(train_dir)
    print("load model time: {}".format(time.time() - cur_time))
    cur_time = time.time()
    wav_file = "cmd_list.wav"
    test_dir = "./data/train/abrir_puerta_principal"
    test_dir = "./data/test"
    test_dir = "./data/debug"
    file_idx = 0
    for wav_file in os.listdir(test_dir):
        if wav_file.find(".wav") == -1:
            continue
        file_idx += 1
        #if file_idx == 1:
        #    continue
        print(f"proc {wav_file}")
        cur_time = time.time()
        detect_cmd(os.path.join(test_dir, wav_file), voice_cmds, same_model, diff_model)
        print("detection time: {}".format(time.time() - cur_time))

    #wav_file = "./data/test/test2.wav"
    #detect_cmd(wav_file, voice_cmds, same_model, diff_model)
    #print("detection time: {}".format(time.time() - cur_time))
