import RPi.GPIO as GPIO

import pyaudio

import random
import os
import shutil
import torch
import json
import time
import array
import subprocess

import numpy

from voice_util import detect_audio_segment, read_voice_file, total_energy_list, convert2wav, trim_audio_with_sox
from train_model import mfcc_from_file, similarity_mfcc, mfcc_from_audio
from gmm_model import load_gmm_from_npz

import webrtcvad
import librosa


from threading import Thread
from time import sleep
cmdEngine = None


GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
def angulo(ang):
    return float(pos)/10.+5.
def angulo2(ang2):
    return float(pos2)/10.+5.


Cortina_cuarto = 12
GPIO.setup(Cortina_cuarto, GPIO.OUT)
pwm = GPIO.PWM(Cortina_cuarto,100)

depart = 0
arrive = 65
delay = 0.1
incStep = 5
pos = depart

Cortina_sala = 13
GPIO.setup(Cortina_sala, GPIO.OUT)
pwm2 = GPIO.PWM(Cortina_sala,100)

depart2 = 0
arrive2 = 65
pos2 = depart



listening = True

luzcuarto = 2
luzsala = 3
luzcocina = 17
luzbano = 4

puerta_principal = 23
puerta_cuarto = 24

ventilador = 27
cocineta = 22

GPIO.setup(luzsala, GPIO.OUT)
GPIO.setup(luzcuarto, GPIO.OUT)
GPIO.setup(luzcocina, GPIO.OUT)
GPIO.setup(luzbano, GPIO.OUT)

GPIO.setup(puerta_principal, GPIO.OUT)
GPIO.setup(puerta_cuarto, GPIO.OUT)

GPIO.setup(ventilador, GPIO.OUT)
GPIO.setup(cocineta, GPIO.OUT)


def encender_luz_cuarto():
    GPIO.output(luzcuarto, GPIO.HIGH)
    print("Encendido luz cuarto")


def apagar_luz_cuarto():
    GPIO.output(luzcuarto, GPIO.LOW)
    print("Apagando luz de cuarto")


def encender_luz_sala():
    GPIO.output(luzsala, GPIO.HIGH)
    print("Encendido luz sala")


def apagar_luz_sala():
    GPIO.output(luzsala, GPIO.LOW)
    print("Apagando luz de sala")


def encender_luz_cocina():
    GPIO.output(luzcocina, GPIO.HIGH)
    print("Encendiendo luz de cocina")


def apagar_luz_cocina():
    GPIO.output(luzcocina, GPIO.LOW)
    print("Apagando luz de cocina")


def encender_luz_bano():
    GPIO.output(luzbano, GPIO.HIGH)
    print("Encendiendo luz de baño")


def apagar_luz_bano():
    GPIO.output(luzbano, GPIO.LOW)
    print("Apagando luz de baño")


def abrir_puerta_principal():
    GPIO.output(puerta_principal, GPIO.HIGH)
    print("Puerta principal abierta")


def cerrar_puerta_principal():
    GPIO.output(puerta_principal, GPIO.LOW)
    print("Puerta principal cerrada")


def abrir_puerta_cuarto():
    GPIO.output(puerta_cuarto, GPIO.HIGH)
    print("Puerta del cuarto abierta")


def cerrar_puerta_cuarto():
    GPIO.output(puerta_cuarto, GPIO.LOW)
    print("Puerta del cuarto cerrada")


def encender_ventilador():
    GPIO.output(ventilador, GPIO.HIGH)
    print("Ventilador encendido")


def apagar_ventilador():
    GPIO.output(ventilador, GPIO.LOW)
    print("Ventilador apagado")


def encender_cocineta():
    GPIO.output(cocineta, GPIO.HIGH)
    print("Cocineta encendida")


def apagar_cocineta():
    GPIO.output(cocineta, GPIO.LOW)
    print("Cocineta apagada")


def vad_audio_segments(audio_file, sample_rate=16000):
    #sample_rate = 16000
    frame_win = sample_rate*0.05
    frame_step = sample_rate * 0.05
    tr = 0.0001

    wav_data = read_voice_file(audio_file, sample_rate=sample_rate)
    engery_list = total_energy_list(wav_data, sample_rate, frame_win, frame_step)
    audio_segs, sil_segs = detect_audio_segment(engery_list, 0.05, tr)
    return audio_segs


def find_command(cmd_data, seg_st, seg_end, sample_rate):
    voz, cmdEngine.max_time = cmdEngine.find_cmd(cmd_data, seg_st, seg_end, sample_rate)
    if voz:
        print("cmd : {}".format(voz))
        if voz == "encender_luz_cuarto":
            encender_luz_cuarto()

        if voz == "encender_luz_sala":
            encender_luz_sala()

        if voz == "encender_luz_cocina":
            encender_luz_cocina()

        if voz == "encender_luz_baño":
            encender_luz_bano()

        if voz == "encender_ventilador":
            encender_ventilador()

        if voz == "apagar_luz_cuarto":
            apagar_luz_cuarto()

        if voz == "apagar_luz_sala":
            apagar_luz_sala()

        if voz == "apagar_luz_cocina":
            apagar_luz_cocina()

        if voz == "apagar_luz_baño":
            apagar_luz_bano()

        if voz == "apagar_ventilador":
            apagar_ventilador()

        if voz == "abrir_cortina_cuarto":
            pwm.start(angulo(arrive))
            for pos in range(arrive, depart, -incStep):
                duty = angulo(pos)
                pwm.ChangeDutyCycle(duty)
                print(duty)
                time.sleep(delay)

        if voz == "cerrar_cortina_cuarto":
            pwm.start(angulo(depart))
            for pos in range(depart, arrive, incStep):
                duty = angulo(pos)
                pwm.ChangeDutyCycle(duty)
                print(duty)
                time.sleep(delay)

        if voz == "abrir_cortina_sala":
            pwm2.start(angulo(arrive2))
            for pos2 in range(arrive2, depart2, -incStep):
                duty = angulo2(pos2)
                pwm2.ChangeDutyCycle(duty)
                print(duty)
                time.sleep(delay)

        if voz == "cerrar_cortina_sala":
            pwm2.start(angulo(depart2))
            for pos2 in range(depart2, arrive2, incStep):
                duty = angulo2(pos2)
                pwm2.ChangeDutyCycle(duty)
                print(duty)
                time.sleep(delay)

        if voz == "encender_cocineta":
            encender_cocineta()

        if voz == "apagar_cocineta":
            apagar_cocineta()

        if voz == "abrir_puerta_principal":
            abrir_puerta_principal()

        if voz == "cerrar_puerta_principal":
            cerrar_puerta_principal()

        if voz == "abrir_puerta_cuarto":
            abrir_puerta_cuarto()

        if voz == "cerrar_puerta_cuarto":
            cerrar_puerta_cuarto()


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


class StreamingVAD:
    def __init__(self, sample_rate, seg_sil_time):
        self.proc_time = 0
        self.seg_sil_time = seg_sil_time
        self.sample_rate = sample_rate
        self.vad_data = []
        self.temp_data = []
        self.energy_list = []
        self.vad_segment = []
        self.vad = webrtcvad.Vad(3)

    def get_floatdata(self, frame):
        short_array = array.array('h', frame)
        float_array = []

        for sample in short_array:
            if sample < 0:
                float_array.append(float(sample / 32768.0))
            else:
                float_array.append(float(sample / 32767.0))

        return float_array


    def update_vad(self, float_data):
        # update audio_data
        frame_win = 0.05
        vad_tr = 0.0001
        self.temp_data += float_data
        vad_samples = int(self.sample_rate*frame_win)

        # check vad frame time
        if len(self.temp_data) > vad_samples*5:
            frames = len(self.temp_data) // vad_samples
            self.vad_data += self.temp_data[:int(frames*vad_samples)]
            self.energy_list += total_energy_list(self.temp_data[:int(frames*vad_samples)], self.sample_rate,
                                                  vad_samples, vad_samples).tolist()
            self.temp_data = self.temp_data[int(frames*vad_samples):]
            self.proc_time += frames * frame_win
            audio_segs, sil_segs = detect_audio_segment(numpy.array(self.energy_list), frame_win, vad_tr)

            if audio_segs and audio_segs[-1][1] + self.seg_sil_time < self.proc_time:
                self.vad_segment = [audio_segs[0][0], audio_segs[-1][1]]
                #print("vad segment: [{}, {}]".format(self.vad_segment[0], self.vad_segment[1]))

    def proc_frame(self, frame):
        float_data = self.get_floatdata(frame)
        self.update_vad(float_data)
        return self.vad_segment

    def reset_vad_seg(self):
        self.vad_segment = []
        self.vad_data = []
        self.energy_list = []
        self.proc_time = 0

    def get_command_data(self):
        cmd_data = []
        if self.vad_segment:
            seg_st_idx = int(self.sample_rate * self.vad_segment[0])
            seg_end_idx = int(self.sample_rate * self.vad_segment[1])
            cmd_data = self.vad_data[seg_st_idx:seg_end_idx]

        return cmd_data

    def finalize_vad(self):
        # update audio_data
        frame_win = 0.05
        vad_tr = 0.0005
        vad_samples = int(self.sample_rate*frame_win)

        frames = len(self.temp_data) // vad_samples
        self.vad_data += self.temp_data[:int(frames*vad_samples)]
        self.energy_list += total_energy_list(self.temp_data[:int(frames*vad_samples)], self.sample_rate,
                                              vad_samples, vad_samples).tolist()
        self.temp_data = self.temp_data[int(frames*vad_samples):]
        audio_segs, sil_segs = detect_audio_segment(numpy.array(self.energy_list), frame_win, vad_tr)
        if audio_segs:
            self.vad_segment = [self.proc_time + audio_segs[0][0], self.proc_time + audio_segs[-1][1]]
            print("vad segment: [{}, {}]".format(self.vad_segment[0], self.vad_segment[1]))
            self.proc_time += audio_segs[-1][1]


class VoiceCmdEngine():
    def __init__(self, voice_cmds, same_model, diff_model, stream_vad):
        self.stream_vad = stream_vad
        self.cur_time = 0
        self.max_time = 0
        self.cmd_num = 0
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

                if diff_time < 0.3:
                    seg_st_idx = int(seg_st/0.01)
                    seg_end_idx = int(seg_end/0.01)
                    seg_feat, _, _ = mfcc_from_audio(speech_data, sample_rate)
                    #seg_feat = seg_feat[:, seg_st_idx:seg_end_idx]
                    similarity = similarity_mfcc(cmd_feat, seg_feat)
                    similarity = numpy.array(similarity)
                    cmd_score1 = self.same_model.score(similarity) - self.diff_model.score(similarity)
                    cmd_score2 = -1.000

                    if cmd_score1 > 0.1:
                        similarity = similarity_mfcc(seg_feat, cmd_feat)
                        similarity = numpy.array(similarity)
                        cmd_score2 = self.same_model.score(similarity) - self.diff_model.score(similarity)

                    if cmd_score2 > 0.1:
                    #if cmd_score1 > 0.1:
                        recog_cmd = cmd_name
                        return recog_cmd, max_cmd_time
                        #recog_cmd = cmd_name
                        #break

        return recog_cmd, max_cmd_time

    def get_command(self, frame):
        # vad segment
        vad_segment = self.stream_vad.proc_frame(frame)
        sample_rate = self.stream_vad.sample_rate

        # check vad segment
        if vad_segment:
            # get command signal data
            cmd_data = self.stream_vad.get_command_data()
            seg_st, seg_end = vad_segment
            self.stream_vad.reset_vad_seg()
            # detect segment
            print("detected seg: [{}, {}]".format(seg_st, seg_end))
            # find command
            #cmd_name, self.max_time = self.find_cmd(cmd_data, seg_st, seg_end, sample_rate)
            #if cmd_name:
            #    print("cmd : {}".format(cmd_name))
            cmdEngineThread = Thread(target=find_command, args=(cmd_data, seg_st, seg_end, sample_rate,))
            cmdEngineThread.start()
        elif self.stream_vad.proc_time > 5:
            self.stream_vad.reset_vad_seg()

        stt_res = {"partial" : ""}

        return json.dumps(stt_res)


def detect_cmd(voice_cmds, same_model, diff_model):
    sample_rate = 8000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = sample_rate
    CHUNK = int(SAMPLE_RATE / 10)

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    # declare streaming vad
    sil_seg_time = 0.7
    global cmdEngine
    stream_vad = StreamingVAD(sample_rate, sil_seg_time)
    cmdEngine = VoiceCmdEngine(voice_cmds, same_model, diff_model, stream_vad)

    while True:
        data = stream.read(CHUNK)
        if len(data) == 0:
            break

        cmdEngine.get_command(data)


if __name__ == "__main__":
    # load classification model
    same_model = load_gmm_from_npz("models/same_model.npz")
    diff_model = load_gmm_from_npz("models/diff_model.npz")
    cur_time = time.time()

    # load command directory
    cmd_dir = "data/register"
    #cmd_dir = "./data/debug"
    voice_cmds = voice_cmd_files(cmd_dir)
    print("load time: {}".format(time.time() - cur_time))

    test_dir = "./data/test"
    test_dir = "./data/erick_mct-attachments"
    #test_dir = "./data/train/abrir_cortina_cuarto"
    detect_cmd(voice_cmds, same_model, diff_model)
