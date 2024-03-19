import os
import threading
import sounddevice as sd
import scipy.io.wavfile as wav


class Player(object):

    def __init__(self):

        self.status = {'cont_playing': threading.Event(),  # check if playing
                       'finished': threading.Event(),
                       'playing': threading.Event(),  # stop or play
                       'time': 0}
        self.file = None
        self.frame_rate = -1
        self.buffer = None
        self.frames = -1
        self.position = -1
        self.duration = 0

    def load_file(self, file):
        self.file = file
        try:
            self.frame_rate, self.buffer = wav.read(file)
            self.duration = float(len(self.buffer) / self.frame_rate)
        except:
            raise()

    def toggle_pause(self):
        if self.status['cont_playing'].isSet():
            self.status['cont_playing'].clear()
        else:
            self.status['cont_playing'].set()

    def pretty_pos(self):
        pos = self.file.position
        min = int(pos) / 60
        sec = int(pos) - min * 60
        tin = "%02d:%02d" % (min, sec)

        pos = self.file.length
        min = int(pos) / 60
        sec = int(pos) - min * 60
        tot = "%02d:%02d" % (min, sec)

        return tin, tot

    def pause(self):
        self.status['cont_playing'].clear()

    def unpause(self):
        self.status['cont_playing'].set()

    def seek(self):
        pass

    def stop(self):
        self.status['playing'].clear()
        sd.stop()

    def play(self):
        sd.play(self.buffer, self.frame_rate)

    def play_segment(self, start_time, end_time):
        start_frame = int(start_time * self.frame_rate)
        end_frame = int(end_time * self.frame_rate)
        segment_buffer = self.buffer[start_frame:end_frame]
        sd.play(segment_buffer, self.frame_rate)


pl = Player()


def audio_play(audio_file, status):
    if status:
        pl.load_file(audio_file)
        pl.play()
    else:
        pl.stop()


def audio_segment_play(audio_file, start_time, end_time):
    if not os.path.exists(audio_file):
        return
    if end_time <= start_time:
        return
    pl.load_file(audio_file)
    pl.play_segment(start_time, end_time)



