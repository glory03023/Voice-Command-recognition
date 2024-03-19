import os
import numpy as np
import librosa
#import audioPlayer as player
import matplotlib.pyplot as plt
import tempfile
import queue
import sys
import array
import struct
#import sounddevice as sd
import wave
import pandas as pd
import scipy

SOX_BIN = os.path.join('sox', 'sox')
FFMPEG_BIN = 'ffmpeg.exe'
rec_status = False
rec_filename = ''


def get_platform():
    if 'win32' in sys.platform:
        return 'win32'
    return 'linux'


def convert2wav(audio_file, wav_file):
    if get_platform() == 'win32':
        commands = '{} -i \"{}\" -acodec pcm_s16le -ac 1 -ar 16000 \"{}\" -y -loglevel panic'.format(
            FFMPEG_BIN, audio_file, wav_file)
    else:
        commands = 'ffmpeg -i \"{}\" -acodec pcm_s16le -ac 1 -ar 16000 \"{}\" -y -loglevel panic'.format(
            audio_file, wav_file)
    # commands = '{} {} {}'.format(SOX_BIN, audio_file, wav_file)
    os.system(commands)


def trim_audio_ffmpeg(src_file, start_tm, end_tm, dst_file):
    if not os.path.exists(src_file):
        return
    if os.path.exists(dst_file):
        os.remove(dst_file)
    if get_platform() == 'win32':
        commands = '{}  -i \"{}\" -ss {:.2f} -to {:.2f} \"{}\" -y -loglevel panic'.format(
            FFMPEG_BIN, src_file, start_tm, end_tm, dst_file)
    else:
        commands = 'ffmpeg  -i \"{}\" -ss {:.2f} -to {:.2f} \"{}\" -y -loglevel panic'.format(
            src_file, start_tm, end_tm, dst_file)
    os.system(commands)
    return dst_file


def play_wave(wave_file='', status=True):
    if not os.path.exists(wave_file):
        return False
    ext = os.path.basename(wave_file)[-4:]
    if ext != '.wav':
        convert2wav(wave_file, 'tmp.wav')
        player.audio_play('tmp.wav', status)
    else:
        player.audio_play(wave_file, status)
    return True


def read_voice_file(voice_file, sample_rate=16000):
    if not os.path.exists(voice_file):
        return []
    y, sr = librosa.load(voice_file, sr=sample_rate)

    #fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    #ax.plot(y)
    #fig.savefig('to.jpg')  # save the figure to file
    #plt.close(fig)  # close the figure
    return np.asarray(y, dtype=np.float)


def record_from_mic():
    global rec_status
    global rec_filename
    rec_status = True
    rec_filename = tempfile.mktemp(prefix='rec_unlimited_', suffix='.wav', dir='')
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # Make sure the file is opened before recording anything:
    rec_buffer = []
    with sd.InputStream(samplerate=16000, device=0,
                        channels=1, callback=callback):
        print('#' * 80)
        print('press Ctrl+C to stop the recording')
        print('#' * 80)
        while rec_status:
            rec_buffer.extend(q.get())

        print("* done recording")

        raw_floats = [x for x in rec_buffer]
        floats = array.array('f', raw_floats)
        samples = [int(sample * 32767)
                   for sample in floats]
        raw_ints = struct.pack("<%dh" % len(samples), *samples)

        wf = wave.open(rec_filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(raw_ints)
        wf.close()


def write_wav(wav_file, wav_dta, sample_rate):
    samples = [int(sample * 32767)
               for sample in wav_dta]
    # samples = wav_dta
    samples = np.asarray(samples, dtype=np.short)
    raw_ints = struct.pack("<%dh" % len(samples), *samples)

    wf = wave.open(wav_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(raw_ints)
    wf.close()


def record_stop():
    global rec_status
    rec_status = False


def feat_extract(signal,
                 sample_rate,
                 win,
                 stride,
                 ceps_order,
                 n_mels,
                 delta,
                 delta2,
                 normalize):
    """
    This function calculates mfcc array with 1-d array signal data.
    :param list signal:     float data array containing pcm data.
    :param int sample_rate: sampling frequency such as 16000, 8000
    :param float win:       window size of frame for extracting spectrum feature, default value is 0.25
    :param float stride:    step size of frame for extracting spectrum feature, default value is 0.1
    :param int ceps_order:  keeping cepstrum size of total spectrum feature, default value is 13
    :param int n_mels:      total cepstrum size, default value is 40
    :param bool delta:      if it is True, calculate delta of feature, else not.
    :param bool delta2:     if it is True, calculate double delta of feature, else not.
    :param bool normalize   if it is True, normalize feature, else not.
    :return:                (frame energy, total cepstrum feature, mfcc)
    """
    # Normalization of pcm data

    signal = np.double(signal)
    # signal = signal / (2.0 ** 15)
    assert isinstance(signal, np.ndarray)
    sig_mean = signal.mean()
    sig_std = signal.std()
    # sig_max = (numpy.abs(signal)).max()
    signal = (signal - sig_mean) / (sig_std + 0.0000000001)
    # signal = (signal - sig_mean)
    # signal = (signal - sig_mean) / (sig_max + 0.0000000001)
    frame_length = int(win * sample_rate)
    frame_step = int(stride * sample_rate)
    energy = librosa.feature.rmse(signal, frame_length=frame_length, hop_length=frame_step)
    mel_spectrogram = librosa.feature.melspectrogram(
        signal, sample_rate, n_fft=frame_length, hop_length=frame_step, n_mels=n_mels, power=1.0)
    mel_filter_bank = librosa.power_to_db(mel_spectrogram)
    mfcc_ = librosa.feature.mfcc(S=mel_filter_bank, n_mfcc=ceps_order, dct_type=2)
    mfcc_ = mfcc_[1:ceps_order + 1]
    if normalize:
        mean = np.mean(mfcc_, axis=1)
        std = np.std(mfcc_, axis=1)
        mfcc_ = (mfcc_.T-mean)/(std + 0.000000001)
        mfcc_ = mfcc_.T

    # mel_filter_bank = mel_filter_bank / 10
    mel_filter_bank = mel_filter_bank.T
    if delta:
        mfcc_delta = librosa.feature.delta(mfcc_)
        mfcc_ = np.concatenate((mfcc_, mfcc_delta), axis=0)
    if delta2:
        mfcc_2delta = librosa.feature.delta(mfcc_, order=2)
        mfcc_ = np.concatenate((mfcc_, mfcc_2delta), axis=0)
    mfcc_ = mfcc_.T
    return energy, mel_filter_bank, mfcc_


def audio_bic_feat(wav_file):
    """
    This function extract mfcc from audio file.
    :param str wav_file:    audio file path
    :return:
    """
    audio_data = librosa.load(wav_file, sr=16000)
    wav_data = audio_data[0]
    sample_rate = audio_data[1]
    return feat_extract(wav_data, sample_rate,
                        win=0.025,
                        stride=0.01,
                        n_mels=40,
                        ceps_order=13,
                        delta=False,
                        delta2=False,
                        normalize=False)


def trim_audio_with_sox(path, sample_rate, start_time, end_time, out_file):
    """
    crop and resample the recording with sox and loads it.
    """
    # tmp_file = "tmp.wav"
    if os.path.exists(out_file):
        os.remove(out_file)

    if get_platform() == 'win32':
        sox_params = "{} \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={}".format(
            SOX_BIN, path, sample_rate, out_file, start_time, end_time)
    else:
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={}".format(
            path, sample_rate, out_file, start_time, end_time)

    os.system(sox_params)
    return out_file


def audio_merge_with_sox(path, audio_list):
    """
    crop and resample the recording with sox and loads it.
    """
    if os.path.exists(path):
        os.remove(path)

    audio_files = " ".join(audio_list)

    if get_platform() == 'win32':
        sox_params = "{} {} \"{}\"".format(SOX_BIN, audio_files, path)
    else:
        sox_params = "sox {} \"{}\"".format(audio_files, path)
    os.system(sox_params)
    return path


def div_gauss(cep, show='empty', win=250, shift=0):
    """
    Segmentation based on gaussian divergence.
    The segmentation detects the instantaneous change points corresponding to
    segment boundaries. The proposed algorithm is based on the detection of
    local maxima. It detects the change points through a gaussian divergence
    (see equation below), computed using Gaussians with diagonal covariance
    matrices. The left and right gaussians are estimated over a five-second
    window sliding along the whole signal (2.5 seconds for each gaussian,
    given *win* =250 features).
    A change point, i.e. a segment boundary, is present in the middle of the
    window when the gaussian divergence score reaches a local maximum.
        :math:`GD(s_l,s_r)=(\\mu_r-\\mu_l)^t\\Sigma_l^{-1/2}\\Sigma_r^{-1/2}(\\mu_r-\\mu_l)`
    where :math:`s_l` is the left segment modeled by the mean :math:`\mu_l` and
    the diagonal covariance matrix :math:`\\Sigma_l`, :math:`s_r` is the right
    segment modeled by the mean :math:`\mu_r` and the diagonal covariance
    matrix :math:`\\Sigma_r`.
    :param cep: numpy array of frames
    :param show: speaker of the show
    :param win: windows size in number of frames
    :return: a diarization object (s4d annotation)
    """

    length = cep.shape[0]
    # start and stop of the rolling windows A
    start_a = win - 1  # end of NAN
    stop_a = length - win
    # start and stop of the rolling windows B
    start_b = win + win - 1  # end of nan + delay
    stop_b = length

    # put features in a Pandas DataFrame
    df = pd.DataFrame(cep)
    # compute rolling mean and std in the window of size win, get numpy array
    # mean and std have NAN at the beginning and the end of the output array
    # mean = pd.rolling_mean(df, win).values
    # std = pd.rolling_std(df, win).values
    r = df.rolling(window=win, center=False)
    mean = r.mean().values
    std = r.std().values

    # compute GD scores using 2 windows A and B
    dist = (np.square(mean[start_a:stop_a, :] - mean[start_b:stop_b, :]) / (
        std[start_a:stop_a, :] * std[start_b:stop_b, :])).sum(axis=1)

    # replace missing value to match cep size
    dist_pad = np.lib.pad(dist, (win - 1, win), 'constant',
                          constant_values=(dist[0], dist[-1]))

    # remove non-speech frame
    # find local maximal at + or - win size
    borders = scipy.signal.argrelmax(dist_pad, order=win)[0].tolist()
    # append the first and last
    borders = [0] + borders + [length]
    return borders


def mean_energy(frame):
    return np.sum(frame ** 2) / np.float64(len(frame))


def total_energy_list(signal, sample_rate, frame_win, frame_step):
    frame_win = int(frame_win)
    frame_step = int(frame_step)

    # Signal normalization
    signal = np.double(signal)
    _ = sample_rate
    sample_num = len(signal)  # total number of samples
    cur_pos = 0

    energy_list = []
    while cur_pos < sample_num - frame_win + 1:
        frame = signal[cur_pos : cur_pos + frame_win]
        cur_pos = cur_pos + frame_step
        energy_list.append(mean_energy(frame))

    return np.array(energy_list)


def get_audio_and_silence_segment(audio_labels, frame_step):
    silence_flag = False
    audio_segments = []
    silence_segments = []
    current_audio_label = -1
    st_idx = -1

    for idx, label in enumerate(audio_labels):
        if label != current_audio_label:
            if st_idx == -1:  # start segment
                st_idx = idx
                current_audio_label = label
            else:
                # Set end index of current segment.
                end_idx = idx
                # Append current segment into speaker segments.
                if current_audio_label == silence_flag:
                    silence_segments.append([st_idx * frame_step, end_idx * frame_step])
                else:
                    audio_segments.append([st_idx * frame_step, end_idx * frame_step])
                # Set start index and label id of new audio segment
                st_idx = idx
                current_audio_label = label
        elif idx >= len(audio_labels) - 1:  # Check end silence value.
            end_idx = len(audio_labels)
            if end_idx - st_idx > 1:
                # Append current segment into speaker segments.
                if current_audio_label == silence_flag:
                    silence_segments.append([st_idx * frame_step, end_idx * frame_step])
                else:
                    audio_segments.append([st_idx * frame_step, end_idx * frame_step])
            break

    return audio_segments, silence_segments


def detect_audio_segment(energy_list, frame_step, tr):
    from scipy.signal import medfilt
    sil_threshold = tr
    # sil_threshold = 0.003

    #  Filter speech interval with high energy frames
    speech_non_speech = energy_list >= sil_threshold
    speech_non_speech = medfilt(speech_non_speech, 3)
    #speech_non_speech = medfilt(speech_non_speech, 5)

    audio_segments, silence_segments = get_audio_and_silence_segment(speech_non_speech, frame_step=frame_step)

    min_duration = 0.05
    audio_segments2 = []
    for s in audio_segments:
        if s[1] - s[0] > min_duration:
            # Append speech segment
            audio_segments2.append(s)
    audio_segments = audio_segments2
    return audio_segments, silence_segments


def detect_audio_segment_with_medi_tr(energy_list, frame_step, tr, medi_tr):
    from scipy.signal import medfilt
    sil_threshold = tr
    # sil_threshold = 0.003

    #  Filter speech interval with high energy frames
    speech_non_speech = energy_list >= sil_threshold
    #speech_non_speech = medfilt(speech_non_speech, 5)

    is_speech = False
    engerg_len = len(speech_non_speech)
    for i in range(engerg_len):
       if speech_non_speech[i]:
           is_speech = True
           continue
       elif is_speech == False:
           continue

       if is_speech and energy_list[i] > medi_tr:
           speech_non_speech[i] = True
       else:
           is_speech = False

    for i in range(engerg_len-1, -1, -1):
       if speech_non_speech[i]:
           is_speech = True
           continue
       elif is_speech == False:
           continue

       if is_speech and energy_list[i] > medi_tr*1.5:
           speech_non_speech[i] = True
       else:
           is_speech = False

    audio_segments, silence_segments = get_audio_and_silence_segment(speech_non_speech, frame_step=frame_step)

    """
    min_duration = 0.15
    audio_segments2 = []
    for s in audio_segments:
        if s[1] - s[0] > min_duration:
            # Append speech segment
            audio_segments2.append(s)
    audio_segments = audio_segments2
    """

    if not audio_segments or not silence_segments:
        return audio_segments, silence_segments

    # audio merge
    merge_duration = 0.9
    audio_segments2 = []
    cur_st = audio_segments[0][0]
    cur_end = audio_segments[0][1]
    for i in range(1, len(audio_segments)):
        if audio_segments[i][0] - audio_segments[i-1][1] < merge_duration:
            cur_end = audio_segments[i][1]
        else:
            # add current segment and reset seg
            audio_segments2.append([cur_st, cur_end])
            # reset current segment information
            cur_st = audio_segments[i][0]
            cur_end = audio_segments[i][1]

        if i == len(audio_segments)-1:
            audio_segments2.append([cur_st, cur_end])

    if len(audio_segments) > 1:
        audio_segments = audio_segments2

    audio_segments2 = []
    cur_st = audio_segments[0][0]
    cur_end = audio_segments[0][1]
    for i in range(1, len(audio_segments)):
        if audio_segments[i][0] - audio_segments[i-1][1] < merge_duration:
            cur_end = audio_segments[i][1]
        else:
            # add current segment and reset seg
            audio_segments2.append([cur_st, cur_end])
            # reset current segment information
            cur_st = audio_segments[i][0]
            cur_end = audio_segments[i][1]

        if i == len(audio_segments)-1:
            audio_segments2.append([cur_st, cur_end])
    if len(audio_segments) > 1:
        audio_segments = audio_segments2

    min_duration = 0.25
    audio_segments2 = []
    for s in audio_segments:
        if s[1] - s[0] > min_duration:
            # Append speech segment
            audio_segments2.append(s)
    audio_segments = audio_segments2

    # make sil_segments
    silence_segments2 = []
    for idx, s in enumerate(audio_segments):
        if idx == 0:
            if s[0] > 0:
                silence_segments2.append([0, s[0]])
        else:
            silence_segments2.append([audio_segments[idx-1][1], s[0]])

        if idx == len(audio_segments) -1:
            if s[1] < len(energy_list)*frame_step:
                silence_segments2.append([s[1], len(energy_list)*frame_step])
    silence_segments = silence_segments2

    return audio_segments, silence_segments


def vad_audio_segment(audio_file, concate_gap=0.3):
    audio_data = librosa.load(audio_file, sr=16000)
    wav_data = audio_data[0]
    sample_rate = audio_data[1]
    vad_energy_list = total_energy_list(wav_data, sample_rate, int(sample_rate*0.05), int(sample_rate*0.05))
    vad_mean = np.mean(vad_energy_list)
    vad_min = np.min(vad_energy_list)
    vad_std = np.std(vad_energy_list)
    tr = vad_mean*vad_mean /(vad_std + vad_std + vad_std + 0.0000001)
    audio_segment = detect_audio_segment(vad_energy_list, 0.05, tr)
    refined_segment = []
    cur_sp_start = audio_segment[0][0][0]
    cur_sp_end = audio_segment[0][0][1]
    for idx in range(len(audio_segment[0])-1):
        seg_st, seg_end = audio_segment[0][idx+1]
        if cur_sp_end + concate_gap > seg_st:
            cur_sp_end = seg_end
        else:
            refined_segment.append([cur_sp_start, cur_sp_end])
            cur_sp_start = seg_st
            cur_sp_end = seg_end

    if [cur_sp_start, cur_sp_end] not in refined_segment:
        refined_segment.append([cur_sp_start, cur_sp_end])
    return refined_segment


def get_duration(wave_file):
    y, fs = librosa.load(wave_file, sr=16000)
    nFrames = len(y)
    audio_length = nFrames * (1 / fs)

    return audio_length
