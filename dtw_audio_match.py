import os
import numpy as np
from numpy.linalg import norm
import librosa
#from dtw import *
from voice_util import detect_audio_segment_with_medi_tr


def save_feat_data(feat_file, mfcc, engery, rms):
    np.savez(feat_file, mfcc=mfcc, engery=engery, rms=rms)


def load_feat(feat_file):
    load_data = np.load(feat_file)
    return load_data['mfcc'], load_data['energy'], load_data['rms']


def mfcc_from_file(audio_file, sample_rate):
    frame_len = 256
    frame_step = 128
    y, sr = librosa.load(audio_file, sr=None)
    assert sr==sample_rate
    y = normalize_sig(y)
    energy = librosa.feature.rms(y, frame_length=frame_len, hop_length=frame_step)
    S, phase = librosa.magphase(librosa.stft(y, win_length=frame_len, hop_length=frame_step))
    rms = librosa.feature.rms(S=S)
    mfcc = librosa.feature.mfcc(y, sr, n_fft=frame_len, hop_length=frame_step, n_mels=40, n_mfcc=19)
    mfcc = mfcc[1:, :]

    return mfcc, energy, rms


def normalize_sig(wav_data):
    signal = np.double(wav_data)
    signal = signal / (2.0 ** 15)
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

    dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
    # print ('Normalized distance between the two sounds: {}'.format(dist))
    return dist, path


def get_match_point_list(speech_seg1, speech_seg2):
    match_points = []
    for seg_st1, seg_end1 in speech_seg1:
        for seg_st2, seg_end2 in speech_seg2:
            match_points.append([[seg_st1, seg_st2], [seg_end1, seg_end2]])

    return match_points


# Returns index of x in arr if present, else -1
def binarySearch(arr, l, r, x):
    # Check base case
    if r >= l:
        mid = l + (r - l) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            return mid

            # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binarySearch(arr, l, mid - 1, x)

            # Else the element can only be present in right subarray
        else:
            return binarySearch(arr, mid + 1, r, x)

    else:
        # Element is not present in the array
        return -1

def get_seg_info(dtw_path):
    first_path = dtw_path[0]
    sec_path = dtw_path[1]
    if len(first_path) % 2 == 1:
        first_path = first_path[:-1]
        sec_path = sec_path[:-1]
    assert len(first_path) == len(sec_path)
    assert len(first_path) %2 == 0
    cur_idx = 0
    path_len = len(first_path)

    # Initialize segment information

    audio_seg_list = [[],[]]
    prev_st1 = first_path[cur_idx]
    prev_st2 = sec_path[cur_idx]
    prev_end1 = first_path[cur_idx + 1]
    prev_end2 = sec_path[cur_idx + 1]

    # check overlap segment
    prev_overlap = 0 # 0- no-overlap, 1 : first audio overlap, 2: second audio overlap
    while cur_idx < path_len:
        cur_overlap = 0  # 0- no-overlap, 1 : first audio overlap, 2: second audio overlap
        if first_path[cur_idx] == first_path[cur_idx+1]:
            cur_overlap = 1
        elif sec_path[cur_idx] == sec_path[cur_idx+1]:
            cur_overlap = 2

        # update next segment information
        if cur_overlap+prev_overlap == 0:
            audio_seg_list[0].append([first_path[cur_idx], first_path[cur_idx+1]])
            audio_seg_list[1].append([sec_path[cur_idx], sec_path[cur_idx+1]])
        elif cur_overlap != prev_overlap and prev_overlap != 0:
            audio_seg_list[0].append([prev_st1, prev_end1])
            audio_seg_list[1].append([prev_st2, prev_end2])

        # update previous position
        if cur_overlap != prev_overlap:
            prev_st1 = first_path[cur_idx]
            prev_st2 = sec_path[cur_idx]

        prev_end1 = first_path[cur_idx + 1]
        prev_end2 = sec_path[cur_idx + 1]

        # update prev overlap
        prev_overlap = cur_overlap

        # move current index into next
        cur_idx += 2
        if cur_idx == path_len and cur_overlap+prev_overlap != 0:
            audio_seg_list[0].append([prev_st1, prev_end1])
            audio_seg_list[1].append([prev_st2, prev_end2])

    return audio_seg_list


def get_audio_time_map_org(audio_file1, audio_file2):
    frame_step = 0.016
    # calculate mfcc
    mfcc1, energy1, rms1 = mfcc_from_file(audio_file1, sample_rate=8000)
    mfcc2, energy2, rms2 = mfcc_from_file(audio_file2, sample_rate=8000)
    wav1_len = len(rms1[0])*frame_step
    wav2_len = len(rms2[0])*frame_step

    # get dtw cost
    dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
    print ('Normalized distance between the two sounds: {}'.format(dist))

    # get vad segment
    speech_seg1, _ = detect_audio_segment_with_medi_tr(energy1[0], frame_step=frame_step, tr=0.09, medi_tr=0.007)
    speech_seg2, _ = detect_audio_segment_with_medi_tr(energy2[0], frame_step=frame_step, tr=0.09, medi_tr=0.007)

    # list of point on audio 1
    idx_list_on_audio1 = []
    # inverse search of dtw result because dtw-master results contain other's index array.
    for seg_st, seg_end in speech_seg1:
        seg_st_idx = int(np.floor(seg_st/frame_step))
        st_idx_on_audio1 = binarySearch(path[0], 0, len(path[0])-1, seg_st_idx)
        seg_end_idx = int(np.floor(seg_end/frame_step))
        end_idx_on_audio1 = binarySearch(path[0], 0, len(path[0])-1, seg_end_idx)
        idx_list_on_audio1 += [st_idx_on_audio1, end_idx_on_audio1]

    idx_list_on_audio2 = []
    for seg_st, seg_end in speech_seg2:
        seg_st_idx = int(np.floor(seg_st/frame_step))
        st_idx_on_audio2 = binarySearch(path[1], 0, len(path[1])-1, seg_st_idx)
        seg_end_idx = int(np.floor(seg_end/frame_step))
        end_idx_on_audio2 = binarySearch(path[1], 0, len(path[1])-1, seg_end_idx)
        idx_list_on_audio2 += [st_idx_on_audio2, end_idx_on_audio2]

    # match paths between
    dist_, cost_, acc_cost_, path_ = dtw(np.array(idx_list_on_audio1).reshape((len(idx_list_on_audio1), 1)),
                                     np.array(idx_list_on_audio2).reshape((len(idx_list_on_audio2), 1)),
                                     dist=lambda x, y: norm(x - y, ord=1))

    audio_seg_path = get_seg_info(path_)
    audio_seg_match = [[], []]
    for first_seg, sec_seg in zip(audio_seg_path[0], audio_seg_path[1]):
        # some filtering dtw data
        if first_seg[0] == first_seg[1] or sec_seg[0] == sec_seg[1]:
            continue
        if abs(first_seg[1]-first_seg[0]) < 0.3 or abs(sec_seg[1]-sec_seg[0]) < 0.3:
            continue

        seg_st1 = path[1][idx_list_on_audio1[first_seg[0]]]*frame_step
        seg_end1 = path[1][idx_list_on_audio1[first_seg[1]]]*frame_step
        seg_st2 = path[0][idx_list_on_audio2[sec_seg[0]]]*frame_step
        seg_end2 = path[0][idx_list_on_audio2[sec_seg[1]]]*frame_step
        audio_seg_match[0].append([seg_st2, seg_end2])
        audio_seg_match[1].append([seg_st1, seg_end1])

    # print(audio_seg_match)
    return audio_seg_match, dist, wav1_len, wav2_len


def get_audio_time_map(audio_file1, audio_file2):
    frame_step = 0.016
    # calculate mfcc
    mfcc1, energy1, rms1 = mfcc_from_file(audio_file1, sample_rate=8000)
    mfcc2, energy2, rms2 = mfcc_from_file(audio_file2, sample_rate=8000)
    wav1_len = len(rms1[0])*frame_step
    wav2_len = len(rms2[0])*frame_step

    # get vad segment
    speech_seg1, _ = detect_audio_segment_with_medi_tr(energy1[0], frame_step=frame_step, tr=0.04, medi_tr=0.007) #0.09
    # speech_seg2, _ = detect_audio_segment_with_medi_tr(energy2[0], frame_step=frame_step, tr=0.02, medi_tr=0.007) #0.09

    # get dtw cost
    dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
    print ('Normalized distance between the two sounds: {}'.format(dist))

    # list of point on audio 1
    audio_seg_match = [[], []]
    idx_list_on_audio1 = []
    # inverse search of dtw result because dtw-master results contain other's index array.
    for seg_st, seg_end in speech_seg1:
        seg_st_idx = int(np.floor(seg_st/frame_step))
        st_idx_on_audio1 = binarySearch(path[0], 0, len(path[0])-1, seg_st_idx)
        seg_end_idx = int(np.floor(seg_end/frame_step))
        end_idx_on_audio1 = binarySearch(path[0], 0, len(path[0])-1, seg_end_idx)
        idx_list_on_audio1 += [st_idx_on_audio1, end_idx_on_audio1]
        seg_st1 = path[1][st_idx_on_audio1]*frame_step
        seg_end1 = path[1][end_idx_on_audio1]*frame_step
        audio_seg_match[0].append([seg_st, seg_end])
        audio_seg_match[1].append([seg_st1, seg_end1])

    # print(audio_seg_match)
    return audio_seg_match, dist, wav1_len, wav2_len



def get_audio_time_map_dtw_python(audio_file1, audio_file2):
    frame_step = 0.016
    # calculate mfcc
    mfcc1, energy1, rms1 = mfcc_from_file(audio_file1, sample_rate=8000)
    mfcc2, energy2, rms2 = mfcc_from_file(audio_file2, sample_rate=8000)
    wav1_len = len(rms1[0])*frame_step
    wav2_len = len(rms2[0])*frame_step

    # get vad segment
    speech_seg1, _ = detect_audio_segment_with_medi_tr(energy1[0], frame_step=frame_step, tr=0.04, medi_tr=0.007) #0.09
    # speech_seg2, _ = detect_audio_segment_with_medi_tr(energy2[0], frame_step=frame_step, tr=0.02, medi_tr=0.007) #0.09

    # get dtw cost
    # dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
    alignment = dtw(mfcc1.T, mfcc2.T, keep_internals=True)
    #alignment.plot(type="threeway")
    #print ('Normalized distance between the two sounds: {}'.format(dist))
    path = [alignment.index1, alignment.index2]
    dist = alignment.distance
    nor_dist = alignment.normalizedDistance

    # list of point on audio 1
    audio_seg_match = [[], []]
    idx_list_on_audio1 = []
    # inverse search of dtw result because dtw-master results contain other's index array.
    for seg_st, seg_end in speech_seg1:
        seg_st_idx = int(np.floor(seg_st/frame_step))
        st_idx_on_audio1 = binarySearch(path[0], 0, len(path[0])-1, seg_st_idx)
        seg_end_idx = int(np.floor(seg_end/frame_step))
        end_idx_on_audio1 = binarySearch(path[0], 0, len(path[0])-1, seg_end_idx)
        idx_list_on_audio1 += [st_idx_on_audio1, end_idx_on_audio1]
        seg_st1 = path[1][st_idx_on_audio1]*frame_step
        seg_end1 = path[1][end_idx_on_audio1]*frame_step
        audio_seg_match[0].append([seg_st, seg_end])
        audio_seg_match[1].append([seg_st1, seg_end1])

    # print(audio_seg_match)
    return audio_seg_match, dist, wav1_len, wav2_len


def dtw_test_audio_dir(audio_dir):
    spk_names = []
    for spk_dir in os.listdir(audio_dir):
        spk_path = os.path.join(audio_dir, spk_dir)
        if not os.path.isdir(spk_path) or spk_dir.find("Files_") == -1:
            continue
        spk_name = spk_dir[len("Files_"):]
        #if spk_name != "7289943383":
        #    continue

        if spk_name not in spk_names:
            print(spk_name)
            spk_names.append(spk_name)
        wav_dir = os.path.join(spk_path, "good")
        if not os.path.exists(wav_dir):
            print("there is not such folder: {}".format(wav_dir))
            continue
        wav_files = []
        for wav_file in os.listdir(wav_dir):
            if wav_file.find(".wav") == -1:
                continue
            # print(wav_file)
            wav_files.append(wav_file)

        if len(wav_files) < 2:
            continue

        for idx in range(len(wav_files)//2):
            cur_idx = idx * 2
            audio_file1 = os.path.join(wav_dir, wav_files[cur_idx])
            audio_file2 = os.path.join(wav_dir, wav_files[cur_idx+1])
            print("comparing dtw between {} and {}".format(audio_file1, audio_file2))
            get_audio_time_map(audio_file1, audio_file2)


if __name__ == "__main__":
    test_audio= "voice_recordings/2020-01-19/noisy/enroll_8447290101_15791774634480.wav"
    # aa =  os.path.basename(test_audio)

    wav_dir1 = "merge/Files_+14082309381/good"
    audio_file1 = "enroll_+14082309381_15766209254.wav"
    audio_file2 = "verify_+14082309381_157664224379.wav"
    audio_file1 = os.path.join(wav_dir1, audio_file1)
    audio_file2 = os.path.join(wav_dir1, audio_file2)

    audio_file1 = "merge/Files_5106764206/good/enroll_5106764206_15795077762598.wav"
    audio_file2 = "merge/Files_5106764206/good/verify_5106764206_15786121117.wav"
    # dist_map_with_dtw(audio_file1, audio_file2)
    get_audio_time_map(audio_file1, audio_file2)
    audio_dir = "merge"
    # dtw_test_audio_dir(audio_dir)
