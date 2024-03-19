import os.path

from sklearn.mixture import GaussianMixture
import numpy as np
from gmm_model import MAX_ITER, N_COMPONENTS
from gmm_model import save_gmm_to_npz, load_gmm_from_npz


def load_training_data(data_file):
    train_data = np.load(data_file)
    mfcc = train_data["mfcc"]
    return mfcc


def train_gmm_model(same_data_file, diff_data_file):
    same_mfcc_data = load_training_data(same_data_file)
    diff_mfcc_data = load_training_data(diff_data_file)
    print("same data len: {}".format(len(same_mfcc_data)))
    print("diff data len: {}".format(len(diff_mfcc_data)))
    same_model_path = "models/same_model.npz"
    if os.path.exists(same_model_path):
        same_gmm = load_gmm_from_npz(same_model_path)
    else:
        same_gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type="diag", max_iter=MAX_ITER)

    same_gmm.fit(same_mfcc_data)

    save_gmm_to_npz(same_model_path, same_gmm)
    print("saved same voice model")

    diff_model_path = "models/diff_model.npz"
    if os.path.exists(diff_model_path):
        diff_gmm = load_gmm_from_npz(diff_model_path)
    else:
        diff_gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type="diag", max_iter=MAX_ITER)
    diff_gmm.fit(diff_mfcc_data)

    save_gmm_to_npz(diff_model_path, diff_gmm)
    print("saved diff voice model")


if __name__ =="__main__":
    same_data_file = "voice_command.npz"
    diff_data_file = "diff_command.npz"
    train_gmm_model(same_data_file, diff_data_file)
