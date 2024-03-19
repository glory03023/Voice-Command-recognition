from sklearn.mixture import GaussianMixture
import numpy as np
from os.path import join, dirname, realpath


MODEL_DIR = "model"


def get_model_path(model_name):
    """Get the path to the model name provided
    :param str model_name: the model name
    :rtype: str
    :return: the path to the provided model name
    """
    return join(dirname(realpath(__file__)), MODEL_DIR, model_name)


N_COMPONENTS = 128
MAX_ITER = 200
#MAX_ITER = 2000


def load_gmm_from_npz(model_path):
    """Load parameters of a Gaussian Mixture Model from an npz file
    :param str model_filename: the filename of the model
    :rtype: GaussianMixture
    :return: a Gaussian Mixture Model
    """
    #model_path = get_model_path(model_filename)
    gmm_params = np.load(model_path)

    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type="diag", max_iter=MAX_ITER)
    gmm.weights_ = gmm_params['weight']
    gmm.means_ = gmm_params['means']
    gmm.covariances_ = gmm_params['covarriances']
    gmm.precisions_ = gmm_params['precisions']
    gmm.precisions_cholesky_ = gmm_params['precision_choesky']
    gmm.lower_bound_ = gmm_params['lowbounds']
    gmm.converged_ = True
    return gmm


def save_gmm_to_npz(model_path, gmm):
    """Save parameters of a Gaussian Mixture Model from an npz file
    :param str model_filename: the destination of the model
    :param GaussianMixture gmm: the GMM to save
    """
    #model_path = get_model_path(model_filename)
    np.savez(
        model_path,
        weight=gmm.weights_,
        means=gmm.means_,
        covarriances=gmm.covariances_,
        precisions=gmm.precisions_,
        precision_choesky=gmm.precisions_cholesky_,
        lowbounds=gmm.lower_bound_
    )
