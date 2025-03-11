from scipy import io
import warnings
warnings.filterwarnings("ignore", message="oneDNN custom operations are on.")
warnings.filterwarnings("ignore", message="ModuleNotFoundError: Tensorflow is not installed.")
warnings.filterwarnings("ignore")
from data_augment import data_aug
import scipy.io

from scipy.sparse import spdiags
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from data_align import centroid_align
import numpy as np


def read_bci_data(idt, data_name, cov_type, aug):
    file = './data/' + data_name + '.npz'

    MI = np.load(file)
    Data_raw = MI['x']
    n_sub = len(Data_raw)
    Label = MI['y']

    # MTS transfer
    tar_data = np.squeeze(Data_raw[idt, :, :, :])
    tar_label = np.squeeze(Label[idt, :])
    ca = centroid_align(center_type='riemann', cov_type=cov_type)  # 'lwf', 'oas'
    _, tar_data = ca.fit_transform(tar_data)

    covar_src = Covariances(estimator=cov_type).transform(tar_data)
    tar_fea = TangentSpace().fit_transform(covar_src)

    # MTS transfer
    ids = np.delete(np.arange(0, n_sub), idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        tmp_data = np.squeeze(Data_raw[ids[i]])
        tmp_lbl = np.squeeze(Label[ids[i]])
        if aug:
            sample_size = tmp_data.shape[2]  # (288, 22, 750)
            # mult_flag, noise_flag, neg_flag, freq_mod_flag
            flag_aug = [False, False, False, False]
            tmp_data = np.transpose(tmp_data, (0, 2, 1))
            tmp_data, tmp_lbl = data_aug(tmp_data, tmp_lbl, sample_size, flag_aug)
            tmp_data = np.transpose(tmp_data, (0, 2, 1))
            tmp_lbl = tmp_lbl.astype(int)
        ca = centroid_align(center_type='riemann', cov_type=cov_type)  # 'lwf', 'oas'
        _, tmp_data = ca.fit_transform(tmp_data)
        src_data.append(tmp_data)
        src_label.append(tmp_lbl)


    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)
    src_label = np.squeeze(src_label)

    covar_tar = Covariances(estimator=cov_type).transform(src_data)
    src_fea = TangentSpace().fit_transform(covar_tar)
    return src_fea, src_label, tar_fea, tar_label


