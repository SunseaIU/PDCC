import warnings
import numpy as np
warnings.filterwarnings("ignore")



def get_combined_virtual_mid_source(Xt, Y_tar_pseudo_list, threshold, alpha, p=3):
    Y_tar_vote = np.squeeze(np.mean(Y_tar_pseudo_list, axis=0))

    ins_std_all = []
    ins_power_mean_all = []

    for ni in range(len(Xt)):
        tmp_prob = np.squeeze(Y_tar_pseudo_list[:, ni, :])

        # 计算标准差 (Standard Deviation, SD)
        tmp_std = np.mean([np.std(tmp_prob[:, i]) for i in range(tmp_prob.shape[1])])
        ins_std_all.append(tmp_std)

        # 计算幂平均 (Power Mean, PM)
        power_mean = np.power(np.mean(np.power(tmp_prob, p)), 1 / p)
        ins_power_mean_all.append(power_mean)

    # 将两个标准的得分进行加权组合 (Combined Score, CS)
    combined_scores = alpha * np.array(ins_power_mean_all) + (1 - alpha) * (1 - np.array(ins_std_all))

    # 选择组合得分高于阈值的样本 (Selected Indices, SI)
    idx_select = np.where(combined_scores > threshold)[0]

    print('select ins num ', len(idx_select))
    Ys_mid = Y_tar_vote[idx_select, :].argmax(axis=1)
    Xs_mid = Xt[idx_select, :]
    print(Xs_mid.shape, Ys_mid.shape)
    return Xs_mid, Ys_mid
