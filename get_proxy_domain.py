import numpy as np
import joblib
from scipy.io import savemat
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from data_loader import read_bci_data
from select_proxy_sample import get_combined_virtual_mid_source
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier


def pre_train_models_combine_voting(Xs, Ys, root_path, mdl_list):
    if mdl_list == 'knn_soft':
        clf4 = KNeighborsClassifier(n_neighbors=1)
        clf4.fit(Xs, Ys.ravel())
        mdl_path = root_path + 'mdl_' + 'knn' + '.pkl'
        joblib.dump(clf4, mdl_path)


    if mdl_list == ['svm_soft','lda_soft', 'lr_soft']:
        clf1 = SVC()
        clf2 = LDA(solver='lsqr', shrinkage='auto')
        clf3 = LogisticRegression(penalty='l2', max_iter=500)

        clf_dict = dict(svm_soft=clf1,lda_soft=clf2, lr_soft=clf3)
        for idx in range(len(mdl_list)):
            clf_base = clf_dict[mdl_list[idx]]
            clf_base.fit(Xs, Ys.ravel())
            mdl_path = root_path + 'mdl_' + str(idx) + '.pkl'
            joblib.dump(clf_base, mdl_path)




data_name_list = ['001-2014', '002-2014','MI1','2015001']
data_name = data_name_list[0]

if data_name == '001-2014': num_sub, chn = 9, 22  # MI2-4
if data_name == '002-2014': num_sub = 14
if data_name == 'MI1': num_sub, chn = 9, 22  # MI2-2
if data_name == '2015001': num_sub = 12


# Generate Source Models except target sub t
mdl_list = ['svm_soft','lda_soft', 'lr_soft']

# for t in range(0, num_sub):
#     print('target', t)
#     Xs, Ys, Xt, Yt = read_bci_data(t, data_name, align=True, cov_type='lwf',aug=True)
#
#     print(Xs.shape,Ys.shape,Xt.shape,Yt.shape)
#     root_path = './model/' + data_name + '_TL_src_NoAlign_' + str(t+1) + '_'
#     pre_train_models_combine_voting(Xs, Ys, root_path, mdl_list)
#
# print('finished pre-training...\n')

#
# # 实验参数设置
# param_values = np.round(np.arange(0.56, 0.81, 0.02), 2)  # 0.4-0.8步长0.05
# results = {tar: [] for tar in range(num_sub)}  # 存储结果
#
# # 预先加载所有数据
# print('Loading all subjects data...')
# data_dict = {}
# for tar in range(num_sub):
#     data_dict[tar] = read_bci_data(tar, data_name, align=True, cov_type='lwf',aug=True)
#
# # 对不同参数进行实验
# for param in param_values:
#     print(f'\nTesting parameter value: {param:.2f}')
#     for tar in range(num_sub):
#         Xs, Ys, Xt, Yt = data_dict[tar]
#
#         # 加载预训练模型
#         root_path = f'./model/{data_name}_TL_src_NoAlign_{tar + 1}_'
#         Y_tar_pseudo_list = []
#         for idx in [1, 2]:  # LDA和LR模型
#             clf = joblib.load(root_path + f'mdl_{idx}.pkl')
#             Y_tar_pseudo_list.append(clf.predict_proba(Xt))
#
#         # 生成中间源数据
#         Xs_mid, Ys_mid = get_combined_virtual_mid_source(
#             Xt,
#             np.array(Y_tar_pseudo_list),
#             threshold=param,
#             alpha=0.7
#         )
#
#         # 训练和评估模型
#         model = LDA(solver='lsqr', shrinkage='auto')
#         model.fit(Xs_mid, Ys_mid.ravel())
#         acc2 = accuracy_score(Yt, model.predict(Xt))
#         results[tar].append(acc2)
#
#         print(f'Sub{tar + 1}: {acc2:.4f}', end=' | ')
#
#
#
# # 绘制结果（保持原始样式）
# plt.figure(figsize=(12, 6))
# colors = plt.cm.get_cmap('tab10', num_sub)  # 恢复原始配色
#
# # 在绘制图表前添加以下代码计算最小精度
# all_acc = [acc for subj in results.values() for acc in subj]
#
# min_acc = np.floor(min(all_acc) * 10) / 10 + 0.1  # 精确向下取整到0.1
# max_acc = 1.0  # 保持最大值为1
#
# for tar in range(num_sub):
#     plt.plot(
#         param_values,
#         results[tar],
#         marker='o',
#         linestyle='-',
#         linewidth=1.5,
#         markersize=4,
#         color=colors(tar),
#         label=f'Subject {tar + 1}'
#     )
#
# # 设置纵轴范围
# plt.ylim(min_acc - 0.05, max_acc + 0.05)  # 留出5%边距
#
# # 调整刻度（自动生成合理间隔）
# from matplotlib.ticker import MultipleLocator
# plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))
#
# # 其余保持原样
# plt.title(f'Effect of threshold on Model Accuracy')
# plt.xlabel('Threshold')
# plt.ylabel('Accuracy')
# plt.xticks(param_values, rotation=30)
# plt.grid(True, alpha=0.3, which='both')  # 显示主次网格线
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()




# 初始化存储结果
all_accuracies = []
all_sample_counts = []

for tar in range(0, num_sub):
    # 读取数据
    Xs, Ys, Xt, Yt = read_bci_data(tar, data_name, align=True, cov_type='lwf')

    # 初始模型评估
    baseline = LDA(solver='lsqr', shrinkage='auto')
    acc = accuracy_score(Yt, baseline.fit(Xs, Ys.ravel()).predict(Xt))
    print(f'Primary accuracy (target {tar+1}):', acc)
#
    # 加载源模型伪标签
    root_path = f'./model/{data_name}_TL_src_NoAlign_{tar+1}_'
    Y_tar_pseudo_list = []
    mdl_idx = [1, 2]  # LDA 和 LR 的模型索引
    for idx in mdl_idx:
        clf = joblib.load(root_path + f'mdl_{idx}.pkl')
        Y_tar_pseudo = clf.predict_proba(Xt)
        Y_tar_pseudo_list.append(Y_tar_pseudo)
    Y_tar_pseudo_list = np.array(Y_tar_pseudo_list)

    # 多模型投票生成伪标签
    Y_tar_vote = np.mean(Y_tar_pseudo_list, axis=0)  # 计算每类的概率平均值
    Yt_pred = np.argmax(Y_tar_vote, axis=1)  # 获得伪标签
    acc1 = accuracy_score(Yt, Yt_pred)
    print(f'After source model pretraining, accuracy (target {tar+1}):', acc1)
#
    Xs_mid, Ys_mid = get_combined_virtual_mid_source(Xt, Y_tar_pseudo_list, 0.58, 3)

    baseline = LDA(solver='lsqr', shrinkage='auto')
    acc2 = accuracy_score(Yt, baseline.fit(Xs_mid, Ys_mid.ravel()).predict(Xt))
    print(f'vvvvvvvvvvvvvvvvvvv dddddddddddddddddd (target {tar+1}):', acc2)

    from pathlib import Path
    Path('./data/data_select/').mkdir(parents=True, exist_ok=True)
    subject_count = num_sub
    print(Xs_mid)
    # 构建要保存的数据字典
    save_dict = {
        'x': Xs_mid,
        'y': Ys_mid
    }
    output_file = f'{'./data/data_select/'}{data_name}subject_{tar+1}_select.mat'
    # 保存对齐后的数据
    savemat(output_file, save_dict)
    print(f'Saved {output_file}')


