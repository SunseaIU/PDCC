% 得到一个时段15个对象的所有样本,以及对应的标签
% 需要修改的参数
% load_dir:数据集存放的系统路径
% session_name:不同时段的名字，选择不同的session时要选择不同的mat_name(i)
% mat_name(i):初始mat文件的名字，
% save_dir: 输出的.mat文件的保存路径,需要新建session1、session2、session3个文件夹用来存放输出数据

% 得到的输出，3个session文件下的15个subject，每个subject对应一个对象观看视频的所有脑电数据集，其中有fea:x*310，label:x*1;

% 每个mat中有4组，psd_movingAve，psd_LDS，de_movingAve，de_LDS分别是PSD滑动平均，PSD线性动力系统平滑，DE滑动平均以及DE线性动力系统平滑特征，仅选取de_LDS组数据集

clc; close all; clear;
%对输出进行初始化
mat_name1 = {'1_20160518','2_20150915','3_20150919','4_20151111','5_20160406','6_20150507','7_20150715','8_20151103','9_20151028','10_20151014','11_20150916','12_20150725','13_20151115','14_20151205','15_20150508'};
mat_name2 = {'1_20161125','2_20150920','3_20151018','4_20151118','5_20160413','6_20150511','7_20150717','8_20151110','9_20151119','10_20151021','11_20150921','12_20150804','13_20151125','14_20151208','15_20150514'};
mat_name3 = {'1_20161126','2_20151012','3_20151101','4_20151123','5_20160420','6_20150512','7_20150721','8_20151117','9_20151209','10_20151023','11_20151011','12_20150807','13_20161130','14_20151215','15_20150527'};

for num = 1:15
fea = [];
label = [];
for i=1:24
    
    %通过设置de_LDS+'j'找到每个人的数据
    array_name = 'de_LDS';
    array_name = strcat(array_name,num2str(i));
    
    %加载数据
    load_dir = 'C:\Users\Liu\Desktop\62702162089005c4302623bd8fa9a67a_c25dce2f3682bed20b7e0a8fc1ffd6d2_8\eeg_feature_smooth';
    session_name = '\1\';
    in_mat_name = strcat(cell2mat(mat_name1(num)),'.mat');
    load_dir = strcat(load_dir,session_name,in_mat_name);
    raw_struct = load(load_dir,array_name);%修改.mat的名字得到其他人的数据，修改[1,2,3]得到不同的时段
    raw_data = getfield(raw_struct,array_name);
    
    %获取数据大小
    [size_x,size_y,size_z] = size(raw_data);
    
    %对数据进行处理，例如62*42*5转换成42*62*5，得到特征
    fea_out = permute(raw_data,[2,1,3]);
    fea_out = reshape(fea_out,size_y,310);
    
    %得到标签
    session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
    session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
    session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];
    
    label_out = [];
    for j=1:size_y
        label_out(j,1) = session1_label(i);
    end
    
    fea = [fea;fea_out];
    label = [label;label_out];
    %对数据进行保存
    save_dir = 'C:\Users\Liu\Desktop\62702162089005c4302623bd8fa9a67a_c25dce2f3682bed20b7e0a8fc1ffd6d2_8\data\session1';
    outmat_name = strcat('\subject',num2str(num),'.mat');
    save_dir = strcat(save_dir,outmat_name);
    save(save_dir,'fea','label')
end
end


