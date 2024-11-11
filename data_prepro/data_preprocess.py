import numpy as np

# 输入文件的路径
# x_train_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_bone.npy'
# y_train_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_label.npy'
# x_val_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_bone.npy'
# y_val_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_label.npy'
x_train_path = '/home/zjl_laoshi/quminghaonan/dataset/data/train_bone.npy'
y_train_path = '/home/zjl_laoshi/quminghaonan/dataset/data/train_label.npy'
x_val_path = '/home/zjl_laoshi/quminghaonan/dataset/data/test_bone.npy'
y_val_path = '/home/zjl_laoshi/quminghaonan/dataset/data/test_label.npy'

# 加载数据
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)
#print(x_train.shape)

# 保存为 .npz 文件
output_path = '/home/zjl_laoshi/quminghaonan/dataset/data_preprocessed/MMVRAC_CSv2_test_bone.npz'
np.savez(output_path, x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val)

print(f'数据已保存到 {output_path}')
