import numpy as np

data_path = '/home/zjl_laoshi/quminghaonan/dataset/data_preprocessed/MMVRAC_CSv2_test_bone_motion.npz'
npz_data = np.load(data_path)

#train
train_data = npz_data['x_train']
train_label = npz_data['y_train']
train_data = train_data[:,[0,2,1],:,:,:]
train_data = train_data.transpose(0, 2, 4, 3, 1)
train_data = train_data.reshape(16724, 300, -1)   #train 16724×3   50172    val 2000      test 4307

num_classes = 155
train_label = np.eye(num_classes)[train_label]

#test
test_data = npz_data['x_test']
test_label = npz_data['y_test']
test_data = test_data[:,[0,2,1],:,:,:]
test_data = test_data.transpose(0, 2, 4, 3, 1)
test_data = test_data.reshape(4307, 300, -1)   #train 16724×3   50172    val 2000      test 4307

num_classes = 155
test_label = np.eye(num_classes)[test_label]

output_path = '/home/zjl_laoshi/quminghaonan/dataset/data_sar/MMVRAC_CSv2_test_bone_motion.npz'
np.savez(output_path, x_train=train_data, y_train=train_label, x_test=test_data, y_test=test_label)

print(f'数据已保存到 {output_path}')






