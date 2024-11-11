import numpy as np  
  
# 读取原始的 .npy 文件  
original_data = np.load('/home/adlab5/ssd_ad52/big_data52/competition/dataset/data/val_label.npy')  
  
# 确保原始数据的大小为 (2000, 155)  
assert original_data.shape == (2000, ), "Original data shape is not (2000, 155)"  
  
# 创建一个新的 NumPy 数组，大小为 (4599,)  
new_data = np.zeros((4307, ), dtype=original_data.dtype)  
  
# 将原始数据复制到新数组的前 2000 行  
# new_data[:2000, :] = original_data  
  
# 保存新数组为 .npy 文件  
np.save('test_label.npy', new_data)  
  
print("New .npy file created successfully.")