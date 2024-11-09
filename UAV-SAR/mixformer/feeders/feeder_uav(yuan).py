import numpy as np

from torch.utils.data import Dataset

from . import tools

import torch

coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, use_angle=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.use_angle = use_angle
        self.load_data()
        if normalization:
            self.get_mean_map()

    # def load_data(self):
    #     # data: N C V T M
    #     npz_data = np.load(self.data_path)
    #     if self.split == 'train':
    #         self.data = npz_data['x_train']
    #         self.label = np.where(npz_data['y_train'] > 0)[1]
    #         self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
    #     elif self.split == 'test':
    #         self.data = npz_data['x_test']
    #         self.label = np.where(npz_data['y_test'] > 0)[1]
    #         self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
    #     else:
    #         raise NotImplementedError('data split only supports train/test')
    #     N, T, _ = self.data.shape
    #     if self.use_angle:
    #         self.data = self.data.reshape((N, T, 2, 17, 9)).transpose(0, 4, 1, 3, 2)
    #     else:
    #         self.data = self.data.reshape((N, T, 2, 17, 3)).transpose(0, 4, 1, 3, 2)

    def load_data(self):
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train']
            self.sample_name = [self.split + '_' + str(i) for i in range(len(self.data))]
        else:
            assert self.split == 'test'
            self.data = npz_data['x_test']
            self.label = npz_data['y_test']
            self.sample_name = [self.split + '_' + str(i) for i in range(len(self.data))]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        data_numpy = self.data[idx] # T M V C
        label = self.label[idx]
        # data_numpy = torch.from_numpy(data_numpy).permute(3, 0, 2, 1) # C,T,V,M
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if(valid_frame_num == 0):
            if  self.window_size == 64:
                return torch.zeros((3, 64, 17, 2)), label, idx
            elif  self.window_size == 128:
                return torch.zeros((3, 128, 17, 2)), label, idx
            elif  self.window_size == 32:
                return torch.zeros((3, 32, 17, 2)), label, idx
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, 17, 1)) # all_joint - 0_joint
        # print(type(data_numpy))
        return data_numpy, label, idx # C T V M

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
