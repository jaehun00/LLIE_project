import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os

batch_w = 600
batch_h = 400


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, low_img_dir, task, gt_img_dir=None):
        self.low_img_dir = low_img_dir
        self.gt_img_dir = gt_img_dir
        self.task = task
        self.train_low_data_names = []
        self.train_gt_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()

        if self.task != 'test':
            assert gt_img_dir is not None, "gt_img_dir must be provided for training or validation"
            for root, dirs, names in os.walk(self.gt_img_dir):
                for name in names:
                    self.train_gt_data_names.append(os.path.join(root, name))

            self.train_gt_data_names.sort()
            
            # Check if file names match
            for low_name, gt_name in zip(self.train_low_data_names, self.train_gt_data_names):
                assert os.path.basename(low_name) == os.path.basename(gt_name), \
                    f"File names do not match: {low_name} and {gt_name}"

        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()] 
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):
        low = self.load_images_transform(self.train_low_data_names[index])

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        if self.task == 'test':
            img_name = self.train_low_data_names[index].split('\\')[-1]
            return torch.from_numpy(low), img_name
        else:
            gt = self.load_images_transform(self.train_gt_data_names[index])
            gt = np.asarray(gt, dtype=np.float32)
            gt = np.transpose(gt[:, :, :], (2, 0, 1))
            return torch.from_numpy(low), torch.from_numpy(gt)

    def __len__(self):
        return self.count

