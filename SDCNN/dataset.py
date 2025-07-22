import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt


def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]  # h, w, channel
    x = random.randint(0, w - size)  # random number
    y = random.randint(0, h - size)

    hsize = size * scale
    hx, hy = x * scale, y * scale

    crop_lr = lr[y:y + size, x:x + size].copy()  # low-resolution patch
    crop_hr = hr[hy:hy + hsize, hx:hx + hsize].copy()  # high-resolution patch

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)  # flip up and down
        im2 = np.flipud(im2)

    if random.random() < 0.5:  # flip left and right
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])  # rotate
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        print('path:', path)
        h5f = h5py.File(path, "r")

        self.hr = [v[:] for v in h5f["HR"].values()]
        # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]

        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size
        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]

        # Print each loaded data
       # for i, (hr, lr) in enumerate(item):
       #     print(f"Index: {index}, Scale: {self.scale[i]}, HR shape: {hr.shape}, LR shape: {lr.shape}")

            # Display each loaded data
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(hr.astype(np.uint8))
            # plt.title(f"HR Image - Index: {index}, Scale: {self.scale[i]}")
            # plt.subplot(1, 2, 2)
            # plt.imshow(lr.astype(np.uint8))
            # plt.title(f"LR Image - Index: {index}, Scale: {self.scale[i]}")
            # plt.show()

        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)


class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name = dirname.split("/")[-1]
        self.scale = scale

        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join("{}/DIV2K_valid_HR".format(dirname), "*.png"))
            self.lr = glob.glob(os.path.join("{}/DIV2K_valid_LR_bicubic".format(dirname),
                                             "X{}/*.png".format(scale)))
        elif 'Set' in self.name:  # Set5, Set14,B100
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]
        elif '100' in self.name:  # Set5, Set14,B100
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]    
        elif 'RealSR' in self.name:
            self.hr = glob.glob(os.path.join(dirname, "HR", "*.png"))
            self.lr = glob.glob(os.path.join(dirname, "bicubic_x{}".format(scale), "*.png"))
        else:
            self.hr = glob.glob(os.path.join(dirname, "HR", "*.png"))
            self.lr = glob.glob(os.path.join(dirname, "bicubic_x{}".format(scale), "*.png"))

        self.hr.sort()
        self.lr.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])
        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        filename = self.hr[index].split("/")[-1]

        # Print each loaded data
      #  print(f"Index: {index}, Filename: {filename}, HR shape: {hr.size}, LR shape: {lr.size}")

        # Display each loaded data
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(hr))
        # plt.title(f"HR Image - Index: {index}, Filename: {filename}")
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(lr))
        # plt.title(f"LR Image - Index: {index}, Filename: {filename}")
        # plt.show()

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)