import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import skimage.measure as measure
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sdcnn") 
    parser.add_argument("--ckpt_path", type=str,
                        default="/data1/wza/SDCNN-main/checkpoint/SDCNN/dsrnet_x3_669000.pth") 
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str, default="SDCNN/results/SDCNN") 
    parser.add_argument("--test_data_dir", type=str, default="/data1/wza/SDCNN-main/SDCNN/Set5")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--loss_fn", type=str, default="L1", choices=["MSE", "L1", "SmoothL1"])
    return parser.parse_args()

def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64) - min_val) / (max_val - min_val)
        return out

    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = compare_psnr(im1, im2, data_range=1)
    return psnr

def calculate_ssim(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def sample(net, device, dataset, cfg):
    scale = cfg.scale
    mean_psnr = 0
    mean_ssim = 0

    for step, (hr, lr, name) in enumerate(dataset):
        if "DIV2K1" in dataset.name:
            t1 = time.time()
            h, w = lr.size()[1:3]
            h_half, w_half = int(h / 2), int(w / 2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            lr_patch = torch.tensor((4, 3, h_chop, w_chop), dtype=torch.float)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
            lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
            lr_patch = lr_patch.to(device)

            sr = net(lr_patch, cfg.scale).detach()

            h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
            w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

            result = torch.tensor((3, h, w), dtype=torch.float).to(device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
            sr = result
            t2 = time.time()
        else:
            t1 = time.time()
            lr = lr.unsqueeze(0).to(device)
            sr = net(lr, cfg.scale).detach().squeeze(0)
            t2 = time.time()

        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        sr_dir = os.path.join(cfg.sample_dir,
                              model_name,
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "SR")
        hr_dir = os.path.join(cfg.sample_dir,
                              model_name,
                              cfg.test_data_dir.split("/")[-1],
                              "x{}".format(cfg.scale),
                              "HR")
        if not os.path.exists(sr_dir):
            os.makedirs(sr_dir, mode=0o777)
        if not os.path.exists(hr_dir):
            os.makedirs(hr_dir, mode=0o777)

        sr_im_path = os.path.join(sr_dir, "{}".format(name.replace("HR", "SR")))
        hr_im_path = os.path.join(hr_dir, "{}".format(name))
        save_image(sr, sr_im_path)
        save_image(hr, hr_im_path)

        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        bnd = scale
        sr_1 = rgb2ycbcr(sr)
        hr_1 = rgb2ycbcr(hr)
        psnr_value = psnr(sr_1, hr_1)
        ssim_value = calculate_ssim(sr_1, hr_1)

        mean_psnr += psnr_value / len(dataset)
        mean_ssim += ssim_value / len(dataset)

        print(step, psnr_value, ssim_value)

    print(mean_psnr, mean_ssim)

def main(cfg):
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.SDCNN(scale=cfg.scale, group=cfg.group)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    net.eval()
    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    sample(net, device, dataset, cfg)

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)