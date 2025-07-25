# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
import torch
from solver import Solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='sdcnn') 
    parser.add_argument("--ckpt_name", type=str, default='SDCNN')  

    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--train_data_path", type=str, default="../dataset/DIV2K_train.h5")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint")
    parser.add_argument("--sample_dir", type=str, default="sample/")

    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--scale", type=int, default=4)

    parser.add_argument("--verbose", action="store_true", default=True)  # 修复默认值
    parser.add_argument("--group", type=int, default=1)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=600000)
    parser.add_argument("--decay", type=int, default=150000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--resume", type=bool, default=False)

    parser.add_argument("--loss_fn", type=str, 
                        choices=["MSE", "L1", "SmoothL1", "Charbonnier", "Perceptual"], default="MSE")
    parser.add_argument("--multi_scale", type=bool, default=False)  
    return parser.parse_args()


def main(cfg):
    net = importlib.import_module("model.{}".format(cfg.model)).SDCNN  
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
    solver = Solver(net, cfg)
    solver.fit()

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)