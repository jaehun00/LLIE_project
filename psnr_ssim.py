import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable

from model import Finetunemodel

import utils
from multi_read_data import MemoryFriendlyLoader
import easydict
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchprofile

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default="./data/eval15/low", help='batch size')
parser.add_argument('--high_path', type=str, default="./data/eval15/high", help='batch size')
parser.add_argument('--model', type=str, default="./Lap/Train6", help='batch size')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()

model_path = args.model

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)

def calculate_flops(model):
    input_size = (1, 3, 400, 600)
    inputs = torch.randn(input_size)
    inputs = inputs.cuda()

    flops = torchprofile.profile_macs(model, inputs)
    return flops

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel(model_path)
    model = model.cuda()
    model.eval()
    MB = utils.count_parameters_in_MB(model)
    print('model size = ', MB, 'MB')
    # FLOPs 계산
    flops = calculate_flops(model)
    gflops = flops / 1e9
    print(f"Total FLOPs: {flops:.2e} FLOPs")
    print(f"Total GFLOPs: {gflops:.4e} GFLOPs")
    psnr_values = []
    ssim_values = []
    diff_values = []
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = input.cuda()
            print("generated_image:", image_name)

            image_name = image_name[0].split('/')[-1].split('.')[0]

            r_name = '%s.png' % (image_name)
            #print(r_name)
            real_path = args.high_path + '/' + r_name
            print("real_image:", real_path)
            i, r = model(input)
            u_name = '%s.png' % (image_name + '_' + "test")
            #print('processing {}'.format(u_name))

            # 이미지를 저장하지 않고 바로 PSNR 및 SSIM을 계산합니다.
            real_image = cv2.imread(real_path)
            generated_image = (np.transpose(r.squeeze().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)

            #contrast = cv2.imread(args.contrast)

            o_height, o_width, o_channel = real_image.shape
            generated_image = cv2.resize(generated_image, dsize=(o_width, o_height), interpolation = cv2.INTER_AREA)

            o_gray = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

            (ssim_val, diff_val) = ssim(o_gray, c_gray, full = True)
            diff_val = (diff_val*255).astype('uint8')

            # PSNR을 계산합니다.
            psnr_val = psnr(real_image, generated_image)

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print("Average PSNR:", avg_psnr)
    print("Average SSIM:", avg_ssim)

if __name__ == '__main__':

    main()
