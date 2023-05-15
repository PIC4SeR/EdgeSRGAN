import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

# GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import argparse
import cv2
import glob
import time
import numpy as np
from collections import OrderedDict
import torch

from utils.util_calculate_psnr_ssim import to_y_channel, calculate_psnr, calculate_ssim
from utils.tools import read_yaml, Logger
from utils.train import Trainer
from utils.niqe import niqe as calculate_niqe
from lpips import LPIPS, im2tensor
from utils.dev.excellogger import Logger as ExcelLogger
from utils.dev.excellogger import Column

SPREADSHEET = "1x0rL1QPW0bFAZK9VTtW1aS_FDdyv--Ku7ihegmzYGAg"
SHEET = "Results"
key_json = "utils/excel_key.json"
ROW_BASE = 0
COLUMN_BASE = 18

DATASETS = ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 4, 8') # 1 for dn and jpeg car ####################
    parser.add_argument('--model', type=str, default='tiny')
    parser.add_argument('--train_type', type=str, default='pre')
    parser.add_argument('--dataset_lr', type=str, default='/media/Datasets/super-resolution/Set5/LR_bicubic/',
                        help='input low-quality test image folder')
    parser.add_argument('--dataset_hr', type=str, default='/media/Datasets/super-resolution/Set5/HR',
                        help='input ground-truth test image folder')
    parser.add_argument('--config', default='config.yaml', type=str, help='Config path')
    parser.add_argument('--predict', help='Predict', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--excel_row', help='Excel row', default=3, type=int)
    parser.add_argument('--excel', help='Excel', default=True, action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    if args.excel:
        excel = ExcelLogger(SPREADSHEET, key_json)
    
    DATASET = args.dataset_hr.split('/')[-2]
    print(f"\n\nTesting on {DATASET}")
    
    logger = Logger('result_log.txt')
    calculate_lpips = LPIPS(net='alex', verbose=False).cuda()
    
    config = read_yaml('config.yaml')

    weights = f'weights/srgan/{args.model}'
      
    config['MODE'] = 'TEST'
    
    if args.predict:    
        model = get_model(config, weights)
        # resolve(model, np.random.rand(1,200,200,3))
    
    # setup folder and path
    #     save_dir = f'results/{args.model}'
    #     os.makedirs(save_dir, exist_ok=True)
    
    border = config['SCALE']
    
    test_results = OrderedDict()
    for metric in ['psnr', 'ssim', 'lpips', 'niqe', 'timing']:
        test_results[metric] = []
    
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.dataset_hr, '*')))):
        # read image
        imgname, img_lr, img_hr = get_image_pair(path, args.dataset_lr, args.scale)  # image to HWC-BGR, float32
        # inference
        h_lr, w_lr = img_lr.shape[:-1]
        if args.predict:            
            output, timing = resolve(model, img_lr[None,...,::-1])
            output = output[0]       
        else:
            h_lr, w_lr = h_lr//args.scale, w_lr//args.scale
            output = img_lr[...,::-1]
            timing = -1
        
        test_results['timing'].append(timing)          
        output = output[:h_lr * args.scale, :w_lr * args.scale][:,:,::-1] # RGB to BGR
        img_hr = img_hr[:h_lr * args.scale, :w_lr * args.scale]

#         # save image
#         output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         if output.ndim == 3:
#             output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
#         output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
#         cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

        # evaluate psnr/ssim/nique/lpips      
        
        output_y = to_y_channel(output)
        img_hr_y = to_y_channel(img_hr)

        psnr = calculate_psnr(output_y, img_hr_y, crop_border=border)
        ssim = calculate_ssim(output_y, img_hr_y, crop_border=border)
        with torch.no_grad():
            lpips = calculate_lpips(im2tensor(output[...,::-1]).cuda(), im2tensor(img_hr[...,::-1]).cuda()).item()
        niqe = calculate_niqe(output_y)
        
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        test_results['lpips'].append(lpips)
        test_results['niqe'].append(niqe)
        
        print('Testing {:d} {:10s} - PSNR: {:.4f} dB; SSIM: {:.4f}; '
              'LPIPS: {:.4f}; NIQE: {:.4f}; \ttiming: {:.4f} ms '.format(idx, imgname, psnr, ssim, lpips, niqe, timing))

    # summarize psnr/ssim/lpips/niqe
    ave_psnr = np.mean(test_results['psnr'])
    ave_ssim = np.mean(test_results['ssim'])
    ave_lpips = np.mean(test_results['lpips'])
    ave_niqe = np.mean(test_results['niqe'])
    ave_timing = np.mean(test_results['timing'])
    print('\nAverage PSNR/SSIM/LPIPS/NIQE: {:.4f} dB; {:.4f}; {:.4f}; {:.4f}'.format(ave_psnr, ave_ssim, ave_lpips, ave_niqe))
    print('Average timing: {:.4f} ms'.format(ave_timing))
    text = '{:s}: PSNR/SSIM/LPIPS/NIQE {:.4f} dB; {:.4f}; {:.4f}; {:.4f}'.format(DATASET, ave_psnr, ave_ssim, ave_lpips, ave_niqe)
    logger.save_log(text)
    
    if args.excel:
        excel.write(f'{args.model} {args.train_type}', SHEET, ROW_BASE + args.excel_row, Column('A'))
        column = Column('B') + COLUMN_BASE + DATASETS.index(DATASET)*4
        excel.write([ave_psnr, ave_ssim, ave_lpips, ave_niqe], SHEET, ROW_BASE + args.excel_row, column, None, column+3)
    
            
def get_model(config, model_weights):
    trainer = Trainer(config=config)
    model = trainer.generator
    model.load_weights(model_weights, by_name=False, skip_mismatch=False)
    return model
    
            
def get_image_pair(path, dataset_lr, scale=4, normalize=False):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_gt = cv2.imread(path, cv2.IMREAD_COLOR)
    img_lq = cv2.imread(f'{dataset_lr}/X{scale}/{imgname}x{scale}{imgext}', cv2.IMREAD_COLOR)
    if normalize:
        img_gt = img_gt/255.
        img_lq = img_lq/255.
    return imgname, img_lq, img_gt


            
def resolve(model, lr_batch, to_numpy=True):
    lr_batch = tf.cast(lr_batch, tf.float32)
    t0 = time.time()
    sr_batch = model(lr_batch, training=False)[0]
    t1 = time.time()
    timing = (t1 - t0)*1000
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    if to_numpy:
        sr_batch = sr_batch.numpy()
    return sr_batch, timing


if __name__ == '__main__':
    main()
