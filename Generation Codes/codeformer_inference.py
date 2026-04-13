import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY

import pickle
import numpy as np
from imutils.face_utils import rect_to_bb
from torchvision.transforms import GaussianBlur
from torchvision.transforms.v2 import GaussianNoise
from torchvision.utils import save_image
from tqdm import tqdm
import time

class ImageDegrade:
    def __init__(self, thickness=30, global_kernel = (5,2), local_kernel = (15,5), global_noise = 0.05, local_noise = 0.15):
        self.thickness = thickness
        self.global_kernel = GaussianBlur(global_kernel[0],global_kernel[1])
        self.local_kernel = GaussianBlur(local_kernel[0],local_kernel[1])
        self.global_noise = GaussianNoise(0,global_noise)
        self.local_noise = GaussianNoise(0,local_noise)

    def __call__(self,img,x, y, w, h):
        mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
        mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 1, self.thickness)
        mask = torch.from_numpy(mask).unsqueeze(0)
        inv_mask = 1-mask
        
        blured = self.local_kernel(img)
        blured = mask*blured + inv_mask*img
        blured = self.global_kernel(blured)
        noised = self.local_noise(blured)
        noised = mask*noised + inv_mask*blured
        augmented = self.global_kernel(noised)
        augmented = self.global_noise(augmented)

        return augmented
    
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5)
    parser.add_argument('-i', '--input_path', type=str, default='/home/simitlab/omer_storage/SynFaceResearch/GANmut/GeneratedDatasets/SynFace70V_DA_10/images_aligned')
    parser.add_argument('-o', '--output_path', type=str, default='/home/simitlab/omer_storage/SynFaceResearch/GANmut/GeneratedDatasets/SynFace70V_DA_10/images_aligned_codeformer')
    parser.add_argument('-a','--augment', action='store_true', default=True)
    parser.add_argument('-d', '--detections_path', type=str, default='/home/simitlab/omer_storage/SynFaceResearch/GANmut/Generator-Scripts/detections.pkl')
    parser.add_argument('-u','--use_realesrgan', action='store_true', default=False)

    args = parser.parse_args()
    
    w = args.fidelity_weight
    if args.use_realesrgan:
        realesrgan = set_realesrgan()

    # ------------------------ input & output ------------------------

    if args.input_path.endswith('/'):  # solve when path ends with /
        args.input_path = args.input_path[:-1]
    input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
    result_root = args.output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n' 
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()
    with torch.no_grad():
        net = torch.compile(net, mode="reduce-overhead")
        #net = torch.compile(net, mode="max-autotune")
        output = net(torch.ones(1,3,512,512).float().cuda(), w=w, adain=True)[0]

    augmenter = ImageDegrade()
    
    with open(args.detections_path, 'rb') as f:
        dets_all = pickle.load(f)
    
    time_count = 0
    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(tqdm(input_img_list)):
        
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        #print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        cropped_face = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        det_name = '../FFHQ_thumbnails256x256/'+basename.split('_')[0]+'.png'
        dets = dets_all[det_name]
        if len(dets)==0:
            print("no face for",basename)
            continue
        (x_origin, y_origin, width, height) = rect_to_bb(dets[0])
        scale_x,scale_y = 512.0/cropped_face.shape[0],512.0/cropped_face.shape[1]
        x_origin, y_origin, width, height = int(x_origin*scale_x),int(y_origin*scale_y),int(width*scale_x),int(height*scale_y)

        # the input faces are already cropped and aligned
        if args.use_realesrgan:
            cropped_face = realesrgan(cropped_face) # BOMBA böyle mi çalışıyo kontrol etmedim 
        else:
            #cropped_face = cv2.resize(cropped_face, (512, 512), interpolation=cv2.INTER_LINEAR)
            cropped_face = cv2.resize(cropped_face, (512, 512), interpolation=cv2.INTER_CUBIC)


        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        if args.augment:
            cropped_face_t = augmenter(cropped_face_t,x_origin, y_origin, width, height)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        
        try:
            with torch.no_grad():
                time1 = time.time()
                output = net(cropped_face_t, w=w, adain=True)[0]
                time2 = time.time()
                time_count+=time2-time1
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for CodeFormer: {error}')
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')

        save_face_name = f'{basename}.png'

        save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
        imwrite(restored_face, save_restore_path)
    print(time_count)
