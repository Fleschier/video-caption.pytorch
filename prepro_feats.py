import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils

C, H, W = 3, 224, 224


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    pathPattern = os.path.join(params['video_path'], '*.mp4')
    print("video path: ", pathPattern)  # data/train-videos/TrainValVideo\*.mp4, 错误出现在路径合成上。
    # 要解决这个问题，就要在路径合成的前一个参数的末尾加上 ‘/’
    video_list = glob.glob(pathPattern)     # bug: 获取的视频list为空 原因：路径问题。原作者默认路径写错了
    print("=========== video_list: ", video_list[0])    # video_list:  data/train_videos/TrainValVideo\video0.mp4
    # 貌似windows下使用glob获取到的路径只要是自动生成的都是反斜杠，而自己指定的又是斜杠，会冲突。。。
    # 只能手动替换反斜杠为斜杠了

    #print(len(video_list),"=================")
    for video in tqdm(video_list):
        # video_id = video.split("/")[-1].split(".")[0]
        video_id = video.split("/")[-1].split('\\')[-1].split(".")[0]   # 获取视频的名称，去掉后缀。由于之前生成的路径名称有反斜杠，故要加一步处理
        print("=========== video_id: ", video_id)
        dst = params['model'] + '_' + video_id
        print("=========== dst: ", dst) # ========== dst:  resnet152_TrainValVideo\video0
        extract_frames(video, dst)      # 有问题**************************

        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        outfile = os.path.join(dir_fc, video_id + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/resnet152/', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/train_videos/TrainValVideo/', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)
    
    model = model.cuda()
    extract_feats(params, model, load_image_fn)
