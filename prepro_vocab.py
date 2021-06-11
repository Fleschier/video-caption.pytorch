import re
import json
import argparse
import numpy as np

import os
import glob

from tqdm.std import tqdm

def build_vocab(vids, params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
    for vid, caps in vids.items():
        caps = caps['captions']
        vids[vid]['final_captions'] = []
        for cap in caps:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            caption = [
                '<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            vids[vid]['final_captions'].append(caption)
    return vocab


def main(params):
    videos = json.load(open(params['input_json'], 'r'))['sentences']
    video_caption = {}
    for i in videos:
        if i['video_id'] not in video_caption.keys():
            video_caption[i['video_id']] = {'captions': []}
        video_caption[i['video_id']]['captions'].append(i['caption'])
    # create the vocab
    vocab = build_vocab(video_caption, params)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    out['videos'] = {'train': [], 'val': [], 'test': []}
    videos = json.load(open(params['input_json'], 'r'))['videos']
    for i in videos:
        out['videos'][i['split']].append(int(i['id']))

    # 加入对目录中测试数据集的写入
    pathPattern = os.path.join(params['test_video'], '*.mp4')
    print("video path: ", pathPattern)  # data/train-videos/TrainValVideo\*.mp4, 错误出现在路径合成上。
    # 要解决这个问题，就要在路径合成的前一个参数的末尾加上 ‘/’
    video_list = glob.glob(pathPattern)
    for video in tqdm(video_list):
        id = video.split("/")[-1].split('\\')[-1].split(".")[0][2:] # windows下需要额外去除反斜杠 由于之前生成的路径名称有反斜杠 
        out['videos']['test'].append(id)        # 不能转int，否则会丢弃前面的0

    json.dump(out, open(params['info_json'], 'w'))
    json.dump(video_caption, open(params['caption_json'], 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # test video
    parser.add_argument('--test_video', type=str, default='data/test_videos/TestVideo/', 
                        help='path of test videos')

    # input json
    parser.add_argument('--input_json', type=str, default='data/videoDataInfo.json',
                        help='msr_vtt videoinfo json')
    parser.add_argument('--info_json', default='data/info.json',
                        help='info about iw2word and word2ix')
    parser.add_argument('--caption_json', default='data/caption.json', help='caption json file')


    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
