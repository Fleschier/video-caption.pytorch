import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TestVideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        super(TestVideoDataset, self).__init__()
        # self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        # self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = opt["feats_dir"]
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """

        while(not os.path.exists('data/feats/resnet152/G_%05d.npy' % (ix))): # 跳过不存在的文件名
            ix += 1
            # print('ix: ', ix)
        
        print('find file : G_%05d.npy' % (ix))

        fc_feat = []
        for dir in self.feats_dir:
            fc_feat.append(np.load(os.path.join(dir, 'G_%05d.npy' % (ix))))
        fc_feat = np.concatenate(fc_feat, axis=1)
        if self.with_c3d == 1:
            c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'G_%05d.npy' % (ix)))
            c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
            fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        #data['gts'] = torch.from_numpy(gts).long()
        data['video_ids'] = 'G_%05d.npy' % (ix)
        return data

    def __len__(self):
        return len(self.splits['test'])
