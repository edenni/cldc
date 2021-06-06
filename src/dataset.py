import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tfrecord.torch.dataset import MultiTFRecordDataset

import sys
sys.path.append('E:\\cldc\\thirdpards\\FMix')
from fmix import sample_mask, make_low_freq_image, binarise_mask

from config import cfg
from utils import *


class CassavaLeafDataset(Dataset):
    def __init__(self, df, transform, test=False):
        super().__init__()

        self.df = df.reset_index(drop=True).copy()
        self.test = test
        self.labels = self.df.label.values
        self.labels = np.eye(self.df['label'].max()+1, dtype=np.float32)[self.labels] # one-hot
        
        if cfg.data.do_smooth:
            eps = cfg.data.smooth_eps
            self.labels = (1 - eps) * self.labels + eps /  self.labels.shape[-1]

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.loc[index]
        img = get_image(cfg.data.data_dir+sample.image_id)
        if self.transform:
            img = self.transform(image=img)['image']
        target = self.labels[index]

        # fmix
        if cfg.data.do_smooth and cfg.data.do_fmix and not self.test and np.random.random() > cfg.data.fmix_prob:
            with torch.no_grad():
                lam = np.clip(np.random.beta(cfg.data.fmix_alpha, cfg.data.fmix_alpha), 0.6, 0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(cfg.data.decay_power, (cfg.data.img_size, cfg.data.img_size))
                mask = binarise_mask(mask, lam, (cfg.data.img_size, cfg.data.img_size), cfg.data.max_soft)

                fmix_idx = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = get_image(cfg.data.data_dir + self.df.iloc[fmix_idx]['image_id'])
                if self.transform:
                    fmix_img = self.transform(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                img = mask_torch * img+(1. - mask_torch) * fmix_img
                rate = mask.sum() / (cfg.data.img_size ** 2)
                target = rate * target + (1. - rate) * self.labels[fmix_idx]

        # cutmix
        if cfg.data.do_smooth and cfg.data.do_cutmix and not self.test and np.random.random() > cfg.data.cutmix_prob:
            with torch.no_grad():
                cmix_idx = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_image(cfg.data.data_dir + self.df.iloc[cmix_idx]['image_id'])
                if self.transform:
                    cmix_img = self.transform(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(cfg.data.cutmix_alpha, cfg.data.cutmix_alpha), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((cfg.data.img_size, cfg.data.img_size), lam)
                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]
                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (cfg.data.img_size * cfg.data.img_size))
                target = rate * target + (1. - rate) * self.labels[cmix_idx]

        return img.float(), target

def collate_fn(batch):
    # print(type(batch[0]['image']), batch[0]['image'].shape)
    images = [torch.tensor(b['image']) for b in batch]
    targets = [torch.tensor(b['target']) for b in batch]
    return torch.stack(images).view(len(batch), -1, cfg.data.img_size, cfg.data.img_size), torch.stack(targets)


class CassavaLeafDataModule(pl.LightningDataModule):
    def __init__(self, df=None, trn_idx=None, val_idx=None, from_tfrec=False, train_path=None, val_path=None):
        super().__init__()
        self.df = df
        self.trn_idx = trn_idx
        self.val_idx = val_idx
        self.from_tfrec = from_tfrec
        self.train_path = train_path
        self.val_path = val_path

    def setup(self, stage=None):
        if not self.from_tfrec:
            self.train_dataset = CassavaLeafDataset(
                df=self.df.loc[self.trn_idx], 
                transform=get_train_transform(),
                test=False
            )
            self.val_dataset = CassavaLeafDataset(
                df=self.df.loc[self.val_idx],
                transform=get_val_transform(),
                test=True
            )
        else:
            # read from tfrecord
            self.train_dataset = self._get_tfrecord(self.train_path)
            self.val_dataset = self._get_tfrecord(self.val_path)

    def _get_tfrecord(self, tfrecord_pattern):
        len_file = len(glob.glob(tfrecord_pattern.format('*')))
        splits = dict(zip(['{:02}'.format(i) for i in range(len_file)], [1/len_file]*len_file))
        description = {"image": "float", "target": "float"}
        return MultiTFRecordDataset(tfrecord_pattern, None, splits, description)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=cfg.data.batch_size, 
            shuffle=True if not self.from_tfrec else False, 
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_fn if self.from_tfrec else None)
     

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn if self.from_tfrec else None)

def get_train_transform():
    return A.Compose([
        A.ShiftScaleRotate(p=0.6),
        # A.CenterCrop(cfg.data.img_size, cfg.data.img_size, p=0.5),
        # fcrs
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ], p=0.5),
        # # blur
        # A.OneOf([
        #     A.MotionBlur(blur_limit=(7,21), p=0.8),
        #     A.GaussianBlur(blur_limit=(7,21), p=0.8),
        # ], p=0.5),
        # # noise
        # A.OneOf([
        #     A.GaussNoise(p=1.0),
        #     A.ISONoise(p=0.8),
        # ], p=0.5),
        # # color
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=60, val_shift_limit=40, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3, 0.3), p=1.0),
        ], p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        A.Cutout(16, max_h_size=16, max_w_size=16, p=0.8),
        A.Resize(cfg.data.img_size, cfg.data.img_size),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(cfg.data.img_size, cfg.data.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])