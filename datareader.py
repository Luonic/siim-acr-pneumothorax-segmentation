import glob
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import pydicom as pdc
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, RandomGamma, OneOf,
    GridDistortion, OpticalDistortion, RandomSizedCrop, ShiftScaleRotate
)
from torchvision import transforms
from tqdm import tqdm

from mask_functions import rle2mask

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class SIIMDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_csv_path, hw_size, augment: object = False, filenames_whitelist: object = None) -> object:
        self.image_dir = image_dir
        self.height = hw_size[0]
        self.width = hw_size[1]
        self.default_height = 1024
        self.default_width = 1024
        self.image_info = defaultdict(dict)
        self.augment = augment
        self.augmentations = Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                RandomGamma(),
            ], p=0.3),
            OneOf([
                # ElasticTransform(alpha=max(self.height), sigma=max(self.height) * 0.05, alpha_affine=max(self.height) * 0.03),
                GridDistortion(),
                # OpticalDistortion(distort_limit=1.5, shift_limit=0.5),
            ], p=0.3),
            ShiftScaleRotate(shift_limit=float(self.default_height / 4 / self.default_height),
                             scale_limit=0.25,
                             rotate_limit=45,
                             interpolation=cv2.INTER_LINEAR,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=0.,
                             always_apply=False,
                             p=0.5),
        ], p=1)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.view_pos_lut = {'AP': 0,
                             'PA': 1}
        self.age_converter = lambda x: float(x) / 100
        self.sex_lut = {'M': 0,
                        'F': 1}

        print('Reading dataset')

        images = glob.glob(os.path.join(image_dir, '*/*/*.dcm'), recursive=True)
        print('Found', len(images), 'images')
        images = {os.path.splitext(os.path.basename(filepath))[0]: filepath for filepath in images}

        if mask_csv_path:
            self.dataframe = pd.read_csv(mask_csv_path)
            for index, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):

                # if row[" EncodedPixels"].strip() == '-1':
                #     continue

                image_id = row['ImageId']

                if image_id in images:

                    image_path = images[image_id]

                    if filenames_whitelist is not None:
                        if image_path not in filenames_whitelist: continue

                    self.image_info[image_id]["image_id"] = image_id
                    self.image_info[image_id]["image_path"] = image_path
                    if 'annotations' not in self.image_info[image_id].keys():
                        self.image_info[image_id]["annotations"] = []
                    self.image_info[image_id]["annotations"].append(row[" EncodedPixels"].strip())
            del self.dataframe
        else:
            for image_id, image_path in images.items():
                self.image_info[image_id]["image_id"] = image_id
                self.image_info[image_id]["image_path"] = image_path

        self.image_info = {i: value for i, value in enumerate(self.image_info.values())}

    def __getitem__(self, idx):
        try:
            info = self.image_info[idx]
        except KeyError:
            raise StopIteration()
        img_path = info["image_path"]
        ds = pdc.dcmread(img_path)
        img = ds.pixel_array
        img = Image.fromarray(img)
        width, height = img.size
        img = img.resize((self.default_width, self.default_height), resample=Image.BILINEAR)
        img = np.array(img)

        if 'annotations' in info.keys():
            mask = None
            has_pneumothorax = False
            for rle_mask in info['annotations']:
                tmp_mask = rle2mask(rle_mask, width, height)
                tmp_mask = Image.fromarray(tmp_mask.T)
                tmp_mask = tmp_mask.resize((self.default_width, self.default_height), resample=Image.BILINEAR)
                # tmp_mask = np.expand_dims(tmp_mask, axis=0)
                if mask is None:
                    mask = tmp_mask
                else:
                    mask = np.maximum(mask, tmp_mask)

                if rle_mask != '-1':
                    has_pneumothorax = True

                mask = np.array(mask)

            if self.augment:
                img, mask = self.augment_image_and_mask(img, mask)

            mask = mask / 255
            mask = torch.as_tensor(mask, dtype=torch.float32)
            mask.unsqueeze_(0)
            mask = self.resize_th_3d_image(mask, (self.get_max_height(), self.get_max_width()))

            target = {}
            target["mask"] = mask
            # target["class"] = torch.from_numpy(np.expand_dims(np.array(has_pneumothorax, dtype=np.float32), axis=0))
            target["class"] = torch.unsqueeze(mask.max() > 0, dim=-1).type(torch.float32)

        img = transforms.ToTensor()(img)

        img = self.resize_th_3d_image(img, (self.get_max_height(), self.get_max_width()))

        # 1C to 3C to support pretrained imagenet models
        img = img.repeat(3, 1, 1)
        # img = self.normalize(img)

        view_pos = ds.get_item((0x0018, 0x5101)).value
        try:
            view_pos = view_pos.decode("utf-8").strip()
        except AttributeError as e:
            pass
        view_pos = self.view_pos_lut[view_pos]

        sex = ds.get_item((0x0010, 0x0040)).value.decode("utf-8").strip()
        sex = self.sex_lut[sex]

        age = ds.get_item((0x0010, 0x1010)).value.decode("utf-8").strip()
        age = self.age_converter(age)

        # attributes = [float(view_pos), float(sex), float(age)]
        # attributes = torch.from_numpy(np.array(attributes, dtype=np.float32)).view(len(attributes), 1, 1)
        # attributes = attributes.repeat(1, img.shape[1], img.shape[2])

        # total_input = torch.cat([img, attributes], dim=0)

        input = {'scan': img,
                 # 'attributes': attributes,
                 # 'total_input': total_input,
                 'view_pos': view_pos,
                 'sex': sex,
                 'age': age,
                 'image_path': img_path,
                 'image_id': info['image_id'],
                 'idx': idx}

        if 'annotations' in info.keys():
            return input, target
        else:
            return input

    def __len__(self):
        return len(self.image_info)

    def get_image_from_dcm(self, path):
        ds = pdc.dcmread(path)
        return ds.pixel_array

    def get_height(self):
        return random.randint(min(self.height), max(self.height))

    def get_width(self):
        return random.randint(min(self.width), max(self.width))

    def get_max_height(self):
        return max(self.height)

    def get_max_width(self):
        return max(self.width)

    def augment_image_and_mask(self, image, mask):
        augmented = self.augmentations(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    def resize_th_3d_image(self, image, new_size):
        image.unsqueeze_(0)
        image = F.interpolate(image, new_size, mode='bilinear')
        image = torch.squeeze(image, dim=0)
        return image


if __name__ == '__main__':
    import cv2

    # train_dataset = SIIMDataset('data/dicom-images-train', 'data/train-rle.csv',
    #                             hw_size=[[1024], [1024]], augment=True)
    train_dataset = SIIMDataset('data/dicom-images-test', 'logs/65-Folds-Adam-b4-CustomUResNet34-BN-MaskOHEMBCEDice-ClassOHEMBCE-FullData-Res1024-Aug/submission.csv',
                                hw_size=[[1024], [1024]], augment=False)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16*4,
    #                                                shuffle=True,
    #                                                num_workers=os.cpu_count())
    # for item in train_dataloader:
    #     print('iteration')
    #
    # exit(0)

    pixels_per_class = [0., 0.]
    total_pixels = 0.

    for example in tqdm(train_dataset):
        scan = np.transpose(example[0]['scan'].numpy(), (1, 2, 0))
        mask = np.squeeze(np.transpose(example[1]['mask'].numpy(), (1, 2, 0)), axis=-1)
        print('minmax_scan', np.min(scan), np.max(scan))
        print('minmax_mask', np.min(mask), np.max(mask))
        gt = mask = np.expand_dims(mask, axis=-1)
        zeros = np.zeros_like(mask)
        mask = np.concatenate([zeros, zeros, mask], axis=-1)
        scan_with_mask = (scan.astype(np.float32) + mask.astype(np.float32))  # .astype(np.uint8)

        print('view_pos', example[0]['view_pos'])
        print('age', example[0]['age'])
        print('sex', example[0]['sex'])
        print('class', example[1]['class'])

        cv2.imshow('scan', scan_with_mask)
        # # cv2.imshow('attributes', np.transpose(example[0]['attributes'].numpy(), (1, 2, 0)))
        cv2.waitKey(0)

        pixels_per_class[0] += gt.size - np.sum(gt).item()
        pixels_per_class[1] += np.sum(gt).item()

        total_pixels += float(gt.size)

    weights = [1 / np.log(1.02 + pixels / total_pixels) for pixels in pixels_per_class]
    print(weights)
    weights = [weight / sum(weights) for weight in weights]
    print(weights)