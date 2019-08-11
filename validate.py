import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim
from apex import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datareader
import hrnet
import losses
import lr_utils
import models
import utils


def validate(model, dataloader, device):
    print('Validating ...')
    model.eval()
    model.to(device)

    dice_values = []
    for i, (inputs, targets) in enumerate(tqdm(dataloader)):
        images = inputs['scan']
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        pred = model(images)
        pred_mask = pred['mask'].cpu().detach().numpy()
        pred_class = pred['class'].cpu().detach().numpy()
        target_mask = targets['mask'].cpu().detach().numpy()
        target_class = targets['class'].cpu().detach().numpy()

        threshold = 0.2
        idx = pred_mask[:, :, :, :] > threshold
        pred_mask_binary = np.zeros_like(pred_mask)
        pred_mask_binary[idx] = 1.0
        pred_mask_binary = torch.from_numpy(pred_mask_binary)
        target_mask = torch.from_numpy(target_mask)
        batch_dice_values = losses.dice_metric(pred_mask_binary, target_mask).cpu().detach().numpy()
        for dice_value in batch_dice_values:
            dice_values.append(dice_value)

    mean_dice = np.mean(np.array(dice_values))
    print('Validation dice', mean_dice)
    return mean_dice
