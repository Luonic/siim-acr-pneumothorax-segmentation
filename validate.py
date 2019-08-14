import numpy as np
import torch
import torch.optim
from tqdm import tqdm
import datareader
import models
import os
import utils
from collections import OrderedDict
import mask_functions

import losses
import cv2

def validate(model, dataloader, device, show=False):
    with torch.no_grad():
        # TODO: Add loss as a validation result
        print('Validating ...')
        model.eval()
        model.to(device)

        mask_scores = {}
        mask_losses = []
        class_scores = {}
        class_losses = []
        thresholds = np.arange(start=0.05, stop=1.0, step=0.05, dtype=np.float32).tolist()
        criterion = losses.LossCalculator()
        for i, (inputs, targets) in enumerate(tqdm(dataloader)):
            # if i == 5:
            #     break

            images = inputs['scan']
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            preds = model(images)
            pred_mask = preds['mask']
            # pred_mask = preds['mask_logits']
            pred_class = preds['class']
            target_mask = targets['mask']
            target_class = targets['class']

            loss_dict = criterion(preds, targets)
            # print('dice loss:', loss_dict['mask_dice'])
            mask_losses.append(loss_dict['mask_total'])
            class_losses.append(loss_dict['class_total'])

            pred_class_binary = (pred_class > 0.5).type(dtype=pred_class.type(), non_blocking=False)
            pred_class_binary = pred_class_binary.view((pred_class_binary.size(0), 1, 1, 1))
            # print(pred_mask.size())
            # print(pred_class_binary.size())
            pred_mask *= pred_class_binary

            if show:
                zeros = torch.zeros_like(pred_mask)
                # threshold = 0.5
                # threshold = pred_mask.mean()
                # pred_mask = pred_mask_binary = (pred_mask > threshold).type(dtype=pred_mask.type(), non_blocking=False)
                mask = torch.cat([zeros, zeros, pred_mask], dim=1)
                gt = torch.cat([target_mask, zeros, zeros], dim=1)
                images = images * (1. - gt) + mask + gt
                print(torch.mean(pred_mask), torch.max(pred_mask))
                cv2.imshow('scan with mask', np.transpose(images[0].cpu().detach().numpy(), axes=(1, 2, 0)))
                cv2.waitKey(0)

            for threshold in thresholds:
                if threshold not in mask_scores.keys():
                    mask_scores[threshold] = []

                if threshold not in class_scores.keys():
                    class_scores[threshold] = []

                pred_mask_binary = (pred_mask > threshold).type(dtype=pred_mask.type(), non_blocking=False)
                # pred_mask_binary = mask_functions.zero_out_the_small_regions(pred_mask_binary, area_threshold=0.002)

                batch_dice_values = losses.siim_dice_metric(pred_mask_binary, target_mask).cpu().detach().numpy()
                for dice_value in batch_dice_values:
                    mask_scores[threshold].append(dice_value)

                pred_class_binary = (pred_class > threshold).type(dtype=pred_class.type(), non_blocking=False)
                batch_class_equals = (pred_class_binary == target_class).cpu().detach().numpy()
                for batch_class_equals_value in batch_class_equals:
                    class_scores[threshold].append(batch_class_equals_value)

        # TODO: Fix validation result with respect to correct classification threshold

        mean_mask_scores = OrderedDict()
        for threshold, scores_list in mask_scores.items():
            mean_mask_scores[threshold] = np.mean(scores_list).item()

        mean_mask_scores = sorted(mean_mask_scores.items(), key=lambda x: x[1], reverse=True)
        best_mask_score = mean_mask_scores[0]

        mean_class_scores = OrderedDict()
        for threshold, scores_list in class_scores.items():
            mean_class_scores[threshold] = np.mean(scores_list).item()
        mean_class_scores = sorted(mean_class_scores.items(), key=lambda x: x[1], reverse=True)
        best_class_score = mean_class_scores[0]

        print('mask score', best_mask_score[1], 'threshold', best_mask_score[0])
        print('class score', best_class_score[1], 'threshold', best_class_score[0])
        return {'best_mask_score': best_mask_score, 'mean_mask_scores': mean_mask_scores,
                'best_class_score': best_class_score, 'mean_class_scores': mean_class_scores}

if __name__ == '__main__':
    import kfold
    folds = kfold.KFold('data/folds.json')
    train_filenames, test_filenames = folds.get_fold_split(0)
    val_dataset = datareader.SIIMDataset('data/dicom-images-train', 'data/train-rle.csv', ([512], [512]),
                                           filenames_whitelist=test_filenames)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())

    model = models.ResNetUNet(n_classes=1)
    # model = models.HRNetWithClassifier()
    utils.try_load_checkpoint('logs/64-Folds-Adam-b16-CustomUResNet34-BN-MaskLovaszBatch-ClassOHEMBCE-FullData-512x512-Aug/fold_0', model,
                              device='cpu', load_optimizer=False)
    print(validate(model, val_dataloader, 'cpu', show=True)['best_mask_score'])