import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim
# from apex import amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import datareader
import hrnet
import losses
import lr_utils
import models
import utils
import validate
import kfold
import samplers


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler=None, summary_writer=None, print_freq=100):
    model.train()
    # if torch.cuda.device_count() > 1:
    #     print('Using', torch.cuda.device_count(), 'GPUs!')
    #     model_with_loss = nn.DataParallel(model_with_loss)

    # model_with_loss = model_with_loss.to(device)

    header = 'Epoch: [{}]'.format(epoch)
    print(header)

    # lr_scheduler = None
    # if epoch == 0:
    #     warmup_factor = 1. / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)
    #
    #     lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # pc = 0
    # tc = 0
    epoch_loss = 0.
    criterion = losses.LossCalculator()
    for i, (inputs, targets) in enumerate(tqdm(data_loader)):
        # if i == 5:
        #     break
        images = inputs['scan']
        if not isinstance(model, torch.nn.DataParallel):
            images = images.to(device, non_blocking=True)

        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        # pc += torch.sum(targets['mask']).cpu().detach()
        # tc += torch.numel(targets['mask'])

        # if i == 0:
        #     model = torch.jit.trace(model, torch.rand_like(images))

        # pred, loss_dict = model_with_loss(images, targets)
        preds = model(images)
        # weight = targets['mask'] * (1 - pc / tc) + (1 - targets['mask']) * (1 - (tc - pc) / tc)
        # loss_mask = torch.nn.functional.binary_cross_entropy(preds['mask'], targets['mask'], weight, reduction='mean', )
        # loss_mask = ce_plus_dice(preds['mask'], targets['mask'])

        # loss_mask = losses.correct_dice_loss(preds['mask'], targets['mask'])
        # loss_mask = torch.nn.functional.binary_cross_entropy(preds['mask'], targets['mask'], reduction='mean')
        # loss_class = torch.nn.functional.binary_cross_entropy(preds['class'], targets['class'], reduction='mean')

        # loss = loss_mask # + loss_class
        # loss = loss_mask
        loss_dict = criterion(preds, targets)
        loss_mask = loss_dict['mask_ce']
        loss_mask_dice = loss_dict['mask_dice']
        loss_class = loss_dict['class_ce']
        # loss = loss_mask + loss_class
        # loss = loss_dict['mask_ce']
        loss = loss_dict['total']
        # loss = loss_dict['class_ce']

        if isinstance(data_loader.batch_sampler, samplers.OnlineHardBatchSampler):
            batch_indieces = inputs['idx'].cpu().detach().numpy()
            preds_mask_binary = (preds['mask'] > 0.5).type(dtype=preds['mask'].type(), non_blocking=False)
            batch_dice_values = losses.siim_dice_metric(preds_mask_binary, targets['mask']).cpu().detach().numpy()
            # batch_losses = loss_dict['mask_dice'].cpu().detach().numpy()
            batch_losses = 1. - batch_dice_values
            data_loader.batch_sampler.set_batch_losses(zip(batch_indieces, batch_losses))

        epoch_loss += torch.mean(loss).item()


        if lr_scheduler is not None:
            new_lr = lr_scheduler.get_new_lr()
            lr_utils.set_lr(optimizer, new_lr)

        if optimizer is not None:
            optimizer.zero_grad()

        torch.mean(loss).backward()
        # # loss.backward() changed to:
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        if optimizer is not None:
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if optimizer is not None:
            lr = lr_utils.get_lr(optimizer)

        if i % print_freq == 0:
            print('loss', torch.mean(loss).item(), 'lr', lr)

            # scan = np.transpose(inputs['scan'][0].cpu().detach().numpy(), (1, 2, 0))
            # mask = np.transpose(preds['mask'][0].cpu().detach().numpy(), (1, 2, 0))
            # gt = np.transpose(targets['mask'][0].cpu().detach().numpy(), (1, 2, 0))
            #
            # threshold = 0.5
            # idx = mask[:, :] > threshold
            # mask[idx] = 1.0
            # idx = mask[:, :] < threshold
            # mask[idx] = 0.0
            #
            # # mask = np.expand_dims(mask, axis=-1)
            # zeros = np.zeros_like(mask)
            # ones = np.ones_like(mask)
            # # mask = np.concatenate([zeros, zeros, mask], axis=-1)
            # red = np.concatenate([zeros, zeros, ones], axis=-1)
            # blue = np.concatenate([ones, zeros, zeros], axis=-1)
            #
            # # scan_with_mask = (scan.astype(np.float32) + mask.astype(np.float32))  # .astype(np.uint8)
            # scan_with_mask_and_label = (scan.astype(np.float32) * (1 - mask.astype(np.float32))) * (
            #             1 - gt) + red.astype(
            #     np.float32) * mask.astype(np.float32) + blue.astype(np.float32) * gt.astype(np.float32)

            # cv2.imshow('scan', scan_with_mask_and_label)
            # cv2.waitKey(1)
            # summary_writer.add_image('scan_with_masks', np.transpose(scan_with_mask_and_label,(2,0,1)), global_step=i + epoch * len(data_loader))
            global_step = i + epoch * len(data_loader)
            summary_writer.add_scalar('loss_mask', torch.mean(loss_mask).item(), global_step=global_step)
            summary_writer.add_scalar('loss_mask_dice', torch.mean(loss_mask_dice).item(), global_step=global_step)
            summary_writer.add_scalar('loss_mask_boundary', torch.mean(loss_dict['mask_boundary']).item(), global_step=global_step)
            summary_writer.add_scalar('loss_class', torch.mean(loss_class).item(), global_step=global_step)
            summary_writer.add_scalar('loss', torch.mean(loss).item(), global_step=global_step)
            summary_writer.add_scalar('lr', lr, global_step=global_step)

            # for tag, value in [(tag, value) for (tag, value) in model.named_parameters() if value.requires_grad]:
            #     tag = tag.replace('.', '/')
            #     summary_writer.add_histogram(tag, value, global_step=global_step)
            #     summary_writer.add_histogram(tag + '/grad', value.grad, global_step=global_step)

    if optimizer is not None:
        optimizer.zero_grad()
    mean_epoch_loss = epoch_loss / i
    return {'loss': mean_epoch_loss}

if __name__ == '__main__':
    output_dir = 'logs/22_ResNetUNet-34_BCE-GN-Batch-1-Adam'
    utils.seed_everything(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model = models.UNet(6, 1)
    # model = models.MyResNetModel()
    model = models.ResNetUNet(n_classes=1)
    # model = hrnet.HighResolutionNet(out_channels=1)
    # model.init_weights()

    # model = models.HRNetWithClassifier()
    model = model.to(device)


    data_patallel_multiplier = max(1, torch.cuda.device_count())
    data_patallel_multiplier = 1
    print('data_parallel_multiplier =', data_patallel_multiplier)

    folds = kfold.KFold('data/folds.json')
    train_filenames, test_filenames = folds.get_fold_split(0)

    batch_size = 10

    train_dataset = datareader.SIIMDataset('data/dicom-images-train', 'data/train-rle.csv', ([512], [512]),
                                           augment=True, filenames_whitelist=train_filenames)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size * data_patallel_multiplier,
                                                   shuffle=True,
                                                   num_workers=1)

    val_dataset = datareader.SIIMDataset('data/dicom-images-train', 'data/train-rle.csv', ([512], [512]),
                                         filenames_whitelist=test_filenames)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size * data_patallel_multiplier,
                                                 shuffle=False,
                                                 num_workers=1)

    trainable_params = [param for param in model.parameters() if param.requires_grad]

    lr_scaling_coefficient = (1 / 16) * data_patallel_multiplier * batch_size
    max_lr = 2e-3 * lr_scaling_coefficient
    base_lr = 5e-5 * lr_scaling_coefficient

    optim = torch.optim.Adam(params=trainable_params, lr=base_lr, betas=(0.0, 0.9))
    # optim = torch.optim.SGD(params=trainable_params,
    #                         momentum=0.98,
    #                         nesterov=True,
    #                         lr=base_lr)

    criterion = losses.LossCalculator()
    criterion = losses.LossCalculatorLRFinderWrapper(loss_calculator=criterion)

    lr_finder = lr_utils.LRFinder(model, optim, criterion, device=device, memory_cache=False,
                                  input_extraction_fn=lambda x: x['scan'])
    lr_finder.range_test(train_loader=train_dataloader, end_lr=100, num_iter=1000)
    lr_finder.plot(skip_start=10, skip_end=5)
    exit(0)

    lr_scheduler = None
    initial_epoch = 0
    best_metric = 0.0
    loaded_epoch, loaded_best_metric = utils.try_load_checkpoint(output_dir, model, device, optimizer=optim,
                                                                 load_optimizer=True)
    if loaded_epoch is not None: initial_epoch = loaded_epoch
    if loaded_best_metric is not None: best_metric = loaded_best_metric

    epochs_per_cycle = 30
    lr_scheduler = lr_utils.CyclicalLR(max_lr=max_lr, base_lr=base_lr, steps_per_epoch=len(train_dataloader),
                                       epochs_per_cycle=epochs_per_cycle, mode='cosine')
    lr_scheduler.step_value = initial_epoch * len(train_dataloader)

    steps_per_epoch = len(train_dataloader)
    # torch.optim.lr_scheduler.CyclicLR(optimizer=optim, base_lr=base_lr, max_lr=max_lr, step_size_up=steps_per_epoch * 1,
    #                                   step_size_down=steps_per_epoch * 4, mode='triangular', gamma=1.0, scale_fn=None,
    #                                   scale_mode='cycle',
    #                                   cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
    #                                   last_epoch=-1)

    # model, optimizer = amp.initialize(model, optim, opt_level='O0')


    writer = SummaryWriter(output_dir)
    for epoch in range(initial_epoch, 999999999):
        train_one_epoch(model=model, optimizer=optim, data_loader=train_dataloader, device=device, epoch=epoch,
                        lr_scheduler=lr_scheduler, summary_writer=writer, print_freq=100)

        score = validate.validate(model, train_dataloader, device)
        writer.add_scalar('dice', score, global_step=epoch * len(train_dataloader))
        if score > best_metric:
            best_metric = score
            # if epoch % epochs_per_cycle == 0:
        utils.save_checkpoint(output_dir=output_dir, epoch=epoch, model=model, optimizer=optim, best_metric=best_metric)

        # Validate here
