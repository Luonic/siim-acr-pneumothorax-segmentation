import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import datareader
import kfold
import lr_utils
import models
import utils
import validate
from train import train_one_epoch
import samplers
import radam
import torchcontrib


def train_fold(fold_idx, work_dir, train_filenames, test_filenames, batch_sampler, epoch, epochs_to_train):
    os.makedirs(work_dir, exist_ok=True)
    fold_logger = kfold.FoldLogger(work_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    batch_size = 4

    # model = models.UNet(6, 1)
    # model = models.MyResNetModel()
    model = models.ResNetUNet(n_classes=1, upsample=True)
    # model = models.ResNetUNetPlusPlus(n_classes=1)
    # model = models.EfficientUNet(n_classes=1)

    # model = models.HRNetWithClassifier()

    model.to(device)
    model = torch.nn.DataParallel(model)
    # model.to(device)

    data_patallel_multiplier = max(1, torch.cuda.device_count())
    # data_patallel_multiplier = 1
    print('data_parallel_multiplier =', data_patallel_multiplier)

    img_size = 1024

    train_dataset = datareader.SIIMDataset('data/dicom-images-train', 'data/train-rle.csv', ([img_size], [img_size]),
                                           augment=True, filenames_whitelist=train_filenames)
    # if batch_sampler is None:
    #     batch_sampler = samplers.OnlineHardBatchSampler(train_dataset, batch_size * data_patallel_multiplier,
    #                                                    drop_last=False)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=os.cpu_count(),
    #                                                batch_sampler=batch_sampler)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size * data_patallel_multiplier,
                                                   shuffle=True,
                                                   num_workers=os.cpu_count())

    val_dataset = datareader.SIIMDataset('data/dicom-images-train', 'data/train-rle.csv', ([img_size], [img_size]),
                                         filenames_whitelist=test_filenames)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size * data_patallel_multiplier,
                                                 shuffle=False,
                                                 num_workers=os.cpu_count())

    trainable_params = [param for param in model.parameters() if param.requires_grad]

    lr_scaling_coefficient = (1 / 16) * data_patallel_multiplier * batch_size / 10
    # max_lr = 2e-3 * lr_scaling_coefficient
    # base_lr = 5e-5 * lr_scaling_coefficient

    # OHEM Limited loss works with that divided by 10
    max_lr = 2.5e-4 * lr_scaling_coefficient
    base_lr = 3.5e-5 * lr_scaling_coefficient

    # optim = torch.optim.Adam(params=trainable_params, lr=base_lr, betas=(0.0, 0.9))
    # optim = torch.optim.Adam(params=[
    #     {"params": backbone_parameters, "lr": base_lr},
    #     {"params": head_and_classifier_params, "lr": max_lr}], lr=base_lr)
    # optim = torch.optim.Adam(params=trainable_params, lr=max_lr)
    # optim = torch.optim.AdamW(params=trainable_params, lr=base_lr, weight_decay=0.00001)
    optim = radam.RAdam(params=trainable_params, lr=base_lr, weight_decay=0.0001)
    optim = torchcontrib.optim.SWA(optim)
    # optim = torch.optim.SGD(params=trainable_params,
    #                         momentum=0.98,
    #                         nesterov=True,
    #                         lr=base_lr)
    # optim = torch.optim.SGD(params=trainable_params,
    #                         momentum=0.9,
    #                         nesterov=True,
    #                         lr=base_lr)

    best_metric = 0.0
    _, loaded_best_metric = utils.try_load_checkpoint(work_dir, model, device, optimizer=optim,
                                                      load_optimizer=True)
    if loaded_best_metric is not None: best_metric = loaded_best_metric

    # Experiments show that it often is good to set stepsize equal to 2 − 10 times the number of iterations in an epoch.
    # For example, setting stepsize = 8 ∗ epoch with the CIFAR-10 training run(as shown in Figure 1) only gives slightly
    # better results than setting stepsize = 2 ∗ epoch. (https://arxiv.org/pdf/1506.01186.pdf)
    # cycle_len = 4 == stepsize = 2
    # in my implementation
    epochs_per_cycle = 20
    lr_scheduler = lr_utils.CyclicalLR(max_lr=max_lr, base_lr=base_lr, steps_per_epoch=len(train_dataloader),
                                       epochs_per_cycle=epochs_per_cycle, mode='cosine')
    lr_scheduler.step_value = epoch * len(train_dataloader)

    steps_per_epoch = len(train_dataloader)
    # torch.optim.lr_scheduler.CyclicLR(optimizer=optim, base_lr=base_lr, max_lr=max_lr, step_size_up=steps_per_epoch * 1,
    #                                   step_size_down=steps_per_epoch * 4, mode='triangular', gamma=1.0, scale_fn=None,
    #                                   scale_mode='cycle',
    #                                   cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
    #                                   last_epoch=-1)

    # model, optimizer = amp.initialize(model, optim, opt_level='O0')

    writer = SummaryWriter(work_dir)
    for i in range(epochs_to_train):
        train_result_dict = train_one_epoch(model=model, optimizer=optim, data_loader=train_dataloader, device=device,
                                            epoch=epoch, lr_scheduler=lr_scheduler, summary_writer=writer, print_freq=100)

        val_result_dict = validate.validate(model, val_dataloader, device)
        mask_thresh, mask_score = val_result_dict['best_mask_score']
        class_thresh, class_score = val_result_dict['best_class_score']
        global_step = epoch * len(train_dataloader)
        writer.add_scalar('dice', mask_score, global_step=global_step)
        writer.add_scalar('classification_accuracy', class_score, global_step=global_step)
        writer.add_scalar('mean_epoch_loss', train_result_dict['loss'], global_step=global_step)
        writer.add_scalar('epoch', epoch, global_step=global_step)

        # {'best_mask_score': best_mask_score, 'mean_mask_scores': mean_mask_scores,
        #  'best_class_score': best_class_score, 'mean_class_scores': mean_class_scores}
        log_data = {'score': val_result_dict['best_mask_score'][1],
                    'mask_threshold': val_result_dict['best_mask_score'][0],
                    'class_accuracy': val_result_dict['best_class_score'][1],
                    'class_thresold': val_result_dict['best_class_score'][0]}
        if (epoch + 1) % epochs_per_cycle == 0 and epoch != 0:
            print('Updating SWA running average')
            optim.update_swa()
        epoch += 1
        break

    # if mask_score > best_metric:
    #     best_metric = mask_score
        # if epoch % epochs_per_cycle == 0:
    fold_logger.log_epoch(epoch - 1, log_data)
    utils.save_checkpoint(output_dir=work_dir, epoch=epoch - 1, model=model, optimizer=optim, best_metric=best_metric)

    if (epoch) % epochs_per_cycle == 0 and epoch != 0:
        optim.swap_swa_sgd()
        print('Swapped SWA buffers')
        print('Updating BatchNorm statistics...')
        optim.bn_update(utils.dataloader_image_extract_wrapper(train_dataloader), model, device)
        print('Updated BatchNorm statistics')
        print('Validating SWA model...')
        val_result_dict = validate.validate(model, val_dataloader, device)
        log_data = {'score': val_result_dict['best_mask_score'][1],
                    'mask_threshold': val_result_dict['best_mask_score'][0],
                    'class_accuracy': val_result_dict['best_class_score'][1],
                    'class_thresold': val_result_dict['best_class_score'][0]}
        fold_logger.log_epoch('swa', log_data)
        print('Saved SWA model')
        utils.save_checkpoint(output_dir=work_dir, epoch=None, name='swa', model=model, optimizer=optim, best_metric=best_metric)

    return {'mask_score': mask_score,
            'class_score': class_score,
            'global_step': global_step,
            'batch_sampler': batch_sampler}


if __name__ == '__main__':
    utils.seed_everything(1)
    utils.set_cudnn_perf_param()
    # TODO: Train on 768x768 random crops of 1024x1024 to be able to capture pneumo features for classification
    # TODO: Finetune decoder for pretrained model a few epochs
    output_dir = 'logs/RAdam-b16-Intepolation-Dice'
    os.makedirs(output_dir, exist_ok=True)
    folds = kfold.KFold('data/folds.json')
    writer = SummaryWriter(output_dir)

    # TODO: Detect latest common epoch
    epoch = folds.get_last_common_epoch(output_dir)
    fold_batch_samplers = [None] * folds.n_splits
    epochs_to_train_per_run = 5
    for epoch in range(epoch, 999, epochs_to_train_per_run):
        mask_scores = []
        class_scores = []
        for fold_idx in range(folds.n_splits):
        # for fold_idx in range(1):
            print('Training fold {} out of {}'.format(fold_idx + 1, folds.n_splits))
            train_filenames, test_filenames = folds.get_fold_split(fold_idx)
            fold_train_result = train_fold(fold_idx=fold_idx,
                                           work_dir=os.path.join(output_dir, 'fold_{}'.format(fold_idx)),
                                           train_filenames=train_filenames,
                                           test_filenames=test_filenames,
                                           epoch=epoch,
                                           epochs_to_train=epochs_to_train_per_run,
                                           batch_sampler=fold_batch_samplers[fold_idx])
            # fold_batch_samplers[fold_idx] = fold_train_result['batch_sampler']
            mask_scores.append(fold_train_result['mask_score'])
            class_scores.append(fold_train_result['class_score'])
        mean_folds_mask_score = np.mean(np.array(mask_scores, dtype=np.float32))
        mean_folds_class_score = np.mean(np.array(class_scores, dtype=np.float32))
        writer.add_scalar('mean_folds_mask_score', mean_folds_mask_score,
                          global_step=fold_train_result['global_step'])
        writer.add_scalar('mean_folds_class_score', mean_folds_class_score,
                          global_step=fold_train_result['global_step'])
