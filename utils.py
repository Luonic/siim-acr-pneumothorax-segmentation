import random
import os
import numpy as np
import torch
import torch.distributed as dist
import glob
import torch.backends.cudnn as cudnn

seed = 0


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except Exception as e:
        print('Failed to set cuda seed')


def set_cudnn_perf_param():
    # cudnn related setting
    cudnn.benchmark = True
    # cudnn.deterministic = True
    cudnn.enabled = True


# TODO: Itegrate this to build submission
def float2int_mask(image):
    image = image * 255
    image = torch.clamp(image, min=0, max=255)
    image = image.type(torch.uint8)
    return image


def save_checkpoint(output_dir, epoch, model, optimizer=None, best_metric=None, name=None):
    own_state = model.module if isinstance(model, torch.nn.DataParallel) else model
    os.makedirs(output_dir, exist_ok=True)
    save_dict = {'state_dict': own_state.state_dict(),
                 'epoch': epoch}

    if best_metric is not None:
        save_dict['best_metric'] = best_metric

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()

    if name is None:
        checkpoint_path = os.path.join(output_dir, 'epoch={}.pth.tar'.format(epoch))
    else:
        checkpoint_path = os.path.join(output_dir, 'swa.pth.tar')
    torch.save(save_dict, checkpoint_path)
    print('Saved checkpoint to', checkpoint_path)


def load_checkpoint(checkpoint_path, model, device, optimizer=None, load_optimizer=True):
    own_state = model.module if isinstance(model, torch.nn.DataParallel) else model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    own_state.load_state_dict(checkpoint['state_dict'])

    if 'best_metric' in checkpoint.keys():
        best_metric = checkpoint['best_metric']
    else:
        best_metric = None

    if 'epoch' in checkpoint.keys():
        epoch = checkpoint['epoch']
    else:
        epoch = None

    if optimizer is not None and load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Successfully loaded ', checkpoint_path)
    return epoch, best_metric


def load_checkpoint_exact_epoch(epoch, checkpoint_dir, model, device, optimizer=None, load_optimizer=True):
    own_state = model.module if isinstance(model, torch.nn.DataParallel) else model
    checkpoint_path = os.path.join(checkpoint_dir, 'epoch={}.pth.tar'.format(epoch))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    own_state.load_state_dict(checkpoint['state_dict'])

    if 'best_metric' in checkpoint.keys():
        best_metric = checkpoint['best_metric']
    else:
        best_metric = None

    if 'epoch' in checkpoint.keys():
        epoch = checkpoint['epoch']
    else:
        epoch = None

    if optimizer is not None and load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Successfully loaded ', checkpoint_path)
    return epoch, best_metric


def try_load_checkpoint(checkpoint_dir, model, device, optimizer=None, load_optimizer=True):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
    checkpoints = list(set(checkpoints) - set(glob.glob(os.path.join(checkpoint_dir, '*swa*.pth.tar'))))
    checkpoints.sort(reverse=True, key=lambda x: int(x.split('=')[1].split('.')[0]))

    if len(checkpoints) > 0:
        return load_checkpoint(checkpoints[0], model, device, optimizer, load_optimizer)
    else:
        print('Checkpoints not found in', checkpoint_dir)
        return None, None


class dataloader_image_extract_wrapper():
    def __init__(self, iterable):
        self.iterable = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterable)[0]['scan']


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
