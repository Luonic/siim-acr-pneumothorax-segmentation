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
    image *= 255
    image = torch.clamp(image, min=0, max=255)
    image = image.type(torch.uint8)
    return image


def save_checkpoint(output_dir, epoch, model, optimizer=None, best_metric=None):
    own_state = model.module if isinstance(model, torch.nn.DataParallel) else model
    os.makedirs(output_dir, exist_ok=True)
    save_dict = {'state_dict': own_state.state_dict(),
                 'epoch': epoch}

    if best_metric is not None:
        save_dict['best_metric'] = best_metric

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()

    checkpoint_path = os.path.join(output_dir, 'epoch={}.pth.tar'.format(epoch))
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
    checkpoints.sort(reverse=True, key=lambda x: int(x.split('=')[1].split('.')[0]))

    if len(checkpoints) > 0:
        return load_checkpoint(checkpoints[0], model, device, optimizer, load_optimizer)
    else:
        print('Checkpoints not found in', checkpoint_dir)
        return None, None


def load_threshold(fold_dir, epoch):
    # TODO: read here json config with specified config
    pass


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
