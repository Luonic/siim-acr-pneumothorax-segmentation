import json
from glob import glob
import datareader
import random
import numpy as np
import itertools
from tqdm import tqdm
import os
import json
from collections import OrderedDict

class KFold():
    def __init__(self, path='data/folds.json'):
        with open(path, 'r') as json_f:
            self.folds = json.load(json_f)
            self.n_splits = len(self.folds)

    def get_fold_split(self, fold_id):
        test_filenames = self.folds[fold_id]
        train_filenames = list(set(itertools.chain.from_iterable(self.folds)) - set(test_filenames))
        return train_filenames, test_filenames

    def get_last_common_epoch(self, folds_dir):
        folds = glob(os.path.join(folds_dir, 'fold_*'))
        epochs_in_folds = {}
        for i, fold in enumerate(folds):
            epochs_in_folds[fold] = set()
            checkpoints = glob(os.path.join(fold, '*.pth.tar'))
            for checkpoint in checkpoints:
                filename = os.path.split(checkpoint)[-1]
                epoch_num = int(filename.split('=')[1].split('.')[0])
                epochs_in_folds[fold].add(epoch_num)

        if len(epochs_in_folds) == 0:
            print('No fold dirs found to get common epoch')
            return 0

        if len(list(epochs_in_folds.values())[0]) == 0:
            print('No checkpoints found to get common epoch')
            return 0

        common_epochs = list(set.intersection(*epochs_in_folds.values()))
        common_epochs.sort(reverse=True)
        if len(common_epochs) != 0:
            return common_epochs[0]
        else:
            print('Can\'t find common epoch')
            return 0


# Logger should be tied to fold
# Logger should store dicts with epoch as key and thresholds and val scores as values
# Logger should be able to set data for current (specified) epoch
# Logger should be able to read log and return values for highest score
class FoldLogger():
    def __init__(self, fold_dir):
        self.log_filename = 'log.json'
        self.log = OrderedDict()
        self.log_file_path = os.path.join(fold_dir, self.log_filename)
        try:
            self.read()
        except FileNotFoundError:
            # self.write()
            pass

    def read(self):
        with open(self.log_file_path, 'r') as log_f:
            self.log = json.load(log_f)

    def write(self):
        with open(self.log_file_path, 'w') as log_f:
            json.dump(self.log, log_f, indent=4)
            log_f.flush()

    def log_epoch(self, epoch, data):
        # epoch: int, training epoch idx
        # data: dict of values from validation and/or validation
        assert data is not None

        try:
            logged_data = self.log[str(epoch)]
            data = logged_data.update(data)
        except KeyError:
            pass

        self.log[str(epoch)] = data

        self.write()

    def get_best_epoch(self):
        if len(self.log) > 0:
            epoch, data = sorted(self.log.items(), key=lambda x: x[1]['score'], reverse=True)[0]
            epoch = int(epoch)
            return epoch, data
        else:
            return None, None


if __name__ == '__main__':
    n_splits = 5
    dst_json = 'data/folds.json'
    folds = []

    empty_masks_part_per_fold = 0.1

    for i in range(n_splits):
        folds.append([])

    dataset = datareader.SIIMDataset('data/dicom-images-train', 'data/train-rle.csv', ([768], [768]))

    rating = []

    for image_dict, target_dict in tqdm(dataset):
        mask = target_dict['mask'].numpy().astype(np.float32)
        area = np.sum(mask) / mask.size

        rating.append((image_dict['image_path'], area))

        # if len(rating) > 50:
        #     break

    rating.sort(key=lambda x: x[1], reverse=True)

    while len(rating) > 0:
        for fold in folds:
            if len(rating) > 0:
                fold.append(rating[0][0])
                print('Area', rating[0][1])
                rating.remove(rating[0])
            else:
                break

    for i, fold in enumerate(folds):
        print('Fold', i, 'size:', len(fold))

    with open(dst_json, 'w') as f:
        json.dump(folds, f, indent=4)