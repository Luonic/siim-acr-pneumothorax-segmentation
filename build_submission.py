import csv
import os
import mask_functions
import torch
import models
import kfold
import datareader
import torch
import os
import utils
from tqdm import tqdm
import numpy as np

def extract_mask(prediction):
    return prediction['mask']

def extract_class(prediction):
    return prediction['class']

def binarize(tensor, threshold):
    return (tensor > threshold).type(tensor.type())

def predict_and_binarize(model, image, mask_thresh, class_thresh):
    pred = model(image)
    cls = extract_class(pred)
    cls_bin = binarize(cls, class_thresh)
    mask = extract_mask(pred)
    mask_bin = binarize(mask, mask_thresh)
    mask_bin *= cls_bin
    return mask_bin

def tta_horizontal_flip(model, image, mask_thresh, class_thresh):
    image_flipped = image.flip(dims=[3])
    mask_flipped = predict_and_binarize(model, image_flipped, mask_thresh, class_thresh)
    return mask_flipped.flip(dims=[3])


def TestTimeAugmentationPrediction(model, image, mask_thresh, class_thresh):
    predictions = []
    predictions.append(predict_and_binarize(model, image, mask_thresh, class_thresh))
    predictions.append(tta_horizontal_flip(model, image, mask_thresh, class_thresh))
    return predictions




class SubmissionWriter():
    def __init__(self, dst_path):
        super(SubmissionWriter, self).__init__()
        self.output_file = open(dst_path, 'w')
        self.writer = csv.DictWriter(self.output_file, fieldnames=['ImageId', 'EncodedPixels'],
                                     delimiter=',', quoting=csv.QUOTE_NONE)
        self.writer.writeheader()

    def write_mask(self, image_id, mask):
        # mask in chw
        mask = torch.squeeze(mask, dim=0)
        height = mask.size()[0]
        width = mask.size()[1]

        if mask.sum() > 0:
            rle_encoded_mask = mask_functions.mask2rle(mask, height=height, width=width)
        else:
            rle_encoded_mask = '-1'

        self.writer.writerow({'ImageId': image_id, 'EncodedPixels': rle_encoded_mask})
        # TODO: check correctness of rle encoding

    def finalize(self):
        self.output_file.close()


if __name__ == '__main__':
    output_dir = 'logs/RAdam-b16-Intepolation-Dice'
    folds = kfold.KFold('data/folds.json')
    # thresholds = load_thresholds()
    models_list = []
    for fold_idx in range(folds.n_splits):
        print('Loading fold {}'.format(fold_idx))
        fold_dir = os.path.join(output_dir, 'fold_{}'.format(fold_idx))
        fold_logger = kfold.FoldLogger(fold_dir)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # model = models.UNet(6, 1)
        # model = models.MyResNetModel()
        model = models.ResNetUNet(n_classes=1, upsample=True)
        # model = hrnet.HighResolutionNet(out_channels=1)
        # model.init_weights()

        # model = models.HRNetWithClassifier()
        model = model.to(device)

        best_epoch, best_epoch_data = fold_logger.get_best_epoch()
        print('best epoch:', best_epoch)
        utils.load_checkpoint_exact_epoch(best_epoch, fold_dir, model, device, optimizer=None)
        model.eval()

        models_list.append({'model': model,
                            'mask_threshold': best_epoch_data['mask_threshold'],
                            'class_threshold': best_epoch_data['class_thresold']})

    img_size = 1024
    test_dataset = datareader.SIIMDataset('data/dicom-images-test', None, hw_size=([img_size], [img_size]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())

    dst_submission_path = os.path.join(output_dir, 'submission.csv')
    sub_writer = SubmissionWriter(dst_submission_path)

    mean_class_threshold = torch.from_numpy(
        np.mean(np.asarray([model['class_threshold'] for model in models_list], dtype=np.float32), keepdims=True)).to(
        device)

    mean_mask_threshold = torch.from_numpy(
        np.mean(np.asarray([model['mask_threshold'] for model in models_list], dtype=np.float32), keepdims=True)).to(
        device)

    with torch.no_grad():
        for input in tqdm(test_dataloader):
            predictions = []
            classes = []

            for model in models_list:
                img = input['scan']
                img = img.to(device)
                preds_dict = model['model'](img)

                pred_class = (preds_dict['class'] > model['class_threshold']).type(preds_dict['mask'].type())
                pred_mask = (preds_dict['mask'] > model['mask_threshold']).type(preds_dict['mask'].type())
                pred_mask *= pred_class

                pred_mask = preds_dict['mask']
                pred_class = preds_dict['class']
                predictions.append(pred_mask)
                classes.append(pred_class)


            predictions = torch.cat(predictions, dim=1)
            predictions, indices = torch.median(predictions, dim=1, keepdim=True)

            # predictions = torch.cumprod(predictions, dim=1)[:,-2:-1]
            # predictions = torch.pow(predictions, exponent=1 / len(models_list))
            #
            # classes = torch.cat(classes, dim=1)
            # classes = torch.cumprod(classes, dim=1)[:, -2:-1]
            # classes = torch.pow(classes, exponent=1 / len(models_list))
            #
            # classes = (classes > mean_class_threshold).type(classes.type())
            # predictions = (predictions > mean_mask_threshold).type(predictions.type())
            #
            # predictions *= classes

            predictions = utils.float2int_mask(predictions)
            # TODO: Resize to support pred on smaller images
            sub_writer.write_mask(input['image_id'][0], predictions[0])

        sub_writer.finalize()
