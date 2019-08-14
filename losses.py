import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
from lovasz_losses import lovasz_hinge


# TODO: Also try Lovasz

class LossCalculatorLRFinderWrapper(nn.Module):
    def __init__(self, loss_calculator):
        super(LossCalculatorLRFinderWrapper, self).__init__()
        self.loss_calculator = loss_calculator

    def forward(self, input, target):
        return self.loss_calculator(input, target)['mask_ce']

class LossCalculator(nn.Module):
    def __init__(self):
        super(LossCalculator, self).__init__()
        # self.pc = 1024 * 1024
        # self.tc = 1024 * 1024
        # self.focal_loss = FocalLoss(gamma=2)
        # self.ohem = OhemCrossEntropy(thresh=0.7, min_kept=0.1)
        # self.ohem = LimitedLossOhemCrossEntropyPerExample(max_kept=0.02) # 0.02
        # self.ohem = LimitedLossOhemCrossEntropyPerExample(max_kept=0.3) # 0.02
        self.ohem = LimitedLossOhemCrossEntropy(max_kept=0.5) # 0.3
        # self.ohem = LimitedLossOhemCrossEntropy(max_kept=0.3)
        self.class_ohem = LimitedLossOhemCrossEntropy(max_kept=0.5)

    def forward(self, preds_dict, targets_dict):
        pred_mask = preds_dict['mask']
        pred_mask_logits = preds_dict['mask_logits']
        pred_class = preds_dict['class']
        pred_class_logits = preds_dict['class_logits']
        target_mask = targets_dict['mask']
        target_class = targets_dict['class']
        # weights = torch.tensor([0.031172769732495487, 0.9688272302675044], dtype=torch.float32, device=pred_mask.device)

        # mask_ce_loss = F.binary_cross_entropy(pred_mask, target_mask, reduction='none')
        # mask_ce_loss = torch.mean(mask_ce_loss, dim=(1, 2, 3))

        mask_ce_loss = self.ohem(pred_mask_logits, target_mask)
        mask_ce_loss *= target_class.view((target_class.size(0)))
        mask_ce_loss = mask_ce_loss.sum() / (target_class.sum() + 0.00001)

        # mask_dice_loss = torch.zeros_like(mask_ce_loss)
        # mask_dice_loss = lovasz_hinge(pred_mask_logits, target_mask, per_image=True)
        mask_dice_loss = correct_dice_loss(pred_mask, target_mask)
        mask_dice_loss *= target_class.view((target_class.size(0)))
        mask_dice_loss = mask_dice_loss.sum() / (target_class.sum() + 0.00001)

        # class_ce_loss = F.binary_cross_entropy(pred_class, target_class, reduction='none')
        # class_ce_loss = class_ce_loss.mean(dim=(1))
        class_ce_loss = self.class_ohem(pred_class_logits, target_class)#.mean()

        # mask_ce_loss = torch.zeros_like(class_ce_loss)

        total_loss = mask_ce_loss + mask_dice_loss + class_ce_loss
        return {'mask_ce': mask_ce_loss, 'mask_dice': mask_dice_loss, 'mask_total': mask_ce_loss + mask_dice_loss,
                'class_ce': class_ce_loss, 'class_total': class_ce_loss,
                'total': total_loss}


class OhemCrossEntropy(nn.Module):
    def __init__(self, thresh=0.7, min_kept=0.5, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thresh
        self.min_kept = max(0.001, min_kept)
        self.criterion = nn.BCELoss(weight=weight, reduction='none')

    def forward(self, pred, target, **kwargs):
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
        pixel_losses = self.criterion(pred, target).contiguous().view(-1)
        tmp_target = target.clone().to(torch.long)
        # pred = pred.gather(1, tmp_target)
        pred, ind = pred.contiguous().view(-1, ).contiguous().sort()
        min_kept = int(self.min_kept * (pred.numel() - 1))
        min_value = pred[min(min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

class LimitedOhemCrossEntropy(nn.Module):
    def __init__(self, max_kept=0.001, weight=None):
        super(LimitedOhemCrossEntropy, self).__init__()
        self.min_kept = 1
        self.criterion = nn.BCELoss(weight=weight, reduction='none')

    def forward(self, pred, target, **kwargs):
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
        pixel_losses = self.criterion(pred, target).contiguous().view(-1)
        # tmp_target = target.clone().to(torch.long)
        # pred = pred.gather(1, tmp_target)
        pred, ind = pred.contiguous().view(-1, ).contiguous().sort()

        # TODO: We should cut of max_elements from tensor so
        threshold = pred[min(int(self.max_kept * pred.numel()), pred.numel() - 1)]
        pixel_losses = pixel_losses[ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()


# class LimitedLossOhemCrossEntropy(nn.Module):
#     # In this implementation of OHEM we are taking max_kept percentage of pixels with highest losses
#     def __init__(self, max_kept=0.001, weight=None):
#         super(LimitedLossOhemCrossEntropy, self).__init__()
#         self.min_kept = 1
#         self.max_kept = max_kept
#         self.criterion = nn.BCELoss(weight=weight, reduction='none')
#
#     def forward(self, pred, target, **kwargs):
#         # ph, pw = pred.size(2), pred.size(3)
#         # h, w = target.size(2), target.size(3)
#         # if ph != h or pw != w:
#         #     pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
#         pixel_losses = self.criterion(pred, target).contiguous().view(-1)
#         pixel_losses, ind = pixel_losses.contiguous().view(-1, ).contiguous().sort(descending=True)
#
#         # TODO: We should cut of max_elements from tensor so
#         threshold = pixel_losses[min(int(self.max_kept * pred.numel()), pred.numel() - 1)]
#         pixel_losses = pixel_losses[pixel_losses > threshold]
#         return pixel_losses.mean()

class LimitedLossOhemCrossEntropy(nn.Module):
    # In this implementation of OHEM we are taking max_kept percentage of pixels with highest losses
    def __init__(self, max_kept=0.001, weight=None):
        super(LimitedLossOhemCrossEntropy, self).__init__()
        self.min_kept = 1
        self.max_kept = max_kept
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction='none')

    def forward(self, logits, target, **kwargs):
        # ph, pw = pred.size(2), pred.size(3)
        # h, w = target.size(2), target.size(3)
        # if ph != h or pw != w:
        #     pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
        pixel_losses = self.criterion(logits, target)
        pl_flat = pixel_losses.contiguous().view(-1)
        pl_flat, ind = pl_flat.contiguous().view(-1, ).contiguous().sort(descending=True)

        # TODO: We should cut of max_elements from tensor so
        threshold = pl_flat[min(int(self.max_kept * logits.numel()), logits.numel() - 1)]
        pixel_losses = pixel_losses.contiguous().view((pixel_losses.size(0), -1))
        hard_mask = (pixel_losses > threshold).type(pixel_losses.type())
        pixel_losses = pixel_losses * hard_mask
        if pixel_losses.size(1) > 1:
            # mask
            pixel_losses = pixel_losses.sum(dim=1) / hard_mask.sum(dim=1)
        else:
            pixel_losses = pixel_losses.sum(dim=1)
        return pixel_losses

class LimitedLossOhemCrossEntropyPerExample(nn.Module):
    # In this implementation of OHEM we are taking max_kept percentage of pixels with highest losses
    def __init__(self, max_kept=0.02, weight=None):
        super(LimitedLossOhemCrossEntropyPerExample, self).__init__()
        self.min_kept = 1
        self.max_kept = max_kept
        self.criterion = nn.BCELoss(weight=weight, reduction='none')

    def forward(self, pred, target, **kwargs):
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
        pixel_losses = self.criterion(pred, target).contiguous()
        pixel_losses, ind = pixel_losses.contiguous().view(pixel_losses.size(0), -1,)\
            .contiguous().sort(dim=-1, descending=True)

        num_el_per_sample = pred.size(1) * pred.size(2) * pred.size(3)
        # print(pixel_losses.size())
        thresholds = pixel_losses[:, min(int(self.max_kept * num_el_per_sample), num_el_per_sample - 1)]
        thresholds = torch.unsqueeze(thresholds, dim=1)
        # print(thresholds)
        hard_mask = (pixel_losses > thresholds).type(pixel_losses.type())
        pixel_losses = pixel_losses * hard_mask
        pixel_losses = pixel_losses.sum(dim=1) / hard_mask.sum(dim=1)
        return pixel_losses

class AdaptiveOhemCrossEntropyPerExample(nn.Module):
    # In this implementation of OHEM we are taking max_kept percentage of pixels with highest losses
    def __init__(self):
        super(AdaptiveOhemCrossEntropyPerExample, self).__init__()
        self.min_kept = 1
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, pred, target, **kwargs):
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
        pixel_losses = self.criterion(pred, target).contiguous()

        thresholds = pixel_losses.mean(dim=(1, 2, 3), keepdim=True)
        # thresholds = torch.unsqueeze(thresholds, dim=1)
        # print(thresholds)
        hard_mask = (pixel_losses > thresholds).type(pixel_losses.type())
        pixel_losses = pixel_losses * hard_mask
        pixel_losses = pixel_losses.sum(dim=(1, 2, 3)) / hard_mask.sum(dim=(1, 2, 3))
        return pixel_losses

class AdaptiveOhemCrossEntropy(nn.Module):
    # In this implementation of OHEM we are taking max_kept percentage of pixels with highest losses
    def __init__(self):
        super(AdaptiveOhemCrossEntropy, self).__init__()
        self.min_kept = 1
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, pred, target, **kwargs):
        # ph, pw = pred.size(2), pred.size(3)
        # h, w = target.size(2), target.size(3)
        # if ph != h or pw != w:
        #     pred = F.upsample(input=pred, size=(h, w), mode='bilinear')
        pixel_losses = self.criterion(pred, target).contiguous()

        threshold = pixel_losses.mean()
        # thresholds = torch.unsqueeze(thresholds, dim=1)
        # print(thresholds)
        hard_mask = (pixel_losses > threshold).type(pixel_losses.type())
        pixel_losses = pixel_losses * hard_mask
        pixel_losses = pixel_losses.sum(dim=(1, 2, 3)) / hard_mask.sum(dim=(1, 2, 3))
        return pixel_losses


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def correct_dice_loss(input, target):
    smooth = 1.

    intersection = (input * target).sum(dim=(1, 2, 3))
    cardinality = (input + target).sum(dim=(1, 2, 3))

    return 1 - ((2. * intersection + smooth) / (cardinality + smooth))


def siim_dice_metric(input, target):
    has_mask = (target.sum(dim=(1, 2, 3)) != 0).type(dtype=target.type(), non_blocking=False)
    is_empty = (input.sum(dim=(1, 2, 3)) == 0).type(dtype=input.type(), non_blocking=False)
    is_empty_correct = (is_empty == 1 - has_mask).type(dtype=input.type(), non_blocking=False)

    dice = dice_metric(input, target)
    result = dice * has_mask + ((1 - has_mask) * is_empty_correct)
    return result


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


def dice_metric(input, target):
    intersection = (input * target).sum(dim=(1, 2, 3))
    smooth = torch.ones_like(intersection) * (1 / torch.numel(input[0]))
    return (2. * intersection) / (input.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)


class TotalLoss(nn.Module):
    def __init__(self, criterion_dict):
        super(TotalLoss, self).__init__()
        self.criterion_dict = criterion_dict

    def forward(self, input_dict, target_dict):
        loss_dict = {}
        for key in input_dict.keys():
            if key not in self.criterion_dict.keys(): continue
            prediction = input_dict[key]
            target = target_dict[key]
            criterion = self.criterion_dict[key]
            loss_dict[key] = criterion(prediction, target)

        loss_dict['total'] = sum(loss for loss in loss_dict.values())
        return loss_dict
