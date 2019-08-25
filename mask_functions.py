import numpy as np
import torch

def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    img = img.detach().cpu().numpy()


    for y in range(height):
        for x in range(width):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1
        if lastColor == 255:
            rle.append(str(runStart))
            rle.append(str(runLength))

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    if rle != '-1':
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 255
            current_position += lengths[index]

    return mask.reshape(width, height)

def zero_out_the_small_regions(mask_batch, area_threshold=0.002):
    above_threshold = mask_batch.sum(dim=(1, 2, 3), keepdim=True) > area_threshold
    mask_batch *= above_threshold.type(dtype=mask_batch.type())
    return mask_batch

