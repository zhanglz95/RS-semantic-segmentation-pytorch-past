import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from collections import namedtuple
Pair = namedtuple('Pair', ['image', 'mask'])

def HorizontalFlip(pair, probability):
    '''
    Horizontal Flip image and mask.
    '''
    if random.random() > probability:
        return pair
    image = pair.image.transpose(Image.FLIP_TOP_BOTTOM)
    mask = pair.mask.transpose(Image.FLIP_TOP_BOTTOM)

    return Pair(image, mask)

def VerticalFlip(pair, probability):
    '''
    Vertical Flip image and mask.
    '''
    if random.random() > probability:
        return pair
    image = pair.image.transpose(Image.FLIP_LEFT_RIGHT)
    mask = pair.mask.transpose(Image.FLIP_LEFT_RIGHT)

    return Pair(image, mask)

def Scale(pair, size):
    '''
    Scale image and mask.
    parameters:
        size: (2-tuple), (width, height)
    '''
    image = pair.image.resize(size, Image.BILINEAR)
    mask = pair.mask.resize(size, Image.BILINEAR)
 
    return Pair(image, mask)

def Translation(pair, factor, probability):
    '''
    Translate image and mask by factor.
    parameters:
        factor: (float), must be in [0, 1].
    return:
        The original image's (width, height) * factor from the top-left corner.
    '''
    if random.random() > probability:
        return pair        
    image_shape = pair.image.size
    corner_x = int(image_shape[0] * factor)
    corner_y = int(image_shape[1] * factor)
    temp = np.asarray(pair.image)
    new_image = np.zeros(temp.shape)
    new_image[corner_x:, corner_y:,: ] = temp[corner_x:, corner_y:,:]
    
    temp = np.asarray(pair.mask)
    new_mask = np.zeros(temp.shape)
    new_mask[corner_x:, corner_y: ] = temp[corner_x:, corner_y:, ]

    image = Image.fromarray(np.uint8(new_image))
    mask = Image.fromarray(np.uint8(new_mask))

    return Pair(image = image,
                    mask = mask)


def Rotation(pair, probability):
    '''
    Rotate image and mask by angle.
    parameters:
        angle (int), must 
    '''
    if random.random() > probability:
        return pair
    angle = random.randint(1, 3) * 90
    image = pair.image.rotate(angle)
    mask = pair.mask.rotate(angle)

    return Pair(image, mask)


def Crop(pair, probability):
    '''
    Crop image and mask from corner to new size.

    '''
    if random.random() > probability:
        return pair
    image_shape = pair.image.size

    size0 = int(random.uniform(0.5, 1) * image_shape[0])
    size1 = int(random.uniform(0.5, 1) * image_shape[1])

    assert size0 > 0 and size0 <= image_shape[0]
    assert size1 > 0 and size1 <= image_shape[1]
    
    corner0 = random.randint(0, image_shape[0] - size0)
    corner1 = random.randint(0, image_shape[1] - size1)

    left, upper = corner0, corner1
    right, lower = corner0 + size0, corner1 + size1

    image = pair.image.crop((left, upper, right, lower))
    mask = pair.mask.crop((left, upper, right, lower))

    return Pair(image, mask)


# TEST
if __name__ == "__main__":
    from pathlib import Path
    from collections import namedtuple
    
    Pair = namedtuple('Pair', ['image', 'mask'])
    
    path = Path("./test.jpg")
    img = Image.open(path)
    pair = Pair(img, img)
    result = []
    print('*****\n', pair)
    
    hf = HorizontalFlip()
    vf = VerticalFlip()
    scale = Scale()
    trans = Translation()
    rotate = Rotation()
    crop = Crop()
    
    result.append(hf(pair))
    result.append(vf(pair))
    result.append(scale(pair, [800,1000]))
    result.append(trans(pair, 0.2))
    result.append(rotate(pair, 10))
    result.append(crop(pair, (0,0), (300,300)))

    print('*****\n',result,'*****\n')
    num = 1
    print(len(result))
    for pair in result:
        pair.image.show()