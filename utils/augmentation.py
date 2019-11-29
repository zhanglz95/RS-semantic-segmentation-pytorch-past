import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from collections import namedtuple
Pair = namedtuple('Pair', ['image', 'mask'])

class HorizontalFlip():
    '''
    Horizontal Flip image and mask.
    '''
    def __call__(self, pair):
        image = pair.image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = pair.mask.transpose(Image.FLIP_TOP_BOTTOM)
        return Pair(image, mask)

class VerticalFlip():
    '''
    Vertical Flip image and mask.
    '''
    def __call__(self, pair):
        image = pair.mask.transpose(Image.FLIP_LEFT_RIGHT)
        mask = pair.image.transpose(Image.FLIP_LEFT_RIGHT)
        return Pair(image, mask)

class Scale():
    '''
    Scale image and mask.
    parameters:
        size: (2-tuple), (width, height)
    '''
    def __call__(self, size, pair):
        image = pair.image.resize(size, Image.BILINEAR)
        mask = pair.mask.resize(size, Image.BILINEAR)
        return Pair(image, mask)

class Translation():
    '''
    Translate image and mask by factor.
    parameters:
        factor: (float), must be in [0, 1].
    return:
        The original image's (width, height) * factor from the top-left corner.
    '''
    def __call__(self, factor, pair):
        image_shape = pair.image.size
        corner_x = int(image_shape[0] * factor) 
        corner_y = int(image_shape[1] * factor)

        temp = np.asarray(pair.image)
        new_image = np.zeros(temp.shape)
        new_image[corner_x:, corner_y:,: ] = temp[corner_x:, corner_y:,:]
        
        temp = np.asarray(pair.mask)
        new_mask = np.zeros(temp.shape)
        new_mask[corner_x:, corner_y: ] = temp[corner_x:, corner_y:, ]
        
        return Pair(image = Image.fromarray(np.uint8(new_image)),
                        mask = Image.fromarray(np.uint8(new_mask)))


class Rotation():
    '''
    Rotate image and mask by angle.
    parameters:
        angle (int), must 
    '''
    def __call__(self, angle, pair):
        image = pair.image.rotate(angle)
        mask = pair.mask.rotate(angle)
        return Pair(image, mask)


class Crop():
    '''
    Crop image and mask from corner to new size.

    '''
    def __call__(self, corner, size, pair):
        image_shape = pair.image.size
        assert size[0] > 0 and size[0] < image_shape[0]
        assert size[1] > 0 and size[1] < image_shape[1]
        
        left, upper = corner
        right, lower = corner[0] + size[0], corner[1] + size[1]

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
    result.append(scale((800,1000), pair))
    result.append(trans(0.2, pair))
    result.append(rotate(10, pair))
    result.append(crop((0,0), (300,300), pair))

    print('*****\n',result,'*****\n')
    num = 1
    print(len(result))
    for pair in result:
        pair.image.show()