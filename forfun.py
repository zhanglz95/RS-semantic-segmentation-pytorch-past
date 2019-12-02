import torch.utils.data as data
from collections import namedtuple
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

path = Path("/home/jjq/baidunetdiskdownload/DOTA/train/images/part1/images")
Pair = namedtuple("Pair", ['image','mask'])

def myCollectFun(image, mask):
    return Pair(image, mask)

transforms1 = T.Compose([
    T.Resize(600),
    T.CenterCrop(224),
    T.ToTensor()

])
class myDataSet(data.Dataset):
    def __init__(self, root, transforms=None):
        self.files = list(root.glob("*.png"))
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.files[index]
        image = Image.open(img_path)
        image = image.convert('RGB')
        if self.transforms:
            image, mask = self.transforms(image), self.transforms(image)
        return Pair(image, mask)

    def __len__(self):
        return len(self.files)


class myDataLoader(data.DataLoader):
    pass

if __name__ == "__main__":
    dataset = myDataSet(path, transforms1)
    dataloader = data.DataLoader(dataset, batch_size = 4, shuffle=True)
    # __getitem__ in dataset return a tuple of image and mask, 
    #   which means that next(iter(dataloader)) returns a tuple of a batch image and a batch mask, 
    #   but a iterable container consists of pairs with single image and mask.
    dataiter = iter(dataloader)
    data = next(dataiter)
    # next return a pair wihch the size of its image and mask is :
    #   batch_size x H x W x C but not a Iterable object filled by pairs which ammount is batch_size. 
    #   Thus the use of namedtuple does not change any developer's habits.
    # But also confused me about if the input of the net is fixed how the BatchSize x H x W x C tensor is plugged into the net ？ 
    #   Maybe it is like: Input->(I1...In)->matrix operation->cost(I1...In)(->sum)->loss->mini_batch gradient descent->just parameters.
    print(type(data))
    print(data.image.size())
    #print(type(imgs)))
    # for image, mask in dataloader:
    #    print(f"{image.size} || {mask.size}")
