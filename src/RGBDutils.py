import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt

from torch.autograd import Variable

import string
import os
import os.path
import numbers

from utils import IMG_EXTENSIONS, Compose


# Data loader for (RGB, HHA) pairs
# Based on torch.utils.data.ImageFolder


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(directory):
    #     print(dir)
    #     print(os.listdir(dir))
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(directory, class_to_idx):
    images = []
    directory = os.path.expanduser(directory)
    dir_rgb = os.path.join(directory, 'rgb')
    #     dirDepth = os.path.expanduser(dirDepth)
    for target in sorted(os.listdir(dir_rgb)):
        #         print(target)
        d_rgb = os.path.join(dir_rgb, target)
        #         d = os.path.join(directory, target)
        if not os.path.isdir(d_rgb):
            continue

        for root, _, f_names in sorted(os.walk(d_rgb)):
            #             print(root)

            for f_name in sorted(f_names):
                if is_image_file(f_name):
                    path_rgb = os.path.join(root, f_name)
                    path_depth = path_rgb.replace('/rgb/', '/hha/')
                    item = (path_rgb, path_depth, class_to_idx[target])
                    #                    item = (path_rgb, class_to_idx[target])
                    images.append(item)
    #                    print(type(images))

    return images


class ImageFolder(data.Dataset):
    """An RGB-D data loader where the images are arranged in this way: ::
        root/rgb/bedroom/xxx.png
        root/rgb/bedroom/xxy.png
        root/rgb/bedroom/xxz.png
        root/rgb/kitchen/123.png
        root/rgb/kitchen/nsdf3.png
        root/rgb/kitchen/asd932_.png
        ...
        root/hha/bedroom/xxx.png
        root/hha/bedroom/xxy.png
        root/hha/bedroom/xxz.png
        root/hha/kitchen/123.png
        root/hha/kitchen/nsdf3.png
        root/hha/kitchen/asd932_.png
        ...
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):

        rootRGB = os.path.join(root, 'rgb')
        classes, class_to_idx = find_classes(
            rootRGB)  # Use RGB as reference. Depth/HHA must replicate same structure and file names

        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        path_rgb, path_depth, target = self.imgs[index]
        img_rgb = self.loader(path_rgb)
        img_depth = self.loader(path_depth)

        if self.transform is not None:
            img_rgb, img_depth = self.transform(img_rgb, img_depth)
        #            img_pair  = self.transform(img_pair)
        if self.target_transform is not None:
            target = self.target_transform(target)

        #        return img_pair, target
        return img_rgb, img_depth, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


if __name__ == "__main__":
    # Test RGB-D data loader, transforms and utilities

    import torch
    import torchvision
    import os
    from utils import Compose, CenterCrop, ToTensor, Normalize, STD_RGB, MEAN_DEPTH, MEAN_RGB, STD_DEPTH, imshow_rgb_d

    data_dir = 'sunrgbd/256'
    data_dir = 'sunrgbd/256_lite'

    #    data_transforms = Compose([Resize(224),
    #                                   RandomHorizontalFlip(),
    #                                   ToTensor(),
    #                                   Normalize(MEAN_RGB, STD_RGB, MEAN_DEPTH, STD_DEPTH))])
    #    data_transforms = Compose([RandomResizedCrop(224),
    #                                   ToTensor(),
    #                                   Normalize(MEAN_RGB, STD_RGB, MEAN_DEPTH, STD_DEPTH)])
    data_transforms = Compose([CenterCrop(224),
                               ToTensor(),
                               Normalize(MEAN_RGB, STD_RGB, MEAN_DEPTH, STD_DEPTH)])

    rgbd_dataset = ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
    dataloader = torch.utils.data.DataLoader(rgbd_dataset, batch_size=4, shuffle=True, num_workers=4)
    class_names = rgbd_dataset.classes

    print(class_names)

    rgbd_iter = iter(dataloader)

    # Get a batch of training data
    imgs_rgb, imgs_depth, labels = next(rgbd_iter)

    # Make a grid from batch
    outRGB = torchvision.utils.make_grid(imgs_rgb)
    outDepth = torchvision.utils.make_grid(imgs_depth)
    #
    imshow_rgb_d(outRGB, outDepth, concat_vert=True, show_img=True)
#    imshow(imgs_rgb[0], imgs_depth[0], concat_vert=False)

# from torchvision import models
# inputs = torch.randn(1,3,224,224)
# resnet18 = models.resnet18()
# y = resnet18(Variable(inputs))
# print(y)
