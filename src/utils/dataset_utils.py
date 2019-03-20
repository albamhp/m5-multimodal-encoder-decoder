import os
import os.path

import torch.utils.data as data
from torchvision.datasets.folder import default_loader

from utils import IMG_EXTENSIONS


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
        images (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):

        root_rgb = os.path.join(root, 'rgb')
        classes, class_to_idx = find_classes(
            root_rgb)  # Use RGB as reference. Depth/HHA must replicate same structure and file names

        self.images = make_dataset(root, class_to_idx)
        if len(self.images) == 0:
            raise (RuntimeError(
                "Found 0 images in subfolders of: " + root + "\n" + "Supported image extensions are: " + ",".join(
                    IMG_EXTENSIONS)))

        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        path_rgb, path_depth, target = self.images[index]
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
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
