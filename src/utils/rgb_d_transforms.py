import collections

from torchvision.transforms import functional as f
import math
import random
from PIL import Image
import numbers

# RGB-D transforms


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_rgb, img_depth):
        for t in self.transforms:
            img_rgb, img_depth = t(img_rgb, img_depth)
        return img_rgb, img_depth

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, img_rgb, img_depth):
        return f.to_tensor(img_rgb), f.to_tensor(img_depth)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean_rgb, std_rgb, mean_depth, std_depth):
        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.mean_depth = mean_depth
        self.std_depth = std_depth

    def __call__(self, tensor_rgb, tensorDepth):
        return f.normalize(tensor_rgb, self.mean_rgb, self.std_rgb), f.normalize(tensorDepth, self.mean_depth,
                                                                                self.std_depth)

    def __repr__(self):
        return self.__class__.__name__ + '(mean_rgb={0}, std_rgb={1},mean_depth={2}, std_depth={3})'.format(
            self.mean_rgb, self.std_rgb, self.mean_depth, self.std_depth)


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_rgb, img_depth):
        return f.resize(img_rgb, self.size, self.interpolation), f.resize(img_depth, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_rgb, img_depth):
        return f.center_crop(img_rgb, self.size), f.center_crop(img_depth, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_rgb, img_depth):
        if self.padding > 0:
            img_rgb = f.pad(img_rgb, self.padding)
            img_depth = f.pad(img_depth, self.padding)

        i, j, h, w = self.get_params(img_rgb, self.size)

        return f.crop(img_rgb, i, j, h, w), f.crop(img_depth, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_rgb, img_depth):
        if random.random() < self.p:
            return f.hflip(img_rgb), f.hflip(img_depth)
        return img_rgb, img_depth

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_rgb, img_depth):
        if random.random() < self.p:
            return f.vflip(img_rgb), f.vflip(img_depth)
        return img_rgb, img_depth

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img_rgb, img_depth):
        i, j, h, w = self.get_params(img_rgb, self.scale, self.ratio)
        return f.resized_crop(img_rgb, i, j, h, w, self.size, self.interpolation), f.resized_crop(img_depth, i, j, h, w,
                                                                                                  self.size,
                                                                                                  self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(round(self.scale, 4))
        format_string += ', ratio={0}'.format(round(self.ratio, 4))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
