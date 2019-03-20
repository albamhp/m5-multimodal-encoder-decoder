
import torchvision
import torch
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
from utils import Compose, CenterCrop, ToTensor, Normalize, STD_RGB, MEAN_DEPTH, MEAN_RGB, STD_DEPTH, imshow_rgb_d, \
    ImageFolder

if __name__ == "__main__":
    # Test RGB-D data loader, transforms and utilities

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
    data_loader = DataLoader(rgbd_dataset, batch_size=4, shuffle=True, num_workers=4)
    class_names = rgbd_dataset.classes

    print(class_names)

    rgbd_iter = iter(data_loader)

    # Get a batch of training data
    imgs_rgb, imgs_depth, labels = next(rgbd_iter)

    # Make a grid from batch
    outRGB = torchvision.utils.make_grid(imgs_rgb)
    outDepth = torchvision.utils.make_grid(imgs_depth)
    #
    imshow_rgb_d(outRGB, outDepth, concat_vert=True, show_img=True)
    #    imshow(imgs_rgb[0], imgs_depth[0], concat_vert=False)

    """""
    inputs = torch.randn(1, 3, 224, 224)
    resnet18 = models.resnet18()
    y = resnet18(Variable(inputs))
    print(y)
    """
