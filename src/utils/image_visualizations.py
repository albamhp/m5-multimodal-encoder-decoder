import numpy as np
import matplotlib.pyplot as plt
from utils import MEAN_RGB, STD_RGB, MEAN_DEPTH, STD_DEPTH


def imshow_unimodal(inp, mean=MEAN_RGB, std=STD_RGB, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Visualize RGB-D images
def imshow_rgb_d(img_rgb, img_depth, title=None, concat_vert=False, mean_rgb=MEAN_RGB, mean_depth=MEAN_DEPTH,
                 std_rgb=STD_RGB, std_depth=STD_DEPTH, show_img=True):
    """Imshow for RGB-D data."""
    img_rgb = img_rgb.numpy().transpose((1, 2, 0))
    img_rgb = np.clip(np.array(std_rgb) * img_rgb + np.array(mean_rgb), 0, 1)
    img_depth = img_depth.numpy().transpose((1, 2, 0))
    img_depth = np.clip(np.array(std_depth) * img_depth + np.array(mean_depth), 0, 1)
    if concat_vert:
        img = np.concatenate((img_rgb, img_depth), axis=0)
    else:
        img = np.concatenate((img_rgb, img_depth), axis=1)

    if show_img:
        plt.imshow(img)
        if title is not None:
            plt.title(title)

        plt.axis('off')
        #        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()
    else:
        return img
