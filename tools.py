
from typing import Tuple
import matplotlib.pyplot as plt
from imagenet_labels import IMAGENET_LABELS as LMap


def inet_name(l, tag=""):
    '''Fixing some ImageNetnames for the example'''
    name = LMap[l]
    name = name.split(',')[0]
    # "great" was added to white shark in order to scare people
    # ImageNet should remove it.
    name = name.replace('great','') 
    name = name.replace('suspension','susp. ')
    name = tag + name
    return name


def dit(x):
    '''Detach and put on cpu one or a list of tensors'''
    if isinstance(x, list):
        res = [xx.detach().cpu() for xx in x]
    else:
        res = x.detach().cpu()
    return res


#
#   A modification of visualization tools from robustness/tools/vis_tools
#       https://github.com/MadryLab/robustness
#


def get_axis(axarr, H:int, W:int, i:int, j:int):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax


def show_image_row(
    xlist:list,
    ylist:list=None,
    fontsize:int=12,
    size:Tuple[float]=(2.5, 2.5),
    tlist:list=None,
    filename:str=None,
):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)
            im = xlist[h][w]
            
            ax.imshow(im.permute(1, 2, 0))
            
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0:
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
    return fig


