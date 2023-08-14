# %%
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn.functional as F

from matplotlib import animation
from typing import Tuple


# %%
def prob2mask(prob: numpy.ndarray, th: float) -> numpy.ndarray:
    """probbality is converted to mask.

    Args:
        prob (numpy.ndarray): 1 probability.
        th (float): threshould. th<prob -> 1

    Returns:
        numpy.ndarray: mask(0 or 1). type is long.
    """
    mask = numpy.zeros_like(prob)

    mask[th < prob] = 1

    mask = mask.astype(numpy.int64)

    return mask


def create_anim(show_img_thwc: numpy.ndarray, fig_size: Tuple[int, int]) -> animation.FuncAnimation:
    """

    Args:
        show_img_thwc (numpy.ndarray): img to show.
        fig_size (Tuple[int, int]): figure size

    Returns:
        animation.FuncAnimation: animation instance

    Examples:
            >>> from IPython import display
            >>> import numpy
            >>> img_thwc = numpy.random.rand(5*50*50*3).reshape(5, 50, 50, 3)
            >>> anim = create_anim(img_thwc, (10, 10))
            >>> display.HTML(anim.to_jshtml())
    """
    fig = plt.figure(figsize=fig_size)
    im = plt.imshow(show_img_thwc[0, ...])

    def draw(i):
        im.set_array(show_img_thwc[i, ...])
        return [im]
    anim = animation.FuncAnimation(
        fig, draw, frames=show_img_thwc.shape[0], interval=500, blit=True
    )
    plt.close()
    # display.HTML(anim.to_jshtml())
    return anim


# %%
def create_shift_grid(h: int, w: int, ty: float, tx: float) -> torch.Tensor:
    """create grid for sample.

    Args:
        h (int): image height
        w (int): image width
        ty (float): translation for y(direction of height).
            ex:case +1, [1, 2, 3] -> [1, 1, 2]
        tx (float): translation for y(direction of width).
            ex:case +1, [1, 2, 3] -> [1, 1, 2]

    Returns:
        torch.Tensor: grid_hwc

    Examples:
        >>> grid_hwc = create_shift_grid(h=5, w=3, ty=1, tx=0)
        >>> print(grid_hwc[..., 0])
        tensor([[-0.6667,  0.0000,  0.6667],
                [-0.6667,  0.0000,  0.6667],
                [-0.6667,  0.0000,  0.6667],
                [-0.6667,  0.0000,  0.6667],
                [-0.6667,  0.0000,  0.6667]])
        >>> print(grid_hwc[..., 1])
        tensor([[-1.2000e+00, -1.2000e+00, -1.2000e+00],
                [-8.0000e-01, -8.0000e-01, -8.0000e-01],
                [-4.0000e-01, -4.0000e-01, -4.0000e-01],
                [ 2.9802e-08,  2.9802e-08,  2.9802e-08],
                [ 4.0000e-01,  4.0000e-01,  4.0000e-01]])
    """
    yy_hw, xx_hw = torch.meshgrid(torch.arange(h), torch.arange(w))
    xx_hw = -1 + 1 / w + 2 / w * xx_hw - 2 / w * tx
    yy_hw = -1 + 1 / h + 2 / h * yy_hw - 2 / h * ty
    grid_hwc = torch.stack([xx_hw, yy_hw], -1).float()
    return grid_hwc


def grid_shift_sample(
    img_chw__bchw: torch.Tensor,
    grid_hwc: torch.Tensor = None,
    ty: float = None, tx: float = None,
    mode: str = 'bilinear',
    padding_mode: str = 'border',
    align_corners: float = False
) -> torch.Tensor:
    """sample acording to grid.

    Args:
        img_chw__bchw (torch.Tensor): src_image. with or without batch channnel.
        grid_hwc (torch.Tensor, optional): grid. Defaults to None.
        ty (float): translation for y(direction of height).
            ex:case +1, [1, 2, 3] -> [1, 1, 2].
            Defaults to None.
        tx (float): translation for y(direction of width).
            ex:case +1, [1, 2, 3] -> [1, 1, 2].
            Defaults to None.
        mode (str, optional): 'bilinear' | 'nearest' | 'bicubic'. Defaults to 'bilinear'.
        padding_mode (str, optional): 'zeros' | 'border' | 'reflection'. Defaults to 'border'.
        align_corners (float, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: dst_img_chw or dst_img_bchw

    Examples:
        >>> w = 3
        >>> h = 5
        >>> src_img_chw = torch.arange(h*w).reshape(1, h, w).float()
        >>> print(src_img_chw)
        tensor([[[ 0.,  1.,  2.],
                [ 3.,  4.,  5.],
                [ 6.,  7.,  8.],
                [ 9., 10., 11.],
                [12., 13., 14.]]])
        >>> dst_img_chw = grid_shift_sample(src_img_chw, ty=1, tx=1)
        >>> print(dst_img_chw)
        tensor([[[ 0.,  0.,  1.],
                [ 0.,  0.,  1.],
                [ 3.,  3.,  4.],
                [ 6.,  6.,  7.],
                [ 9.,  9., 10.]]])
    """
    is_src_batch = len(img_chw__bchw.shape) == 4
    img_bchw = img_chw__bchw if is_src_batch else img_chw__bchw.unsqueeze(0)

    if grid_hwc is None:
        w = img_chw__bchw.shape[-1]
        h = img_chw__bchw.shape[-2]
        grid_hwc = create_shift_grid(h=h, w=w, ty=ty, tx=tx)

    dst_img_bchw = F.grid_sample(
        img_bchw, grid_hwc.unsqueeze(0),
        mode=mode, padding_mode=padding_mode,
        align_corners=align_corners)

    dst = dst_img_bchw if is_src_batch else dst_img_bchw[0, ...]
    return dst
