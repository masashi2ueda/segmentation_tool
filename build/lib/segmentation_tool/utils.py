# %%
import numpy
import matplotlib.pyplot as plt

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
