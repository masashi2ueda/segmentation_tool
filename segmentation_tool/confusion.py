# %%
import numpy

from collections import OrderedDict
from enum import IntEnum, Enum
from typing import Dict, Optional


# %%
class TFPNVal(IntEnum):
    TP = 1
    FP = 2
    FN = 3
    TN = 4


class TFPNColor(Enum):
    TP = [1, 0, 0]  # red
    FP = [0, 1, 0]  # green
    FN = [0, 0, 1]  # blue
    TN = [0, 0, 0]  # black


# %%
def convert_TFPN(trues: numpy.ndarray, preds: numpy.ndarray) -> numpy.ndarray:
    """

    Args:
        trues (numpy.ndarray): true_labels(values is 0 or 1).
        preds (numpy.ndarray): pred_labels.(values is 0 or 1).

    Returns:
        numpy.ndarray: consution_values. shape is same to src.
            See TFPNVal.
    """
    tfpns = numpy.zeros_like(trues)
    tfpns[(trues == 1) & (preds == 1)] = TFPNVal.TP
    tfpns[(trues == 0) & (preds == 1)] = TFPNVal.FP
    tfpns[(trues == 1) & (preds == 0)] = TFPNVal.FN
    tfpns[(trues == 0) & (preds == 0)] = TFPNVal.TN
    return tfpns


def calc_confusion(trues: numpy.ndarray, preds: numpy.ndarray) -> Dict[str, int]:
    """

    Args:
        trues (numpy.ndarray): true_labels(values is 0 or 1).
        preds (numpy.ndarray): pred_labels.(values is 0 or 1).

    Returns:
        Dict[str, int]: count of {TP, FP, FN, TN}
    """
    tfpns = convert_TFPN(trues, preds).reshape(-1)
    confusion_dict = OrderedDict()

    for item in TFPNVal:
        confusion_dict[item.name] = len(numpy.where(tfpns == item.value)[0])
    return confusion_dict


def create_tfpn_image(trues: numpy.ndarray, preds: numpy.ndarray) -> numpy.ndarray:
    """

    Args:
        trues (numpy.ndarray): true_labels(values is 0 or 1).
        preds (numpy.ndarray): pred_labels.(values is 0 or 1).

    Returns:
        numpy.ndarray: image with color of {TP, FP, FN, TN}.
            see color detail: TFPNColor
    """
    h, w = trues.shape
    tfpn_img_hwc = numpy.zeros((h, w, 3))

    tfpns = convert_TFPN(trues, preds)

    for ci in range(3):
        for cd in TFPNColor:
            tfpn_img_hwc[..., ci][tfpns == TFPNVal[cd.name].value] = cd.value[ci]
    return tfpn_img_hwc


def draw_mask_enhancement(src_img_hwc: numpy.ndarray, mask_img_hw: numpy.ndarray, alpha: float) -> numpy.ndarray:
    """add red mask to image.

    Args:
        src_img_hwc (numpy.ndarray): original image.
        mask_img_hw (numpy.ndarray): mask image.
        alpha (float): Strength to put on the mask.0~1.

    Returns:
        numpy.ndarray: image with the mask part painted red.
    """
    img_red = numpy.zeros_like(src_img_hwc)
    img_red[:, :, 0] = 1

    show_img_hwc = src_img_hwc.copy()
    show_img_hwc[mask_img_hw == 1] = (img_red * alpha + (1 - alpha) * src_img_hwc)[mask_img_hw == 1]
    return show_img_hwc


def create_tfpn_merged_img(
        img_hwc: numpy.ndarray, prd_hw: numpy.ndarray,
        prd_mask_hw: numpy.ndarray,
        tru_mask_hw: Optional[numpy.ndarray] = None,
        alpha: float = 0.3) -> numpy.ndarray:
    """create big image with detail info.

    Args:
        img_hwc (numpy.ndarray): original image.
        prd_hw (numpy.ndarray): predicted mask probability.(0~1)
        prd_mask_hw (numpy.ndarray): predicted mask.(0 or 1)
        tru_mask_hw (Optional[numpy.ndarray], optional): true mask.(0 or 1). Defaults to None.
        alpha (float, optional): Strength to put on the mask.0~1.. Defaults to 0.3.

    Returns:
        numpy.ndarray: merged image.
            |img |prd     |img+prd_mask|
            |tfpn|tru_mask|img+tru_mask|
    """
    h , w, c = img_hwc.shape
    show_img_hwc = numpy.zeros((h*2, w*3, 3))

    if tru_mask_hw is not None:
        tfpn_image = create_tfpn_image(tru_mask_hw, prd_mask_hw)
    else:
        tfpn_image = 0

    def f1dto2d(src_hw):
        if src_hw is None:
            return 0
        src_hwc = numpy.expand_dims(src_hw, -1)
        return src_hwc.repeat(3, -1)

    def draw_img(wi, hi, src_img_hwc):
        show_img_hwc[hi*h:(hi+1)*h, wi*w:(wi+1)*w, :] = src_img_hwc

    draw_img(0, 0, img_hwc)
    draw_img(1, 0, f1dto2d(prd_hw))
    draw_img(2, 0, draw_mask_enhancement(img_hwc, prd_mask_hw, alpha))
    draw_img(0, 1, tfpn_image)
    draw_img(1, 1, f1dto2d(tru_mask_hw))
    draw_img(2, 1, draw_mask_enhancement(img_hwc, tru_mask_hw, alpha))
    return show_img_hwc

