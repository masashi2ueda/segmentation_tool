# %%
import numpy as np
import matplotlib.pyplot as plt

from IPython import display

import sys
sys.path.append("..")
import segmentation_tool
# import segmentation_tool

np.random.seed(0)

# %%
# config
t = 3
h = 100
w = 200

# src video and true masks
src_img_thwc = np.random.rand(t*h*w*3).reshape(t, h, w, 3)
tru_mask_thw = np.zeros((t, h, w))
tru_mask_thw[:, :, :50] = 1

# predict mask probablity from src video
prd_prob_thw = np.zeros((t, h, w))
prd_prob_thw[:, :50, :] = 1

# probability to mask
prd_mask_thw = segmentation_tool.prob2mask(prd_prob_thw, th=0.5)

# %%
t = 0
src_img_hwc = src_img_thwc[0]
tru_mask_hw = tru_mask_thw[0]
prd_prob_hw = prd_prob_thw[0]
prd_mask_hw = prd_mask_thw[0]

# to TP/FP/FN/TN
print("convert_TFPN--")
tfpns_hw = segmentation_tool.convert_TFPN(tru_mask_hw, prd_mask_hw)
for item in segmentation_tool.TFPNVal:
    print(item.name, item.value)
print("tfpns_hw:", tfpns_hw)

# show TP/FP/FN/TN
print("create_tfpn_image--")
tfpns_imgage_hw = segmentation_tool.create_tfpn_image(tru_mask_hw, prd_mask_hw)
for item in segmentation_tool.TFPNColor:
    print(item.name, item.value)
plt.imshow(tfpns_imgage_hw)
plt.pause(0.1)
plt.close()

# confusion values
print("calc_confusion--")
conf_vals = segmentation_tool.calc_confusion(tru_mask_hw, prd_mask_hw)
print(conf_vals)


# %%
# play video
anim = segmentation_tool.create_anim(src_img_thwc, fig_size=(10, 5))
display.HTML(anim.to_jshtml())
# %%