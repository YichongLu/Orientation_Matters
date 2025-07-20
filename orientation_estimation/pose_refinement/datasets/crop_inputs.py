import os
from PIL import Image
import numpy as np
import rembg
import kiui
from kiui.op import recenter

path = "/data2/yclu/foundationpose/inputs/gso_30/alarm/rgb/alarm.png"

bg_remover = rembg.new_session()
input_image = Image.open(path)
input_image = np.array(input_image, dtype=np.uint8)[:, :, :3]

# bg removal
carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
mask = carved_image[..., -1] > 0

# recenter
image = recenter(carved_image, mask, border_ratio=0)

# save
Image.fromarray(image).save('./test/alarm_crop.png')
