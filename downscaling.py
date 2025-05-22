import cv2
import numpy as np
import os

path = 'E:/Visual studio code/intel internship'
input = os.path.join(path, 'Flickr2K')
output_r = os.path.join(path, 'blurred_input')
output = os.path.join(path, 'Sharp')

scale_factor = 0.25

inter = cv2.INTER_CUBIC

os.makedirs(output_r, exist_ok=True)
os.makedirs(output, exist_ok=True)

for filename in os.listdir(input):
    if filename.lower().endswith(('.jpg', '.png')):
        img_path = os.path.join(input, filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        downscaled = cv2.resize(img, (new_w, new_h), interpolation=inter)

        upscaled = cv2.resize(downscaled, (w, h), interpolation=inter)

        input_save_path = os.path.join(output_r, filename)
        target_save_path = os.path.join(output, filename)

        cv2.imwrite(input_save_path, upscaled)
        cv2.imwrite(target_save_path, img)

print("Finished preprocessing.")
