import os
from PIL import Image
import re
import shutil
from zarr_tiff import down_sampling_img
from PIL import Image, ImageOps
import numpy as np
import shutil


def downsample():
    test_img_dir = '/n/groups/htem/Segmentation/xg76/mipmap/mipmap'
    s3_dir = '/n/groups/htem/Segmentation/xg76/mipmap/mipmap/s3'
    s3_pics = os.listdir(s3_dir)
    os.makedirs(os.path.join(test_img_dir, 's3x2'), exist_ok=True)
    os.makedirs(os.path.join(test_img_dir, 's3x3'), exist_ok=True)
    os.makedirs(os.path.join(test_img_dir, 's3x4'), exist_ok=True)
    for picname in s3_pics:
        print(picname)
        img = Image.open(os.path.join(s3_dir, picname))
        img = ImageOps.grayscale(img)
        img = np.array(img)
        scale_list = [2,3,4]
        downsample_list = down_sampling_img(img, scale_list)
        for idx, img in enumerate(downsample_list):
            tile = Image.fromarray(img)
            scale = scale_list[idx]
            tile.save(os.path.join(test_img_dir, f's3x{scale}', f'{picname}_x{scale}.tiff'))


# def copy_file():
#     mipmap_dir = '/n/groups/htem/Segmentation/xg76/local_realignment/mipmap'
#     dest_dir = '/n/groups/htem/Segmentation/xg76/local_realignment/mipmap/s3'
#     mipmap_folder = os.listdir(mipmap_dir)
#     for mipmaplayer in mipmap_folder:
#         mipmap_files = os.listdir(os.path.join(mipmap_dir, mipmaplayer))
#         for mipmap in mipmap_files:
#             try:
#                 if '1000' in mipmaplayer:
#                     shutil.copyfile(os.path.join(mipmap_dir, mipmaplayer, mipmap), os.path.join(dest_dir, mipmap))
#                 else:
#                     shutil.copyfile(os.path.join(mipmap_dir, mipmaplayer, mipmap), os.path.join(dest_dir, mipmap + 'f'))
#             except:
#                 print(f'{mipmap} failed')


if __name__ == "__main__":
    downsample()