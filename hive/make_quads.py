#! python

from PIL import Image
import tifffile
import numpy as np
import os
from skimage import io
import fnmatch
import mmcv


# Pull together all the different objects
# original image, cityscapes, default, skyline
# base_dir = "/home/david/model_60681598616c5f4f14e4dd31"
# base_dir = "/home/david/dashcam-test"
base_dir = "/home/david/dashcam-test_model-60c6e4df6cf3c8059f329164-video-60c6d3701a43b372fac0ce4e"
# base_dir = "/home/david/drone-test"
# base_dir = "/home/david/dashcam-test_model-611458dfda844835beb2faad"
# base_dir = "/home/david/dashcam-test_model-60681598616c5f4f14e4dd31"
# base_dir = "/home/david/dashcam-test_video-20210114_185727_EF"
image_dir = "texturing_frames"
# image_dir = "/home/david/git/mmsegmentation/data/hm/images/test"
# cityscapes_dir = "truth"
cityscapes_dir = "setr_pup"
# cityscapes_dir = "cityscapes"
default_dir = "default"
setr_pup = "setr_pup_a2d2"
# sky_dir = "sky"
# sky_suffix = "_with-sky-mask.jpg"
output_dir = "aggregate_setr_a2d2"
if not os.path.exists(os.path.join(base_dir, output_dir)):
    os.makedirs(os.path.join(base_dir, output_dir))

opacity = 0.5
count = 0
total = len(os.listdir(os.path.join(base_dir, image_dir)))
for imagename in os.listdir(os.path.join(base_dir, image_dir)):
    count += 1
    print("Image", count, "of", total, "named", imagename)
    basename = os.path.splitext(os.path.basename(imagename))[0]
    image = Image.open(os.path.join(base_dir, image_dir, imagename))
    width, height = image.size
    # display(image)

    cityscapes = Image.open(os.path.join(base_dir, cityscapes_dir, imagename))
    # cityscapesClass = mmcv.imread(os.path.join(base_dir, cityscapes_dir, imagename))
    # cityscapes = mmcv.imread(os.path.join(base_dir, image_dir, imagename))
    # default = cityscapes.copy()
    # cityscapes = cityscapes * (1 - opacity) + cityscapesClass * opacity
    # # Convert from BGR
    # cityscapes = cityscapes[..., ::-1]
    # cityscapes = Image.fromarray(cityscapes.astype(np.uint8))
    # # display(cityscapes)

    default = mmcv.imread(os.path.join(base_dir, image_dir, imagename))
    defaultClass = mmcv.imread(os.path.join(base_dir, default_dir, imagename))
    default = default * (1 - opacity) + defaultClass * opacity
    # Convert from BGR
    default = default[..., ::-1]
    default = Image.fromarray(default.astype(np.uint8))
    # display(default)
    setrPup = Image.open(os.path.join(base_dir, setr_pup, imagename))
    # display(setrPup)
    I = Image.new('RGB', (2 + width*2, 2+ height*2))
    I.paste(image,(0,0))
    I.paste(cityscapes,(2 + width, 0))
    I.paste(setrPup,(0, 2 + height))
    I.paste(default,(2 + width, 2 + height))
    I.save(os.path.join(base_dir, output_dir, imagename))

print("Done")
