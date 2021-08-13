#! python3
import fnmatch
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mmcv
import mmseg
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
import numpy as np
import os
from PIL import Image
import sys
import torch
os.environ['PATH'] += os.pathsep + '/usr/local/cuda/bin'
os.environ['CUDA_HOME'] = '/usr/local/cuda/lib64/'
# Set this if needed for debugging
#sys.stdout = open('/tmp/inference-file.log', 'w')

# Where the model info is
base_directory = os.path.expanduser("~/git/mmsegmentation")
model_type = "setr"
# model_name = "drone-model"
# model_name = "street-model"
model_name = "setr_pup_512x512_160k_b16_hm"
checkpoint = "latest.pth"

# Where to get and save output
# input_image_directory = "../data/hm/images/test"
# input_image_directory = os.path.expanduser("~/dashcam-test_model-60c6e4df6cf3c8059f329164-video-60c6d3701a43b372fac0ce4e/texturing_frames")
# input_image_directory = os.path.expanduser("~/dashcam-test_model-60681598616c5f4f14e4dd31/texturing_frames")
# input_image_directory = os.path.expanduser("~/dashcam-test_model-611458dfda844835beb2faad/texturing_frames")
input_image_directory = "demo"

# Either save or display
save_images = True
opacity = 0.5
output_image_directory = "output"
if not os.path.exists(input_image_directory):
  print("Input directory", input_image_directory, "missing")
  sys.exit(1)

if input_image_directory == output_image_directory:
  print("Input and output directories must be different")
  sys.exit(1)

if not os.path.exists(output_image_directory):
  print("Making directory", output_image_directory)
  os.makedirs(output_image_directory)

# Check Pytorch installation
print("Torch version", torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
print("MMSeg version", mmseg.__version__)


# TODO test
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).reshape((a.shape[0], a.shape[1], num_classes))


# build the model from a config file and a checkpoint file
config_file = os.path.join(base_directory, 'configs', model_type, model_name + '.py')
checkpoint_file = os.path.join(base_directory, 'work_dirs', model_name, checkpoint)
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
# model = init_segmentor(config_file, checkpoint_file, device='cpu')

# Add class info and palette
COLOR_DICT_m7 = {0: "Other_m7", 1: "Mobile_m7", 2: "Trees_m7", 3: "Ground_m7", 4: "Structure_m7", 5: "Building_m7", 6: "Water_m7"}
model.CLASSES = [val for key, val in COLOR_DICT_m7.items()]
print("Class map:", model.CLASSES)
# Make color palette
REDUCE_MAP_m7 = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (205, 133, 63), 4: (255, 255, 0), 5: (255, 255, 255), 6: (0, 0, 255)}
palette = [val for _, val in REDUCE_MAP_m7.items()]

count = 0
for filename in os.listdir(input_image_directory):
  if fnmatch.fnmatch(filename, '*.jpg') or fnmatch.fnmatch(filename, '*.png'):
    count += 1
    print("Image", count, "of", len(os.listdir(input_image_directory)), "named", filename)

    # result = model.inference_i  (image, [image_meta], rescale=True)
    img = mmcv.imread(os.path.join(input_image_directory, filename))
    result = inference_segmentor(model, img)

    # # TEST Tiff output
    # print("Result", result[0].shape)
    # output = np.multiply(one_hot(result[0], len(model.CLASSES)), 255)
    # print("Output", output.shape)
    # print("Image", img.shape)
    # shadow_layer = np.zeros((output.shape[0], output.shape[1])).reshape((output.shape[0], output.shape[1], 1))
    # cat_vals = np.concatenate((mmcv.bgr2rgb(img), output, shadow_layer), axis=-1).astype('uint8')
    # print("Cat", cat_vals.shape)
    # basename = os.path.splitext(filename)[0]
    # tifffile.imsave(os.path.join(output_image_directory, basename + ".tif"), cat_vals, planarconfig='CONTIG')
    # continue

    if save_images:
      print("Saving output as", os.path.join(output_image_directory, filename))
      img_out = model.show_result(img, result, palette=palette, show=False, opacity=opacity)
      seg_img = Image.fromarray(mmcv.bgr2rgb(img_out))
      seg_img.save(os.path.join(output_image_directory, filename))
    else:
      # show_result_pyplot(model, img, result, palette)
      # Show with legend
      plt.figure(figsize=(8, 6))
      plt.title(filename)
      # Create a patch for every class color
      patches = [mpatches.Patch(color=np.array(palette[i]) / 255., label=model.CLASSES[i]) for i in range(len(palette))]
      # Put those patches as legend-handles into the legend
      # Top position
      # plt.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., fontsize='large')
      # Bottom position
      plt.legend(handles=patches, bbox_to_anchor=(0., -0.2, 1., .102), loc='upper center', ncol=4, mode="expand", borderaxespad=0., fontsize='large')
      img_out = model.show_result(img, result, palette=palette, show=False, opacity=opacity)
      im = plt.imshow(mmcv.bgr2rgb(img_out))
      plt.show()


