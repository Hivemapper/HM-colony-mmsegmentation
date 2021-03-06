#! python3
import fnmatch
import tifffile
import mmcv
from mmcv import Config
import mmseg
from mmseg.apis import init_segmentor, inference_segmentor
import numpy as np
from optparse import OptionParser
import os
import sys
import torch
import torch.nn.functional as F
os.environ['PATH'] += os.pathsep + '/usr/local/cuda/bin'
os.environ['CUDA_HOME'] = '/usr/local/cuda/lib64/'
# Set this if needed for debugging
#sys.stdout = open('/tmp/inference-file.log', 'w')

# Check Pytorch installation
print("Torch version", torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
print("MMSeg version", mmseg.__version__)


class ArgumentException(Exception):
  pass


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).reshape((a.shape[0], a.shape[1], num_classes))


def main() -> None:
  python2 = sys.version_info[0] < 3
  python3 = sys.version_info[0] == 3
  if python2 or (python3 and sys.version_info[1] < 5):
    raise Exception("Python 3.5 or later is required.")

  po = OptionParser()
  po.add_option('-i', '--image', dest='image_input', help=f'Image Required: Name of file with relative paths to images to process')
  po.add_option('-r', '--results', dest='image_output', help=f'Results Required: Name of file with relative paths to desired output tif image results')
  po.add_option('-s', '--scale', dest='image_scale', default=100.0, help=f'Scale output image percentage--default=100.0')
  po.add_option('-t', '--type', dest='model_type', default='setr', help=f'Model type--default=setr')
  po.add_option('-m', '--model', dest='model_name', default='setr_pup_512x512_160k_b16_hm', help=f'The model name to run--default=setr_pup_512x512_160k_b16_hm')
  po.add_option('-c', '--checkpoint', dest='checkpoint', default='latest.pth', help=f'Model checkpoint--default=latest.pth')
  po.add_option('-d', '--root_dir', dest='base_directory_name', default='/usr/etc/hive/segmentation/', help=f'The root directory to find configs--default=/usr/etc/hive/segmentation/')
  po.add_option('-p', '--do-probs', dest='do_probs', default=False, help=f'Output probability fractions--default=False')
  po.add_option('-a', '--graph', dest='graph', help=f'Depricated')
  po.add_option('-q', '--do_quads', dest='quads', help=f'Depricated')
  po.add_option('-x', '--do_hexes', dest='hexes', help=f'Depricated')
  po.add_option('-g', '--use_gpu', dest='use-gpu', help=f'Ignored')
  (opts, args) = po.parse_args()

  try:
    if not opts.image_input:
      raise ArgumentException(f'Input image file must be specified')
    # TODO check if this is a file with and image list or an image file
    predict_image_file_list_file = opts.image_input
    if not os.path.exists(predict_image_file_list_file):
      raise ArgumentException(f'Input image file {predict_image_file_list_file} does not exist')
    if not opts.image_output:
      # TODO set this to the tif version of input if input is not a file
      raise ArgumentException(f'Results output image file must be specified')
    predict_output_file_list_file = opts.image_output
    if not os.path.exists(predict_output_file_list_file):
      raise ArgumentException(f'Output image file {predict_output_file_list_file} does not exist')
    try:
      scale = float(opts.image_scale)
    except ArgumentException as ei:
      raise ArgumentException(f'Scale must be a number and not {opts.image_scale}: {ei}')
    if scale <= 0.:
      raise ArgumentException(f'Scale must be greater than 0 and not {scale}')
    scale /= 100.
    try:
      do_probs = bool(opts.do_probs)
    except ArgumentException as ei:
      raise ArgumentException(f'Do_probs must be a boolean and not {opts.do_probs}: {ei}')

  except ArgumentException as e:
    print(e)
    po.print_help()
    sys.exit(1)

  base_directory = opts.base_directory_name
  config_file = os.path.join(base_directory, 'configs', opts.model_type, opts.model_name + '.py')
  if not os.path.exists(config_file):
    print("Invalid model type or model name in config file name", config_file)
    po.print_help()
    sys.exit(1)
  checkpoint_file = os.path.join(base_directory, 'work_dirs', opts.model_name, opts.checkpoint)
  if not os.path.exists(checkpoint_file):
    print("Invalid model name and/or checkpoint location", checkpoint_file)
    po.print_help()
    sys.exit(1)

  with open(predict_image_file_list_file) as f:
    predict_input_filename_list = f.read().splitlines()
  print("Input files", len(predict_input_filename_list))
  if len(predict_input_filename_list) == 0:
    print("No input files")
    po.print_help()
    sys.exit(1)

  with open(predict_output_file_list_file) as f:
    predict_output_filename_list = f.read().splitlines()
  print("Output files", len(predict_output_filename_list))
  if len(predict_input_filename_list) != len(predict_output_filename_list):
    print("Unequal input and output file lists")
    po.print_help()
    sys.exit(1)

  # Build the model from a config file and a checkpoint file
  model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
  # model = init_segmentor(config_file, checkpoint_file, device='cpu')

  # Get config for later and add classes
  cfg = Config.fromfile(config_file)
  COLOR_DICT_m7 = {0: "Other_m7", 1: "Mobile_m7", 2: "Trees_m7", 3: "Ground_m7", 4: "Structure_m7", 5: "Building_m7", 6: "Water_m7"}
  model.CLASSES = [val for key, val in COLOR_DICT_m7.items()]
  print("Class map:", model.CLASSES)

  # Image tensor data type depends on cuda
  device = next(model.parameters()).device
  is_cuda = next(model.parameters()).is_cuda
  print(device, is_cuda)

  # What is the effect of 2 vs 1?--also this variable only has an effect if do_probs
  upscale = 2.
  # TODO test
  upscale = 1.
  count = 0

  for image_filename, out_filename in zip(predict_input_filename_list, predict_output_filename_list):
    if fnmatch.fnmatch(image_filename, '*.jpg') or fnmatch.fnmatch(image_filename, '*.png'):
      count += 1
      print("Image", count, "of", len(predict_input_filename_list))
      img_array = mmcv.imread(image_filename)
      print("Running", image_filename, "with shape", img_array.shape)

      if do_probs:
        print("Seg logit output")
        image_meta = {}
        image_meta['ori_shape'] = img_array.shape
        img = mmcv.image.geometric.imrescale(img_array, upscale)
        image_meta['pad_shape'] = img.shape
        image_meta['img_shape'] = img.shape
        image_meta['scale_factor'] = scale
        image_meta['flip'] = False
        image_meta['flip_direction'] = 'horizontal'
        image_meta['filename'] = None
        image_meta['ori_filename'] = None
        image_meta['scale_factor'] = np.array([upscale, upscale, upscale, upscale], dtype=np.float32)
        image_meta['img_norm_cfg'] = cfg.img_norm_cfg

        # Prepare the image tensor
        image = torch.from_numpy(mmcv.bgr2rgb(img))
        image = image[np.newaxis, :]
        # Then to swap the axes as desired--note: _unsqueeze works fine here too
        image = image.permute(0, 3, 1, 2)
        image = torch.tensor(image, dtype=torch.float32)
        # print("Input:", image.shape)

        if is_cuda:
          image = image.cuda()

        # Run data but disable gradient
        with torch.no_grad():
          seg_logit = model.inference(image, [image_meta], rescale=True)
          result = F.softmax(seg_logit, dim=1)

        # Get data from result
        if is_cuda:
          # pull tensor out of GPU into system ram and convert to numpy
          output = np.array(result[0].permute(1, 2, 0).cpu())
        else:
          # Just convert to numpy
          output = np.array(result[0].permute(1, 2, 0))

        # Normalize each pixel val here
        class_counts = [0 for _ in model.CLASSES]
        for iy in range(output.shape[0]):
          for ix in range(output.shape[1]):
            element = output[iy][ix]
            min_val = np.min(element)
            element = np.subtract(element, min_val)
            class_counts[np.argmax(element)] += 1
            if np.sum(element) > 0.001:
              # Note this will be slightly off due to rounding when switched to uint8, but we don't need to be strict
              output[iy][ix] = np.round(np.multiply(element, 255/np.sum(element)))
            else:
              # The value is unknown so set the first element to 255
              element[0] = 255.
              output[iy][ix] = element
        print("Classes:", dict(zip(model.CLASSES, class_counts)))

      else:
        # Use the segmentation output to create output where chosen class is 100% and everything else is 0
        print("No probability output")
        result = inference_segmentor(model, img_array)
        output = np.multiply(one_hot(result[0], len(model.CLASSES)), 255)

      # Add rgb layers and an empty zero-valued shadow layer for compatibility and make 8-bit for saving as image
      shadow_layer = np.zeros((output.shape[0], output.shape[1])).reshape((output.shape[0], output.shape[1], 1))
      cat_vals = np.concatenate((mmcv.bgr2rgb(img_array), output, shadow_layer), axis=-1).astype('uint8')
      print("Saving output", out_filename, "with shape", cat_vals.shape, "and type", cat_vals.dtype)
      dir_name = os.path.dirname(out_filename)
      if not os.path.exists(dir_name):
        print("Making directory", dir_name)
        os.makedirs(dir_name)
      tifffile.imsave(out_filename, cat_vals, planarconfig='CONTIG')


if __name__ == '__main__':
  main()
