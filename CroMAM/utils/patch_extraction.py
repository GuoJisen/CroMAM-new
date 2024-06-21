import os
import cv2
import glob
import openslide
import numpy as np
from PIL import Image
from skimage import morphology
import imageio
from multiprocessing import Pool
import logging
import argparse
import SimpleITK as sitk

parser = argparse.ArgumentParser(description='Patch extraction')
parser.add_argument('--cancer', type=str, default='LGG')
parser.add_argument('--wsi-type', type=str, default='DX')
parser.add_argument('--num-cpus', type=int, default=1)
parser.add_argument('--magnification', type=int, default=5)
parser.add_argument('--patch-size', type=int, default=224)
parser.add_argument('--wsi_path', type=str, default='')
parser.add_argument('--wsi_mask_path', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
parser.add_argument('--stratify',
                    type=str, default='idh',
                    help='when spliting the datasets, stratify on which variable')

args = parser.parse_args()

logging.basicConfig(
    filename='../logs/patch-extraction-%s-%s-%s-%s.log' % (
        args.cancer, args.stratify, args.magnification, args.patch_size),
    format='%(message)s', level=logging.DEBUG)
UseMask = []
process_id = 0
tissue_threshold = 0.7,
cell_threshold = 0.05,
blur_threshold = 50


def get_mask_use(folder_dir):
    """
    Get the absolute path list of all "mask_use" mask files
    """
    mask_dirs = []
    for root, dirs, files in os.walk(folder_dir, topdown=False):
        for file in files:
            if "mask_use" in file:
                mask_dirs.append(os.path.join(root, file))
    return mask_dirs


def get_cell_core_ratio(img):
    """
    Calculate the proportion of cell nuclei in the patch and return the corresponding ratio
    (the cell nucleus ratio is generally 10%-5%)
    """
    image_array = img[:, :, 0]
    sitk_image = sitk.GetImageFromArray(image_array)

    lower_threshold = 0
    upper_threshold = 180

    binary_image = sitk.BinaryThreshold(sitk_image, lowerThreshold=lower_threshold, upperThreshold=upper_threshold,
                                        insideValue=255, outsideValue=0)
    binary_array = sitk.GetArrayFromImage(binary_image)
    binary_array = morphology.remove_small_objects(binary_array.astype('bool'), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_array = cv2.dilate(binary_array.astype(np.uint8), kernel, iterations=1)

    ts = int(img.shape[0])
    the_imagea = np.nan_to_num(binary_array.astype(np.uint8))
    mask = (the_imagea > 0).astype(np.uint8)
    white = np.sum(mask) / (ts * ts)
    return white


def get_blur_score(img):
    """
    Quantifying the blur of an image
    """
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(imgg, cv2.CV_64F).var()
    return blur


def objcheck(img, ts, t=240):
    """
    Calculate the image area ratio of the target
    """
    if img.shape == (ts, ts):
        the_imagea = img
        the_imagea = np.nan_to_num(the_imagea)
        mask = (the_imagea > t).astype(np.uint8)
        obj_ratio = np.sum(mask) / (ts * ts)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        the_imagea = np.nan_to_num(img)
        mask = (the_imagea < t).astype(np.uint8)
        obj_ratio = np.sum(mask) / (ts * ts)
    return obj_ratio


def get_patch_name(x, y):
    return os.path.join(str(x), str(y) + '.png')


def cut_image(slide, use_mask, target_path, scale):
    patch_nums = 0
    H_rx, W_rx = np.array(use_mask).shape
    for x_rx in range(0, W_rx, args.patch_size):
        if x_rx + args.patch_size > W_rx:
            x_rx = W_rx - args.patch_size
        for y_rx in range(0, H_rx, args.patch_size):
            if y_rx + args.patch_size > H_rx:
                y_rx = H_rx - args.patch_size
            use_mask_patch = use_mask.crop((x_rx, y_rx, x_rx + args.patch_size, y_rx + args.patch_size))
            use_mask_patch = np.array(use_mask_patch)
            mask_rate = objcheck(use_mask_patch, args.patch_size)
            if mask_rate >= tissue_threshold:
                img = slide.read_region((x_rx * scale, y_rx * scale), 0,
                                        (args.patch_size * scale, args.patch_size * scale)).convert(
                    'RGB')
                img = np.array(img.resize((args.patch_size, args.patch_size)))
                # tissue_rate = objcheck(img, args.patch_size)
                cell_rate = get_cell_core_ratio(img)
                blur_score = get_blur_score(img)
                if cell_rate >= cell_threshold and blur_score >= blur_threshold:
                    final_path = os.path.join(target_path, get_patch_name(x_rx, y_rx))
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    imageio.imwrite(final_path, img, format='jpg')
                    patch_nums += 1
    return patch_nums


def my_extract_patch_from_svs(idx, fname, target_path, wsi_num):
    if args.wsi_type in fname.split('/')[-1]:
        wsiname = fname.split('/')[-1].split('.svs')[0]
        iddir = '-'.join(wsiname.split('-')[:3])
        alldir = os.path.join(target_path, iddir, wsiname)
        if not os.path.exists(alldir):
            use_mask = None
            slide = openslide.open_slide(fname)
            mag = int(float(slide.properties['aperio.AppMag']))
            if args.magnification <= mag:
                MAX_W, MAX_H = slide.dimensions
                scale = int(mag / args.magnification)  # Zoom ratio
                W_rx, H_rx = int(MAX_W / scale), int(MAX_H / scale)
                for i in range(len(UseMask)):  # Find the mask that matches
                    if wsiname in UseMask[i].split('/')[-1]:
                        use_mask = Image.open(UseMask[i])
                        use_mask = use_mask.resize((W_rx, H_rx))
                        break
                if use_mask is not None:
                    print(f"({idx + 1}/{int(wsi_num)})----processing:{wsiname}")
                    patch_nums = cut_image(slide, use_mask, alldir, scale)
                    # When the number of patches is insufficient, random slicing
                    if patch_nums < 64:
                        index = np.where(np.array(use_mask) == 255)
                        random_index = np.array(range(0, int(len(index[0]))))
                        np.random.shuffle(random_index)
                        for idx in random_index:
                            x_rx = index[1][idx]
                            y_rx = index[0][idx]
                            final_path = os.path.join(alldir, get_patch_name(x_rx, y_rx))
                            if (not os.path.exists(final_path)
                                    and use_mask.size[0] >= x_rx + args.patch_size
                                    and use_mask.size[1] >= y_rx + args.patch_size):
                                use_mask_patch = use_mask.crop(
                                    (x_rx, y_rx, x_rx + args.patch_size, y_rx + args.patch_size))
                                use_mask_patch = np.array(use_mask_patch)
                                mask_rate = objcheck(use_mask_patch, args.patch_size)
                                if mask_rate >= tissue_threshold:
                                    img = slide.read_region((x_rx * scale, y_rx * scale), 0,
                                                            (args.patch_size * scale, args.patch_size * scale)).convert(
                                        'RGB')
                                    img = np.array(img.resize((args.patch_size, args.patch_size)))
                                    # tissue_rate = objcheck(img, args.patch_size)
                                    cell_rate = get_cell_core_ratio(img)
                                    blur_score = get_blur_score(img)
                                    if cell_rate >= cell_threshold and blur_score >= blur_threshold:
                                        os.makedirs(os.path.dirname(final_path), exist_ok=True)
                                        imageio.imwrite(final_path, img, format='jpg')
                                        patch_nums += 1
                                if patch_nums >= 64:
                                    break
                else:
                    print(f"DX-WSI discarded during quality control:{wsiname}")
                    pass
            else:
                print(f"WSI with insufficient magnification:{wsiname}")
            logging.info(fname)
            slide.close()
            if not os.path.exists(alldir):
                os.makedirs(alldir, exist_ok=True)
        else:
            print(f"Processed WSI:{alldir}")
            pass
    else:
        print(f"Not a DX type WSI:{fname.split('/')[-1]}")
        pass


def extract_patches_for_cancer(input_path, output_path):
    with Pool(args.num_cpus) as pool:
        # Revise %s/*/*.svs
        iterable = [(svs_file, output_path) for svs_file in glob.glob('%s/*/*.svs' % input_path)]
        try:
            for idx, item in enumerate(iterable):
                svs_file, output_dir = item
                _ = pool.apply_async(
                    func=my_extract_patch_from_svs,
                    args=(idx, svs_file, output_dir, len(iterable))
                )
        finally:
            pool.close()
            pool.join()


if __name__ == '__main__':
    # get mask
    UseMask = get_mask_use(args.wsi_mask_path)
    print('Working on cancer %s' % args.cancer)
    output_path = args.output_path + '/%s/%s_%s' % (args.cancer, args.magnification, args.patch_size)
    os.makedirs(output_path, exist_ok=True)
    extract_patches_for_cancer(args.wsi_path, output_path)
