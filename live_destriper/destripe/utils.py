from pathlib import Path
import os, time
import numpy as np
import tifffile
import imageio as iio
from skimage.filters import threshold_otsu
import tqdm
from . import raw
import warnings
import psutil
import shutil
from typing import Optional
warnings.filterwarnings("ignore")

from . import supported_extensions, supported_output_extensions


def get_extension(path):
    """Extract the file extension from the provided path

    Parameters
    ----------
    path : str
        path with a file extension

    Returns
    -------
    ext : str
        file extension of provided path

    """
    return Path(path).suffix


def imread(path):
    """Load a tiff or raw image

    Parameters
    ----------
    path : str
        path to tiff or raw image

    Returns
    -------
    img : ndarray
        image as a numpy array

    """
    img = None
    extension = get_extension(path)
    if extension == '.raw':
        img = raw.raw_imread(path)
    elif extension == '.tif' or extension == '.tiff':
        img = tifffile.imread(path)
    elif extension == '.png':
        img = iio.imread(path)
    return img

def threshold_img(img, threshold_prompt=None):
    if threshold_prompt is None or threshold_prompt < 0:
        th = threshold_otsu(img)
        return th
    else:
        return threshold_prompt

def attempt_read_threshold(input_path, threshold_prompt=None):
    try:
        img = imread(input_path)
    except Exception as e:
        return None, None
    except KeyboardInterrupt:
        raise

    
    try:
        # accoutn for when the image is successfully read but there is something wrong with the data.
        th = threshold_img(img, threshold_prompt)
    except Exception as e:
        return None, None
    except KeyboardInterrupt:
        raise

    return img, th

def imsave(path : str, 
            img : np.ndarray,
            compression : int | None = None, 
            output_format : Optional[str] = None, 
            rotate_and_flip : bool = False,
            extra_rotate_and_flip : bool = False):
    """Save an array as a tiff or raw image

    The file format will be inferred from the file extension in `path`

    Parameters
    ----------
    path : str
        path to tiff or raw image
    img : ndarray
        image as a numpy array
    compression : int
        compression level for tiff writing
    output_format : Optional[str]
        Desired format extension to save the image. Default: None
        Accepted ['.tiff', '.tif', '.png']
     rotate_and_flip : bool
        If True, then rotate counterclockwise + flip (in axis=0) each image in X,Y before saving 
    """
    extension = get_extension(path)
    # print(compression)
    if compression is None:
            compression_scheme, compression_level = False, None
    elif compression is True:
            compression_scheme, compression_level = 'zlib', 1
    elif isinstance(compression, int):
            compression_scheme, compression_level = 'zlib', compression
    else:
        raise Exception('Invalid compression argument: {}'.format(compression))
    # print(compression_scheme, compression_level)

    if extra_rotate_and_flip:
        # Only for the Fire camera
        img = np.rot90(img, 2)

    elif rotate_and_flip:
        img = np.rot90(img)
        img = np.flip(img, axis=0)

    if output_format is None:
        # Saving any input format to tiff
        
        tifffile.imwrite(path, img, compression=compression_scheme, compressionargs={'level': compression_level}) # Use with version 2023.03.21

    else:
        # Saving output images based on the output format
        if output_format not in supported_output_extensions:
            raise ValueError(f"Output format {output_format} is not valid! Supported extensions are: {supported_output_extensions}")

        filename = os.path.splitext(path)[0] + output_format
        if output_format == '.tif' or output_format == '.tiff':
            tifffile.imwrite(path, img, compression=compression_scheme, compressionargs={'level': compression_level}) # Use with version 2023.03.21

def find_all_images_in_dir(input_path):
    """Find all images with a supported file extension within a single directory

    Parameters
    ----------
    input_path : path-like
        root directory to start image search

    Returns
    -------
    img_paths : list
        a list of Path objects for all found images

    """
    input_path = Path(input_path)

    img_paths = []
    for p in input_path.iterdir():
        if p.is_file():
            if p.suffix in supported_extensions:
                img_paths.append(p)
    return img_paths



def interpolate(image_path, input_path, output_path):
    rel_path = Path(image_path).relative_to(input_path)
    o_dir = os.path.dirname(output_path.joinpath(rel_path))
    image_num = int(os.path.splitext(os.path.split(image_path)[1])[0])
    closest_image = {
        'name': os.listdir(o_dir)[0],
        'distance': abs(int(os.path.splitext(os.listdir(o_dir)[0])[0]) - image_num)
        }
    for filename in os.listdir(o_dir):
        try: 
            test_num = int(os.path.splitext(filename)[0])
        except:
            continue
        if abs(test_num - image_num) < closest_image['distance']:
            closest_image['name'] = filename
            closest_image['distance'] = abs(test_num - image_num)
    new_file_name = str(image_num) + os.path.splitext(closest_image['name'])[1]
    try:
        shutil.copyfile(os.path.join(o_dir, closest_image['name']), os.path.join(o_dir, new_file_name))
    except Exception as e:
        pass


# Turbo image reading for destriping (requires to get the threshold as well)
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

def read_threshold_single_img_to_shared_memory(image_path, img_shm_name, th_shm_name, shape, threshold, z):
    """
    Function for assigning to shared memory a loaded image, as well as getting the threshold for that image.
    """
    try:
        img = imread(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}: e")
        raise FileNotFoundError from e
    
    try:
        shm = SharedMemory(name=img_shm_name, create=False)
        # move data to shared array
        shared_array = np.ndarray(shape, dtype=img.dtype, buffer=shm.buf)
        shared_array[z] = img

        # Threshold
        th = threshold_img(img, threshold)
        th_shm = SharedMemory(name=th_shm_name, create=False)
        th_array = np.ndarray((shape[0],), dtype=np.float32, buffer=th_shm.buf)
        th_array[z] = th
    except TypeError as e:
        print(f"Error putting image data to shared_array: {e}, z: {z}")
        raise TypeError from e
    finally:
        shm.close()
        th_shm.close()


def read_threshold_shm(file_list : list, 
                       threshold_prompt,
                       vol_shape, 
                        img_shm : SharedMemory,
                        thresh_shm : SharedMemory,
                        num_workers : str = psutil.cpu_count(logical=False)
                        ):
    """
    Parallel function for reading images into shared memory.
    """
    pool = mp.Pool(processes=num_workers)
    results = []

    mp.freeze_support()
    try:
        for i, tiff in enumerate(tqdm.tqdm(file_list)):
            results.append(
                pool.apply_async(
                    read_threshold_single_img_to_shared_memory, (tiff, img_shm.name, thresh_shm.name, vol_shape, threshold_prompt, i)
                )
            )

        for result in results:
            result.get()

    except Exception as e:
        print(f"Error in  read_threshold_single_img_to_shared_memory: {e}")
        raise

    finally:
        pool.close()
        pool.join()

    print(f"Successfully loaded tiff data to memory")


# For writing from SHM to tiff stack after destriping (or deskewing???)
def write_shm_to_single_tiff(output_path,
                             shm_name, 
                             shape, 
                             img_dtype, 
                             output_format,
                             compression,
                             rotate_and_flip,
                             extra_rotate_and_flip,
                             index):
    """
    Function for writing shared memory to a tiff file.
    """
    try:
        shm = SharedMemory(name=shm_name, create=False)
        shared_array = np.ndarray(shape, dtype=img_dtype, buffer=shm.buf)
        img = shared_array[index]
        imsave(output_path, img, compression = compression, output_format=output_format, rotate_and_flip = rotate_and_flip, extra_rotate_and_flip = extra_rotate_and_flip)
    except Exception as e:
        print(f"Error writing shared memory to tiff: {e}")
        raise
    finally:
        shm.close()



def write_shm_to_tiff(output_paths, shm, shape, img_dtype, output_format, compression, rotate_and_flip = True, extra_rotate_and_flip = False, num_workers = psutil.cpu_count(logical=False)):
    shm_name = shm.name

    pool = mp.Pool(processes=num_workers)
    results = []

    mp.freeze_support()
    try:
        for idx, output_path in enumerate(tqdm.tqdm(output_paths)):
            results.append(
                pool.apply_async(
                    write_shm_to_single_tiff, (output_path, shm_name, shape, img_dtype, output_format, compression, rotate_and_flip, extra_rotate_and_flip, idx)
                )
            )

        for result in results:
            result.get()

    except Exception as e:
        print(f"Error in  write_shm_to_single_tiff(): {e}")
        raise

    finally:
        pool.close()
        pool.join()

    print(f"Successfully wrote shared memory to tiff stack")