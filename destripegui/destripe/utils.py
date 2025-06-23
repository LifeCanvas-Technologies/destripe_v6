from pathlib import Path
import os, time
import numpy as np
import tifffile
import imageio as iio
from skimage.filters import threshold_otsu
import tqdm
from . import raw
import warnings
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
        tic = time.time()
        th = threshold_otsu(img)
        # print("Thresholding time: " + str(time.time() - tic) + " seconds")
        return th
    else:
        return threshold_prompt

def attempt_read_threshold(input_path, threshold_prompt=None):
    num_attempts = 3
    for _ in range(num_attempts):
        try:
            img = imread(input_path)
            break
        except:
            continue
    else:
        return None, None
    
    th = threshold_img(img, threshold_prompt)

    return img, th

def imsave(path : str, 
            img : np.ndarray,
            compression : int | None = None, 
            output_format : Optional[str] = None, 
            rotate_and_flip : bool = False):
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

    if rotate_and_flip:
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

def find_all_images(search_path, input_path, output_path):
    """Find all images with a supported file extension within a directory and all its subdirectories

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
    output_path = Path(output_path)
    search_path = Path(search_path)

    assert search_path.is_dir()
    img_paths = []
    for p in search_path.iterdir():
        if p.is_file():
            if p.suffix in supported_extensions:
                img_paths.append(p)
        elif p.is_dir():
            rel_path = p.relative_to(input_path)
            o = output_path.joinpath(rel_path)
            if not o.exists():
                o.mkdir(parents=True)
            img_paths.extend(find_all_images(p, input_path, output_path))
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


def normalize_flat(flat):
    """
    Normalize a flat field image by dividing it by its maximum value.
    
    Args:
    flat (numpy.ndarray): The flat field image to be normalized.
    
    Returns:
    numpy.ndarray: The normalized flat field image.
    """
    flat_float = flat.astype(np.float32)
    return flat_float / flat_float.max()

def notch(n, sigma):
    """Generates a 1D gaussian notch filter `n` pixels long

    Parameters
    ----------
    n : int
        length of the gaussian notch filter
    sigma : float
        notch width

    Returns
    -------
    g : ndarray
        (n,) array containing the gaussian notch filter

    """
    if n <= 0:
        raise ValueError('n must be positive')
    else:
        n = int(n)
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    x = np.arange(n)
    g = 1 - np.exp(-x ** 2 / (2 * sigma ** 2))
    return g


def gaussian_filter(shape, sigma):
    """Create a gaussian notch filter

    Parameters
    ----------
    shape : tuple
        shape of the output filter
    sigma : float
        filter bandwidth

    Returns
    -------
    g : ndarray
        the impulse response of the gaussian notch filter

    """
    g = notch(n=shape[-1], sigma=sigma)
    g_mask = np.broadcast_to(g, shape).copy()
    return g_mask