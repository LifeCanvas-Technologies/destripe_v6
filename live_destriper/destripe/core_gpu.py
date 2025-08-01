import argparse
from argparse import RawDescriptionHelpFormatter
from pathlib import Path
import os, time
import numpy as np
from scipy import fftpack
from skimage.filters import threshold_otsu
import tifffile
import pywt
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import tqdm
import warnings
import shutil
from typing import Optional, List, Union
import torch, torchvision
import ptwt
import psutil
import more_itertools as mit
from functools import partial
warnings.filterwarnings("ignore")
import logging
import atexit

from . import supported_output_extensions
from .utils import find_all_images_in_dir, interpolate, imread, attempt_read_threshold, read_threshold_shm, write_shm_to_tiff
from .filter_utils import normalize_flat, gaussian_filter

from live_destriper.logger import get_logger, get_default_logger
from live_destriper.utils import cleanup_shared_memory

DESTRIPE_SHM_NAME = "DESTRIPE_SHM"
DESTRIPE_WRITE_SHM_NAME = "DESTRIPE_WRITE_SHM"
THRESHOLD_SHM_NAME = "THRESHOLD_SHM"


# This destriper only destripes a single folder of images at a time.
class Destriper:
    def __init__(self,
                input_path : Path,
                output_path : Path,
                sigma : List[float],
                level : int = 0,
                wavelet : str = 'db3',
                crossover : float = 10,
                threshold : float = -1,
                compression : int = 1,
                flat : Optional[np.ndarray] = None,
                dark : float = 0,
                num_workers : Optional[int] = None,
                ram_loadsize : Optional[int] = None,
                gpu_chunksize : Optional[int] = None,
                extra_smoothing : int = 1,
                dont_convert_16bit : bool = False,
                output_format : Optional[str] = None, 
                logger : Optional[logging.Logger] = None,
                rotate_and_flip = False,
                extra_rotate_and_flip = False,
                use_shared_memory = True):
        """Destriper class that applies `streak_filter` to all images in `input_path` and write the results to `output_path`.

        Parameters
        ----------
        input_path : Path
            root directory to search for images to filter
        output_path : Path
            root directory for writing results
        sigma : list or float
            bandwidth of the stripe filter in pixels. If single float, then assumes same bandwidth for foreground and background.
        level : int
            number of wavelet levels to use
        wavelet : str
            name of the mother wavelet
        crossover : float
            intensity range to switch between filtered background and unfiltered foreground. Default: 100 a.u.
        threshold : float
            intensity value to separate background from foreground. Default is Otsu
        compression : int
            compression level to use in tiff writing
        flat : ndarray
            reference image for illumination correction. Must be same shape as input images. Default is None
        dark : float
            Intensity to subtract from the images for dark offset. Default is 0.
        ram_loadsize : int
            Number of images to load at once on CPU RAM prior to chunked GPU destriping. Imputed from system metrics if not provied.
        gpu_chunksize : int
            number of images for GPU to process at a time. Please note that GPU wavelet destriping uses
            on the order of 8 times the amount of memory relative to input. Caution is advised to avoid
            GPU memory limit errors by submitting only a small number of images. If None, automatically calculates ideal chunk size.
        dont_convert_16bit : bool
            Flag for converting to 16-bit
        output_format: str
            Desired output format [.tiff, .tif]. Default None
        extra_smoothing : int
            Whether extra gaussian smoothing is applied to the foreground fraction.
            calculation. If so, can be an int to specify the magnitude of doing so.
        """
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok = True)
        self.destripe_list_path = os.path.join(output_path, "destriped_folder_list.txt") # make this by each folder that contains images
        self.sigma = sigma 
        self.level = level
        self.wavelet = wavelet
        self.crossover = crossover
        self.provided_threshold = threshold
        self.compression = compression
        self.flat = flat
        self.dark = dark
        self.num_workers = num_workers if num_workers is not None else psutil.cpu_count(logical=False)
        self.ram_loadsize = ram_loadsize
        self.gpu_chunksize = gpu_chunksize 

        self.dont_convert_16bit = dont_convert_16bit
        self.output_format = output_format
        self.extra_smoothing = extra_smoothing
        self.logger = logger if logger is not None else get_default_logger()
        self.rotate_and_flip = rotate_and_flip
        self.extra_rotate_and_flip = extra_rotate_and_flip
        self.use_shared_memory = use_shared_memory


    def max_level(self, min_len):
        w = pywt.Wavelet(self.wavelet)
        return pywt.dwt_max_level(min_len, w.dec_len)
    
    def foreground_fraction_torch(self, imgs_torch, threshold):
        """
        Calculates the foreground fraction of the input images using a threshold and crossover value.
        
        Args:
            imgs_torch (torch.Tensor): Input images as a torch.Tensor.
        
        Returns:
            torch.Tensor: Foreground fraction of the input images.
        """
        z = (imgs_torch - threshold)/self.crossover
        f = 1 / (1 + torch.exp(-z))

        if self.extra_smoothing:
            ks = (9, 9)  # kernal size, set to ~ NDimage defaullt
            return torchvision.transforms.functional.gaussian_blur(f, ks, self.extra_smoothing)
        else:
            return f

    @staticmethod
    def apply_flat_torch(imgs_torch, flat):
        if flat.is_tensor() is False:
            flat = torch.from_numpy(flat.astype(np.float32))
        if imgs_torch.get_device() != flat.get_device():
            flat = flat.to(device=imgs_torch.get_device())
        return (imgs_torch / flat)

    
    @staticmethod
    def num_cpu_readers():
        if os.cpu_count() <= 16:
            return 12
        elif os.cpu_count() <= 32:
            return 24
        elif os.cpu_count() <= 64:
            return 32
        elif os.cpu_count() >= 120:
            return 60 
        else:
            # return int(0.60 * os.cpu_count())
            return os.cpu_count()
    
    
    def prepare_batch(self, args_batch, img_dtype=np.uint16, img_shape=None, img_sham = None, thresh_sham = None):
        """
        Convert a batch of images to chunks of int16 torch tensors, and return alongside updated arguments.

        Args:
            args_batch (list): A list of dictionaries, where each dictionary contains the arguments for loading an image.

        Returns:
            torch.Tensor: A torch tensor containing the batch of images with dtype torch.int16.
        """
        def arrs_to_torch(arrs):
            stacked = np.stack(list(arrs))
            if stacked.dtype == np.int16:
                raise Exception('Native signed int 16 images are not supported. Please convert to unsigned int 16 or another compatible format.')
            elif stacked.dtype == np.uint16:
                stacked.dtype = np.int16  # Temporarlly store as offset int16 for pytorch compatability
            return torch.from_numpy(stacked)

        if not self.use_shared_memory:
            inputs_paths = [str(args['input_path']) for args in args_batch]
            if len(args_batch) > 1 and self.num_workers not in [0, 1]:
                with multiprocessing.Pool(self.num_workers) as p:  # use multiprocessing for reading PNGs and RAWs
                    f = partial(attempt_read_threshold, threshold_prompt=self.provided_threshold)
                    print("Image loading progress:")
                    imgs_and_thresholds = list(tqdm.tqdm(p.imap(f, inputs_paths), total=len(inputs_paths)))
                    imgs, thresholds = zip(*imgs_and_thresholds)
            else:
                imgs, thresholds = zip(*[attempt_read_threshold(input_path, threshold_prompt=self.provided_threshold) for input_path in tqdm.tqdm(inputs_paths)])
        else:
            img_paths = [str(args['input_path']) for args in args_batch]
            threshold_prompt = self.provided_threshold
            read_threshold_shm(img_paths, threshold_prompt, img_shape, img_sham, thresh_sham, num_workers = self.num_workers)

            # And then we get the data from the shared memory
            imgs = np.ndarray(img_shape, dtype=img_dtype, buffer=img_sham.buf)
            thresholds = np.ndarray(len(args_batch), dtype=np.float32, buffer=thresh_sham.buf)
            


        for i, args in enumerate(args_batch):
            args['threshold'] = thresholds[i]

        
        loaded = filter(lambda img_args: img_args[0] is not None, zip(imgs, args_batch))
        retvals = []
        for load_chunk in mit.chunked_even(loaded, self.gpu_chunksize):
            imgs, args_chunk = zip(*load_chunk)
            retvals.append((arrs_to_torch(imgs), args_chunk))

        return retvals

    @staticmethod
    def offsign16_to_32(int16_tensor):
        """
        Converts a tensor of int16 values to int32 values by adding a shift value.
        
        Args:
            int16_tensor (torch.Tensor): The input tensor containing int16 values.
        
        Returns:
            torch.Tensor: The converted tensor with int32 values.
        """
        shift = int(2**15)
        t = int16_tensor + shift
        t = t.to(dtype=torch.float32)
        return t + shift


    @staticmethod
    def set_ram_load_size(img_dims=(2000, 1600)):
        """
        Imputes a decent load size that can optimally be loaded at scale available amount of RAM memory and image dimensions.

        Args:
            img_dims (tuple): Dimensions of the input images. Default is (2048, 2048).

        Returns:
            int: Maximum batch size that can be used on the GPU.
        """
        try:
            ram_info = psutil.virtual_memory()    
        except FileNotFoundError:
            print("RAM info not available on this system")
            return 3000  # safe albeit arbitrary default
        
        x, y = img_dims
        bitsize = 16
        use_factor = 0.25
        img_bytsize = x * y * (bitsize / 8)

        return int(use_factor * ram_info.available / img_bytsize)


    @staticmethod
    def set_gpu_batch_size(img_dims=(2048, 2048)):
        """
        Calculates the batch size that can safely be used on the GPU based on the available memory and image dimensions.

        Args:
            img_dims (tuple): Dimensions of the input images. Default is (2048, 2048).

        Returns:
            int: Maximum batch size that can be used on the GPU.
        """
        gpu_mem_for_use, _ = torch.cuda.mem_get_info()

        # approx_rtx4090_mem = 24125636608 # gpu_mem_for_use from RTX 4090 
        trial_maxdim = 2048
        trial_nimgs = 100
        trial_x = trial_nimgs * (trial_maxdim**2)  # 64 2048x2048 imgs
        #  2048^2 * 64 * k <= 24 Gigs
        k = 24e9 / trial_x

        batch_size = np.floor(gpu_mem_for_use/(k * img_dims[0]*img_dims[1]))
        return batch_size.astype('int')


    def destripe_torch32(self, imgs_torch, args_chunk):
        """
        Destripes one batch of images using the GPU. 
        """
        assert(imgs_torch.shape[-1] % 2 == 0 and imgs_torch.shape[-2] % 2 == 0), "Image dimensions must be even (a multiple of two). If non-standard image sizes are being used, contact Ben Kaplan ben.kaplan@lifecanvastech.com for support"

        fimgs = self.filtersmooth_subbands_gpu(imgs_torch, args_chunk)

        if self.dark > 0:
            fimgs = fimgs - self.dark

        # Divide by the flat
        if self.flat is not None:
            fimgs = self.apply_flat_torch(fimgs, self.flat)
        
        max_uint16 = float(2**16 - 1)
        fimgs = torch.clip(fimgs, 0, max_uint16)
        fimgs = torch.round(fimgs)
        return fimgs

    @staticmethod
    def smooth_ch(coeffs, width_frac):
        for i in range(1, len(coeffs)):
            ch, cv, cd = coeffs[i]
            s = ch.size(-2) * width_frac
            fch = torch.fft.rfft(ch, axis=-1)
            g = gaussian_filter(shape=fch.shape[-2:], sigma=s)
            g = torch.from_numpy(np.float32(g)).to(device='cuda')
            fch_filt = fch * g
            dim_n = ch.size(-1)
            ch_filt = torch.fft.irfft(fch_filt, n=dim_n)
            coeffs[i] = (ch_filt, cv, cd)
        return coeffs

    def _filter_ground(self, imgs_torch, ground, sigma):
        """
        Apply ground filtering to the input ground image.

        Args:
            ground (torch.Tensor): Input ground image.
            sigma (float): Standard deviation of the Gaussian filter.

        Returns:
            torch.Tensor: Filtered ground image.
        """
        ground_log = torch.log(1 + ground)
        sigma_factor = sigma / imgs_torch.size(-1)
        use_level = self.level if self.level != 0 else None
        ground_coeffs =  ptwt.wavedec2(ground_log, self.wavelet, level=use_level)
        ground_coeffs = self.smooth_ch(ground_coeffs, sigma_factor)
        ground_rec = ptwt.waverec2(ground_coeffs, self.wavelet)
        ground_filtered = torch.exp(ground_rec) - 1
        return ground_filtered

    def filtersmooth_subbands_gpu(self, imgs_torch, args_chunk):
        thresholds_arr = np.array([[[args['threshold']]] for args in args_chunk])
        thresholds_torch = torch.from_numpy(thresholds_arr).to(device='cuda', dtype=torch.float32)
        foreground_sigma = self.sigma[0] # foreground
        background_sigma = self.sigma[1] # background

        do_foreground = foreground_sigma > 0
        do_background = background_sigma > 0 and foreground_sigma != background_sigma

        if do_foreground is False and do_background is False:
            return imgs_torch # no de-striping is done
        
        if do_foreground:
            foreground = (torch.clip(imgs_torch, thresholds_torch, None)

                          if foreground_sigma != background_sigma else imgs_torch)
            foreground_filtered = self._filter_ground(imgs_torch, foreground, foreground_sigma)
        else:
            foreground_filtered = imgs_torch

        if do_background:
            background = torch.clip(imgs_torch, None, thresholds_torch)
            background_filtered = self._filter_ground(imgs_torch, background, background_sigma)
        else:
            background_filtered = imgs_torch

        if foreground_sigma != background_sigma:
            f  = self.foreground_fraction_torch(imgs_torch, thresholds_torch)

            return foreground_filtered * f + background_filtered * (1 - f)
        else:
            return foreground_filtered


    def batch_filter(self):
        """
        Workhorse function for doing parallel GPU destriping.
        """
        # Error logs path
        error_path = os.path.join(self.output_path, 'destripe_log.txt')
        if os.path.exists(error_path):
            os.remove(error_path)

        # Find all the images in the input path to be destriped
        print('Looking for images in {}...'.format(self.input_path))
        img_paths = find_all_images_in_dir(self.input_path)
        print('Found {} compatible images'.format(len(img_paths)))

        temp_img = imread(str(img_paths[0]))
        img_size = temp_img.shape
        img_dtype = temp_img.dtype

        # Get the image size to set the ram_loadsize parameter (if None)
        if self.ram_loadsize is None:
            self.ram_loadsize = self.set_ram_load_size(img_size)

        # Get the image size to set the gpu_chunksize parameter (if None)
        if self.gpu_chunksize is None:
            self.gpu_chunksize = self.set_gpu_batch_size(img_size)
        
        # Do GPU destriping for each batch of images and write to disk 
        args = []
        for p in img_paths:
            # Find the relative output path and make the directory 
            rel_path = p.relative_to(self.input_path)
            o = self.output_path.joinpath(rel_path)
            o = o.with_suffix(self.output_format) if self.output_format is not None else o
            if not o.parent.exists():
                o.parent.mkdir(parents=True)
            
            arg_dict = {
                'input_path': p,
                'output_path': o,
                'threshold': self.provided_threshold
            }
            args.append(arg_dict)

        print('Pystripe batch processing progress:')
        load_args = list(mit.chunked_even(args, self.ram_loadsize))
        
        # base names for the SHM objects in this folder
        img_sham_name = DESTRIPE_SHM_NAME# + f"_{str(self.input_path)}"
        thresh_sham_name = THRESHOLD_SHM_NAME #+ f"_{str(self.input_path)}"
        destriped_imgs_shm_name = DESTRIPE_WRITE_SHM_NAME #+ f"_{str(self.input_path)}"

        batch_counter = 0
        with tqdm.tqdm(total=(len(args)), ascii=True, bar_format=None) as pbar:
            for load_batch in load_args:
                start = time.time()
                # Get the 3D image shape of the load batch (in z)
                img_shape = (len(load_batch),*img_size)

                # The shared memory object to store the destriped images
                destriped_imgs_shm = SharedMemory(create=True, size=int(img_shape[0]*img_shape[1]*img_shape[2] * img_dtype.itemsize),
                                                    name=destriped_imgs_shm_name+f"_{batch_counter}")
                atexit.register(cleanup_shared_memory, destriped_imgs_shm_name+f"_{batch_counter}") # make sure it gets deleted on exit
                img_sham = SharedMemory(create=True, size=int(img_shape[0]*img_shape[1]*img_shape[2]* img_dtype.itemsize),
                                        name=img_sham_name+f"_{batch_counter}")
                atexit.register(cleanup_shared_memory, img_sham_name+f"_{batch_counter}")
                thresh_sham = SharedMemory(create=True, size=len(load_batch) * np.dtype(np.float32).itemsize,
                                        name=thresh_sham_name+"_{}".format(batch_counter))
                atexit.register(cleanup_shared_memory, thresh_sham_name+f"_{batch_counter}")
                shared_array = np.ndarray(img_shape, dtype=img_dtype, buffer=destriped_imgs_shm.buf)
                chunked = self.prepare_batch(load_batch, img_dtype = img_dtype, img_shape = img_shape, img_sham=img_sham, thresh_sham=thresh_sham)
                self.logger.info('Batch %d: %d images loaded in %0.2f seconds'%(batch_counter, len(load_batch), time.time() - start))

                num_images_counter = 0 # total number of images that have been destriped so far for this batch
                last_imgs_chunk_numpy = None
                for imgs_chunk, args_chunk in chunked:
                    imgs_chunk = imgs_chunk.to(device='cuda', non_blocking=False)
                    images32 = self.offsign16_to_32(imgs_chunk)
                    imgs_chunk32 = self.destripe_torch32(images32, args_chunk)

                    imgs_chunk = imgs_chunk32.to(device='cpu', dtype=torch.int16, non_blocking=True)
                    imgs_chunk_numpy = imgs_chunk.numpy()
                    if imgs_chunk_numpy.dtype == np.int16:
                        imgs_chunk_numpy.dtype = np.uint16

                    torch.cuda.synchronize()
                    if last_imgs_chunk_numpy is not None:
                        try:
                            # If we try to save it right away with non-blocking, there will be a racing condition in writing data
                            shared_array[num_images_counter:num_images_counter + len(last_imgs_chunk_numpy)] = last_imgs_chunk_numpy
                            
                        except Exception as e:
                            self.logger.error(f"Error in writing to shared memory: {e}")
                            raise Exception("Error in writing to shared memory")
                        # Update the number of images that have been destriped so we know where in the array to save it
                        num_images_counter += len(last_imgs_chunk_numpy)

                    # Update the next batch of images to be written to shared memory
                    last_imgs_chunk_numpy = imgs_chunk_numpy

                    # Update progress bar
                    pbar.update(len(imgs_chunk))
                
                # After the for loop we need to write the last chunk of images to shared memory
                shared_array[num_images_counter:num_images_counter + len(last_imgs_chunk_numpy)] = last_imgs_chunk_numpy                            

                # Write images to disk from shared memory
                output_paths = [args['output_path'] \
                                for args2 in [args_chunk for _, args_chunk in chunked] \
                                for args in args2]
                write_shm_to_tiff(output_paths, 
                                destriped_imgs_shm, 
                                img_shape, 
                                img_dtype, 
                                self.output_format, 
                                self.compression, 
                                rotate_and_flip = self.rotate_and_flip,
                                extra_rotate_and_flip = self.extra_rotate_and_flip, 
                                num_workers = self.num_workers)
            
                # Close and unlink the shared memory
                destriped_imgs_shm.close()
                destriped_imgs_shm.unlink()
                img_sham.close()
                img_sham.unlink()
                thresh_sham.close()
                thresh_sham.unlink()

                # update counter
                batch_counter += 1

        # Interpolate images that could not be opened
        if os.path.exists(error_path):
            self.logger.info("Some images could not be opened, beginning interpolation...")
            with open(error_path, 'r') as fp:
                images = fp.readlines()
                for image_path in images:
                    interpolate(image_path, self.input_path, self.output_path)
                x = len(images)
                self.logger.info('{} images could not be opened and were interpolated.  See destripe log for more details'.format(x))
                fp.close()
            os.remove(error_path) # get rid of the error_path once we're done interpolate the images.

        self.logger.info("Finished destriping images in folder: {}".format(self.input_path))
        self.logger.info("Time taken: {}".format(time.time() - start))

def _parse_args(raw_args=None):
    parser = argparse.ArgumentParser(description="Pystripe\n\n"
        "If only sigma1 is specified, only foreground of the images will be filtered.\n"
        "If sigma2 is specified and sigma1 = 0, only the background of the images will be filtered.\n"
        "If sigma1 == sigma2 > 0, input images will not be split before filtering.\n"
        "If sigma1 != sigma2, foreground and backgrounds will be filtered separately.\n"
        "The crossover parameter defines the width of the transistion between the filtered foreground and background",
                                     formatter_class=RawDescriptionHelpFormatter,
                                     epilog='Developed by LifeCanvas Technologies.\n')
    parser.add_argument("--input", "-i", help="Path to input image or path", type=str, required=True)
    parser.add_argument("--output", "-o", help="Path to output image or path (Default: x_destriped)", type=str, required=True)
    parser.add_argument("--sigma1", "-s1", help="Foreground bandwidth [pixels], larger = more filtering", type=float, default=0)
    parser.add_argument("--sigma2", "-s2", help="Background bandwidth [pixels] (Default: 0, off)", type=float, default=0)
    parser.add_argument("--level", "-l", help="Number of decomposition levels (Default: max possible)", type=int, default=0)
    parser.add_argument("--wavelet", "-w", help="Name of the mother wavelet (Default: Daubechies 3 tap)", type=str, default='db3')
    parser.add_argument("--threshold", "-t", help="Global threshold value (Default: -1, Otsu)", type=float, default=-1)
    parser.add_argument("--crossover", "-x", help="Intensity range to switch between foreground and background (Default: 10)", type=float, default=10)
    parser.add_argument("--extra-smoothing", help="Magnitude of smoothing between foreground and background (Default: 1)", type=parse_extra_smoothing, default=1)
    parser.add_argument("--workers", help="Number of parallel processes to use for all aspects of destriping (particularly I/O) (Default: Number of logical cores)", type=int, default=None)
    parser.add_argument("--ram-loadsize", help="Number of images to load at once on RAM GPU (Default: None)", type=int, default=None)
    parser.add_argument("--gpu-chunksize", help="Number of images to destripe at once in the GPU (Default: None)", type=int, default=None)
    parser.add_argument("--compression", "-c", help="Compression level for written tiffs (Default: 1)", type=int, default=1)
    parser.add_argument("--flat", "-f", help="Flat reference TIFF image of illumination pattern used for correction", type=str, default=None)
    parser.add_argument("--dark", "-d", help="Intensity of dark offset in flat-field correction", type=float, default=0)
    parser.add_argument("--dont-convert-16bit", help="Is the output converted to 16-bit .tiff or not", action="store_true")
    parser.add_argument("--output-format", "-of", help="Desired format output for the images", type=str, required=False, default=None)
    parser.add_argument('--log-path',type=str,required=False, default=None, help="path to the logs for postprocessing")
    parser.add_argument('--rotate-and-flip', action="store_true")
    parser.add_argument('--extra-rotate-and-flip', action="store_true")
    args = parser.parse_args(raw_args)
    return args

def parse_extra_smoothing(arg_extra_smoothing: str) -> Union[int, bool, float]:
    if arg_extra_smoothing.lower() == 'true':
        return True
    elif arg_extra_smoothing.lower() == 'false':
        return False
    else:
        try:
            return float(arg_extra_smoothing)
        except ValueError:
            msg = f"Invalid value for extra_smoothing: {arg_extra_smoothing}. Must be a float/int or 'True'/'False'"
            raise argparse.ArgumentTypeError(msg)

def main(raw_args=None):
    args = _parse_args(raw_args)
    sigma = [args.sigma1, args.sigma2]

    if args.sigma1 == args.sigma2:
        args.prepared_threshold = 0

    input_path = Path(args.input)
    output_path = Path(args.output)

    flat = None
    if args.flat is not None:
        flat = normalize_flat(imread(args.flat))
    if args.dark < 0:
        raise ValueError('Only positive values for dark offset are allowed')

    if args.output_format is not None and args.output_format not in supported_output_extensions:
        raise ValueError(f"Output format {args.output_format} is currently not supported! Supported formats are: {supported_output_extensions}")
    elif args.output_format is None:
        output_format = ".tiff"
    else:
        output_format = args.output_format
    
    
    if args.log_path is not None:
        log_path = args.log_path
        logger = get_logger(log_path)
    else:
        logger = get_default_logger()
    

    # Make output path if it doesn't exist
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        
    destriper = Destriper(input_path,
                        output_path,
                        sigma=sigma,
                        level=args.level,
                        wavelet=args.wavelet,
                        crossover=args.crossover,
                        threshold=args.threshold,
                        compression=args.compression,
                        flat=flat,
                        dark=args.dark,
                        num_workers=args.workers,
                        ram_loadsize=args.ram_loadsize,
                        gpu_chunksize=args.gpu_chunksize,
                        dont_convert_16bit=args.dont_convert_16bit,
                        output_format=output_format,
                        logger = logger,
                        rotate_and_flip = args.rotate_and_flip,
                        extra_rotate_and_flip = args.extra_rotate_and_flip
                        )
    destriper.batch_filter()

if __name__ == "__main__":
    main()