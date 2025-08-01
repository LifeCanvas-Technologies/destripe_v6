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
from collections import namedtuple, OrderedDict, Counter

from . import supported_output_extensions
from .utils import find_all_images, interpolate, imread, imsave, get_extension, attempt_read_threshold, threshold_img
from .filter_utils import normalize_flat, gaussian_filter


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
                cpu_readers : Optional[int] = None,
                io_readers : Optional[int] = None,
                ram_loadsize : Optional[int] = None,
                gpu_chunksize : Optional[int] = None,
                auto_mode : Optional[bool] = False,
                # rotate : bool = False,
                # post_rotate_flip : bool = False,
                extra_smoothing : int = 1,
                timeprint : bool = False,
                dont_convert_16bit : bool = False,
                output_format : Optional[str] = None):
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
        auto_mode : bool
            If True, doing live destriping 
        io_readers: int
            Maximum number of threads to execute
            :py:attr:`FileSequence.imread` asynchronously.
            If *None*, default to processors multiplied
            by 5.
            Using threads can significantly improve runtime when reading
            many small files from a network share.
        # rotate : bool
        #     Flag for 90 degree rotation.
        # post_rotate_flip : bool
        #     Flag for 90 degree rotation and flip.
        dont_convert_16bit : bool
            Flag for converting to 16-bit
        output_format: str
            Desired output format [.tiff, .tif]. Default None
        extra_smoothing : int
            Whether extra gaussian smoothing is applied to the foreground fraction.
            calculation. If so, can be an int to specify the magnitude of doing so.
        tiemprint : bool
            flag for printing out time taken for each batch step
        """
        self.input_path = input_path
        self.output_path = output_path
        self.sigma = sigma 
        self.level = level
        self.wavelet = wavelet
        self.crossover = crossover
        self.provided_threshold = threshold
        self.compression = compression
        self.flat = flat
        self.dark = dark
        # self.rotate = rotate
        # self.post_rotate_flip = post_rotate_flip
        self.cpu_readers = cpu_readers if cpu_readers is not None else self.num_cpu_readers()
        self.io_readers = io_readers if io_readers is not None else self.num_io_readers(self.cpu_readers)
        self.ram_loadsize = ram_loadsize
        self.gpu_chunksize = gpu_chunksize 
        self.auto_mode = auto_mode 

        self.dont_convert_16bit = dont_convert_16bit
        self.output_format = output_format
        self.extra_smoothing = extra_smoothing
        self.timeprint = timeprint


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
    
    @staticmethod
    def num_io_readers(specified_cpu_readers=None):
        if specified_cpu_readers:
            return specified_cpu_readers + 4
        elif os.cpu_count() <= 16:
            return 8
        elif os.cpu_count() <= 32:
            return 12
        elif os.cpu_count() <= 64:
            return 16
        else:
            return int(0.25 * os.cpu_count)

    def _single_read(self, args): 
        # For non-Tiffs or in the event of a corrupted image.
        n = 3
        input_path = args['input_path'] # path to the input image, not the overall input path
        for i in range(n):
            try:
                img = imread(str(input_path))
                # if self.rotate:
                #     return np.rot90(img)
                if args["threshold"] < 0:
                    return img, threshold_otsu
                else:
                    return img, args["threshold"]

            except:
                if i == n-1:
                    file_name = os.path.join(self.output_path, 'destripe_log.txt')
                    if not os.path.exists(file_name):
                        error_file = open(file_name, 'w')
                        error_file.write('Error reading the following images.  Pystripe will interpolate their content.')
                        error_file.close()
                    error_file = open(file_name, 'a+')
                    error_file.write('\n{}'.format(str(input_path)))
                    error_file.close()
                    return None, None
                else:
                    time.sleep(0.05)
                    continue

    # def prepare_batch(self, args_batch):
    #     imgs_torch, args_batch = self._prepare_batch(args_batch)
    #     assert(imgs_torch.size(0) == len(args_batch))
    
    def prepare_batch(self, args_batch):
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

        
        thresholds = [args['threshold'] for args in args_batch]
        tiff_turbo = (len(args_batch) > 1 and
                      all(get_extension(args['input_path']) in ('.tiff', '.tif') for args in args_batch) and all(th >= 0 for th in thresholds) and
                      self.io_readers not in [0, 1])

        file_paths = [str(args['input_path']) for args in args_batch]
        if tiff_turbo:
            num_attempts = min(3, len(args_batch))
            for _ in range(num_attempts):
                imgs = tifffile.imread(files=file_paths, ioworkers=self.io_readers)
                # try:
                #     imgs = list(tifffile.imread(files=file_paths, io_readers=self.io_readers))
                # except:
                #     continue

                if any(th < 0 for th in thresholds):
                    with multiprocessing.Pool(self.cpu_readers) as p:
                        thresholds = tuple(p.imap(threshold_otsu, list(imgs)))
                    # for args in args_batch:
                    #     args['threshold'] = thresholds.pop(0)
                break
            else:
                print('A batch failed to load via tifffile multithreading. Files will instead be loaded via multiprocessing')
                tiff_turbo = False

        if tiff_turbo is False:
            if len(args_batch) > 1 and self.cpu_readers not in [0, 1]:
                with multiprocessing.Pool(self.cpu_readers) as p:  # use multiprocessing for reading PNGs and RAWs
                    inputs_paths = [str(args['input_path']) for args in args_batch]
                    # imgs, thresholds = zip(*p.starmap(attempt_read_threshold, zip(inputs_paths, thresholds)))

                    # Option: use imap 
                    f = partial(attempt_read_threshold, threshold_prompt=self.provided_threshold)
                    # imgs_and_thresholds = list(tqdm.tqdm(p.imap(f, inputs_paths), total=len(inputs_paths)))
                    imgs_and_thresholds = list(p.imap(f, inputs_paths))
                    imgs, thresholds = zip(*imgs_and_thresholds)
            else:
                imgs, thresholds = zip(*[self._single_read(args) for args in args_batch])
        
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
        gpu_mem_for_use, total_gpu_memory = torch.cuda.mem_get_info()

        # approx_rtx4090_mem = 24125636608 # gpu_mem_for_use from RTX 4090 
        trial_maxdim = 2048
        trial_nimgs = 64
        trial_x = trial_nimgs * (trial_maxdim**2)  # 64 2048x2048 imgs
        #  2048^2 * 64 * k <= 24 Gigs
        k = total_gpu_memory / trial_x

        batch_size = np.floor(gpu_mem_for_use/(k * max(img_dims)**2))
        # print(batch_size)
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

        # Rotates and flips the images for deskewing purposes
        # if self.post_rotate_flip:
        #     fimgs = torch.rot90(fimgs, k=1, dims=(1, 2))
        #     fimgs = torch.flip(fimgs, dims=(1,2))
        
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


    def _prep_threshold(self, imgs_chunk, args_chunk):

        """
        Preprocesses the threshold value for destriping.

        Args:
            imgs_chunk (torch.Tensor): Batch of input images.
            args_chunk (dict): Batch of input arguments.


        Returns:
            threshold (float): threshold to use for this batch 
        """
        # TODO: figure out better way to determine threshold for each image

        assert(len(set(args['threshold'] for args in args_chunk)) == 1), "Batch loading can only be done on inputs with the same `threshold`. If threshold is set as -1, then a threshold for a batch will be calculated"

        if args_chunk[0]["threshold"] < 0:
            mid_idx = int(imgs_chunk.size(0)/2)
            mid_img = imgs_chunk[mid_idx].numpy()
            if mid_img.dtype == np.int16:
                mid_img = mid_img.astype(np.uint16, copy=True)
            try:
                threshold = threshold_otsu(mid_img)
            except ValueError:
                threshold = 1
            return threshold 
        else:
            return args_chunk[0]["threshold"]
                        
    def torch_imwrite(self, imgs_chunk, args_chunk):
        """
        Write a batch of images to disk using the TIFF format.

        Args:
            imgs_chunk (torch.Tensor): Batch of images to be written.
            args_chunk (list): List of arguments for each image in the batch.


        Returns:
            None
        """
        images = imgs_chunk.numpy()
        if images.dtype == np.int16:
            images.dtype = np.uint16
        
        for img, args in zip(images, args_chunk):
            out_path = str(args['output_path'])
            # tifffile.imwrite(out_path, img)#, compression='zlib', compressionargs={'level': self.compression})
            imsave(out_path, img, compression=self.compression, output_format=self.output_format, rotate_and_flip=False)

    # def torch_imwrite2(self, imgs_chunk, args_chunk):
    #     """
    #     Write a batch of images to disk using the TIFF format.

    #     Args:
    #         imgs_chunk (torch.Tensor): Batch of images to be written.
    #         args_chunk (list): List of arguments for each image in the batch.


    #     Returns:
    #         None
    #     """
    #     images = imgs_chunk.numpy()
    #     if images.dtype == np.int16:
    #         images.dtype = np.uint16
        
    #     def f(img_and_path, compression=self.compression, output_format=self.output_format, rotate_and_flip=False):
    #         imgarr, imgpath = img_and_path
    #         out_path = str(imgpath)
    #         imsave(out_path, imgarr, compression=self.compression, output_format=self.output_format, rotate_and_flip=False)

    #     img_and_paths = zip(images, [args['output_path'] for args in args_chunk])
        
    #     with multiprocessing.Pool(processes=16) as p:
    #        p.imap(partial(f, compression=self.compression, output_format=self.output_format, rotate_and_flip=False), img_and_paths)



    def batch_filter(self):
        # Error logs path
        error_path = os.path.join(self.output_path, 'destripe_log.txt')
        if os.path.exists(error_path):
            os.remove(error_path)

        # Find all the images in the input path to be destriped
        print('Looking for images in {}...'.format(self.input_path))
        img_paths = find_all_images(self.input_path, self.input_path, self.output_path)
        print('Found {} compatible images'.format(len(img_paths)))

        if self.auto_mode:
            img_path_strs = list(str(path) for path in img_paths)
            list_path = os.path.join(self.output_path, 'destriped_image_list.txt')
            # print('writing image_list.  {} images'.format(len(img_path_strs)))
            with open(list_path, 'w') as fp:
                fp.write('\n'.join(img_path_strs) + '\n')
                fp.close

        # Get the image size to set the ram_loadsize parameter (if None)
        if self.ram_loadsize is None:
            img_size = imread(str(img_paths[0])).shape
            self.ram_loadsize = self.set_ram_load_size(img_size)

        # Get the image size to set the gpu_chunksize parameter (if None)
        if self.gpu_chunksize is None:
            img_size = imread(str(img_paths[0])).shape
            self.gpu_chunksize = self.set_gpu_batch_size(img_size)
                
        # copy text and ini files
        for file in self.input_path.iterdir():
            if Path(file).suffix in ['.txt', '.ini']:
                output_file = os.path.join(self.output_path, os.path.split(file)[1])
                shutil.copyfile(file, output_file)
        
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
        
        batch_counter = 0
        chunk_counter = 0

        with tqdm.tqdm(total=(len(args)), ascii=True, bar_format=None) as pbar:

            for load_batch in load_args:
                # print('Loading batch #{}'.format(batch_counter))
                if True:#with multiprocessing.Pool(processes=16) as p:
                    def torch_imwrite3( imgs_chunk, args_chunk):
                        """
                        Write a batch of images to disk using the TIFF format.

                        Args:
                            imgs_chunk (torch.Tensor): Batch of images to be written.
                            args_chunk (list): List of arguments for each image in the batch.


                        Returns:
                            None
                        """
                        images = imgs_chunk.numpy()
                        if images.dtype == np.int16:
                            images.dtype = np.uint16
                        
                        # def f(img_and_path, compression=self.compression, output_format=self.output_format, rotate_and_flip=False):
                        #     imgarr, imgpath = img_and_path
                        #     out_path = str(imgpath)
                        #     imsave(out_path, imgarr, compression=self.compression, output_format=self.output_format, rotate_and_flip=False)

                        img_and_paths = zip(images, [args['output_path'] for args in args_chunk])
                        
                        # with multiprocessing.Pool(processes=16) as p:
                        # p.imap(partial(f, compression=self.compression, output_format=self.output_format, rotate_and_flip=False), img_and_paths)

                        for img_and_path in img_and_paths:
                            imgarr, imgpath = img_and_path
                            out_path = str(imgpath)
                            imsave(out_path, imgarr, compression=self.compression, output_format=self.output_format, rotate_and_flip=False)



                    if self.timeprint: tic = time.time()
                    chunked = self.prepare_batch(load_batch)
                    if self.timeprint: print("Batch #{} loading time: {}".format(batch_counter, time.time() - tic))
                    # imgs_args = list(zip(read_imgs, args_batch))
                    # process_chunks = list(mit.chunked_even(imgs_args, self.gpu_chunksize))
                        
                    last_args_chunk = None
                    last_imgs_chunk = None
                    for imgs_chunk, args_chunk in chunked:
                        # if self.timeprint: tic = time.time()
                        # imgs_chunk = self.batch_to_torch16(args_chunk)
                        # if self.timeprint: print("Batch {} loading time: {}".format(counter, time.time() - tic))

                        # if self.provided_threshold is None or self.provided_threshold < 0:
                        #     if self.timeprint: tic = time.time()
                        #     threshold = self._prep_threshold(imgs_chunk, args_chunk)
                        #     if self.timeprint: print("Threshold chunk # {} prep time: {}".format(chunk_counter, time.time() - tic))

                        if self.timeprint: tic = time.time()
                        imgs_chunk = imgs_chunk.to(device='cuda', non_blocking=False)
                        images32 = self.offsign16_to_32(imgs_chunk)
                        imgs_chunk32 = self.destripe_torch32(images32, args_chunk)

                        imgs_chunk = imgs_chunk32.to(device='cpu', dtype=torch.int16, non_blocking=True)
                        if self.timeprint: print("Destriping chunk #{} time: {}".format(chunk_counter, time.time() - tic))

                        # if self.post_rotate_flip:
                        #     imgs_chunk = torch.rot90(imgs_chunk, k=1, dims=(1, 2))
                        #     imgs_chunk = torch.flip(imgs_chunk, dims=(1,2))

                        if last_imgs_chunk is not None:
                            if self.timeprint: tic = time.time()
                            torch.cuda.synchronize()
                            torch_imwrite3(last_imgs_chunk, last_args_chunk)
                            if self.timeprint: print("Writing chunk #{} time: {}".format(chunk_counter, time.time() - tic))
                            pbar.update(len(last_args_chunk))
                            # del last_imgs_chunk; del last_args_chunk

                        last_imgs_chunk = imgs_chunk
                        last_args_chunk = args_chunk
                        chunk_counter += 1
                    else:
                        if self.timeprint: tic = time.time()
                        torch.cuda.synchronize()
                        torch_imwrite3(last_imgs_chunk, last_args_chunk)
                        if self.timeprint: print("Writing chunk #{} time: {}".format(chunk_counter, time.time() - tic))
                        pbar.update(len(last_args_chunk))
                        # del last_imgs_chunk; del last_args_chunk
                    batch_counter += 1
                # print('Done!')

                # Interpolate images that could not be opened
                if os.path.exists(error_path):
                    with open(error_path, 'r') as fp:
                        first_line = fp.readline()  # seems to purposefully not use this line
                        images = fp.readlines()
                        for image_path in images:
                            interpolate(image_path, self.input_path, self.output_path)
                        x = len(images)
                        print('{} images could not be opened and were interpolated.  See destripe log for more details'.format(x))
                        fp.close()


def _parse_args(raw_args=None):
    parser = argparse.ArgumentParser(description="Pystripe\n\n"
        "If only sigma1 is specified, only foreground of the images will be filtered.\n"
        "If sigma2 is specified and sigma1 = 0, only the background of the images will be filtered.\n"
        "If sigma1 == sigma2 > 0, input images will not be split before filtering.\n"
        "If sigma1 != sigma2, foreground and backgrounds will be filtered separately.\n"
        "The crossover parameter defines the width of the transistion between the filtered foreground and background",
                                     formatter_class=RawDescriptionHelpFormatter,
                                     epilog='Developed 2018 by Justin Swaney, Kwanghun Chung Lab\n'
                                            'Massachusetts Institute of Technology\n')
    parser.add_argument("--input", "-i", help="Path to input image or path", type=str, required=True)
    parser.add_argument("--output", "-o", help="Path to output image or path (Default: x_destriped)", type=str, default='')
    parser.add_argument("--sigma1", "-s1", help="Foreground bandwidth [pixels], larger = more filtering", type=float, default=0)
    parser.add_argument("--sigma2", "-s2", help="Background bandwidth [pixels] (Default: 0, off)", type=float, default=0)
    parser.add_argument("--level", "-l", help="Number of decomposition levels (Default: max possible)", type=int, default=0)
    parser.add_argument("--wavelet", "-w", help="Name of the mother wavelet (Default: Daubechies 3 tap)", type=str, default='db3')
    parser.add_argument("--threshold", "-t", help="Global threshold value (Default: -1, Otsu)", type=float, default=-1)
    parser.add_argument("--crossover", "-x", help="Intensity range to switch between foreground and background (Default: 10)", type=float, default=10)
    parser.add_argument("--extra-smoothing", help="Magnitude of smoothing between foreground and background (Default: 1)", type=parse_extra_smoothing, default=1)
    parser.add_argument("--cpu-readers", help="Number of CPU processors for multiprocessed PNG reading and/or imputed thresholding . Only matters for PNG inputs (Default: None)", type=int, default=None)
    parser.add_argument("--io-readers", help="Number of IO workers for multi-threaded TIFF reading. Only matters for TIFF inputs (Default: None)", type=int, default=None)
    parser.add_argument("--ram-loadsize", help="Number of images to load at once on RAM GPU (Default: None)", type=int, default=None)
    parser.add_argument("--gpu-chunksize", help="Number of images to destripe at once in the GPU (Default: None)", type=int, default=None)
    parser.add_argument("--auto-mode", help="Do live destriping", action='store_true')
    parser.add_argument("--compression", "-c", help="Compression level for written tiffs (Default: 1)", type=int, default=1)
    parser.add_argument("--flat", "-f", help="Flat reference TIFF image of illumination pattern used for correction", type=str, default=None)
    parser.add_argument("--dark", "-d", help="Intensity of dark offset in flat-field correction", type=float, default=0)
    # parser.add_argument("--rotate", "-r", help="Rotate output images 90 degrees counter-clockwise", action='store_false')
    # parser.add_argument("--post_rotate_flip", help="After de-striping, rotate output images 90 degrees (counter-clockwise) and flip (for de-skewing purposes)", action='store_true')
    parser.add_argument("--dont-convert-16bit", help="Is the output converted to 16-bit .tiff or not", action="store_true")
    parser.add_argument("--output-format", "-of", help="Desired format output for the images", type=str, required=False, default=None)
    parser.add_argument('--log-path',type=str,required=False, default=None, help="path to the logs for postprocessing")
    parser.add_argument('--timeprint', help="print time taken for each batch step", action='store_true')
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
    # print("Starting GPU destriping...")
    args = _parse_args(raw_args)
    sigma = [args.sigma1, args.sigma2]

    if args.sigma1 == args.sigma2:
        args.prepared_threshold = 0

    input_path = Path(args.input)

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
        
    if input_path.is_dir():  # batch processing
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(str(input_path)+'_destriped')
        else:
            output_path = Path(args.output)
            assert output_path.suffix == ''
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
                            # rotate=args.rotate, ### PRE-ROTATION parameter NOT NEEDED at LCT. Do not confuse with post-rotation
                            # post_rotate_flip=args.post_rotate_flip,
                            cpu_readers=args.cpu_readers,
                            io_readers=args.io_readers,
                            ram_loadsize=args.ram_loadsize,
                            gpu_chunksize=args.gpu_chunksize,
                            auto_mode = args.auto_mode,
                            dont_convert_16bit=args.dont_convert_16bit,
                            output_format=output_format,
                            timeprint=args.timeprint
                            )
        destriper.batch_filter()
    else:
        print('Cannot find input file or directory. Exiting...')


if __name__ == "__main__":
    main()