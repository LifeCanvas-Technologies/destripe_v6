import os, time, csv, json
import math
from pathlib import Path
from datetime import datetime
import shutil
from tqdm import tqdm
import math
from tabulate import tabulate
from PIL import Image
import argparse
import psutil
import glob

from live_destriper.utils import get_spim_type, get_txt_encoding
from live_destriper.logger import get_logger

def parse_args(raw_args=None):
    """
    Specify which folder to perform live destriping on.
    """
    parser = argparse.ArgumentParser(description="Get what destriping parameters to use and which path to perform live destriping on.")
    parser.add_argument("--input", "-i", help="Path to overall input folder", type=str, required=True)
    parser.add_argument("--output", "-o", help="Path to output image or path (Default: x_destriped)", type=str, required=True)
    parser.add_argument("--sigma1", "-s1", help="Foreground bandwidth [pixels], larger = more filtering", type=float, default=256)
    parser.add_argument("--sigma2", "-s2", help="Background bandwidth [pixels] (Default: 0, off)", type=float, default=0)
    parser.add_argument("--num-workers", help="Number of parallel processes to use for all aspects of destriping (particularly I/O) (Default: Number of physical cores)", type=int, default=psutil.cpu_count(logical=False))
    parser.add_argument("--ram-loadsize", help="Number of images to load at once on RAM GPU (Default: None, infer from available RAM)", type=int, default=None)
    parser.add_argument("--gpu-chunksize", help="Number of images to destripe at once in the GPU (Default: None, infer from available vRAM)", type=int, default=None)
    parser.add_argument('--log-path',type=str,required=False, default=None, help="path to the logs for postprocessing")
    parser.add_argument("--use-gpu", action="store_true", help="Whether to use GPU or not when destriping")

    # Options unique to live destriping
    parser.add_argument("--check-corrupt", action="store_true", help="Whether to check for corrupt images after a stack is finished destriping.")
    parser.add_argument("--metadata-version", type=int, default=5, required=False, help="Which version of metadata to read from.")
    parser.add_argument("--stall-limit", required=False, type=int, default=60, help="How much time (in seconds) we wait until we error out (potentially acquisition failed)")
    args = parser.parse_args(raw_args)
    return args

def check_for_bad_images(path):
    print('\nChecking for corrupt files...')
    count = 0
    start = datetime.now()
    for filename in tqdm(os.listdir(path)):
        # print('img path: {}'.format(os.path.join(path, filename)))
        try:
            img = Image.open(os.path.join(path, filename))
            img.verify()
            # img = cv2.imread(os.path.join(directory, filename))
            
        except:
            print("Bad file: {}".format(filename))
            count += 1
    time = datetime.now() - start
    print("{} corrupt files found.".format(count, time.seconds))
    return count
    

def run_pystripe(input_path, 
                 output_path,
                 sigma1 = 256,
                 sigma2 = 0,
                 use_gpu = True,
                 gpu_chunksize = None, 
                 ram_loadsize = None, 
                 num_workers = psutil.cpu_count(logical=False),
                 rotate_and_flip = False,
                 extra_rotate_and_flip = False,
                 log_path = None,
                 check_corrupt = False):
    if use_gpu:
        print("Using GPU Destriper")
        from live_destriper.destripe.core_gpu import main as gpu_destripe
        cmd = ["-i", str(input_path),
                        "-o", str(output_path), 
                        "--sigma1", str(sigma1),
                        "--sigma2", str(sigma2),
                        "--workers", str(num_workers)]
        if ram_loadsize is not None:
            cmd.append("--ram-loadsize")
            cmd.append(str(ram_loadsize))
        if gpu_chunksize is not None:
            cmd.append("--gpu-chunksize")
            cmd.append(str(gpu_chunksize))
        if log_path is not None:
            cmd.append("--log-path")
            cmd.append(str(log_path))
        if rotate_and_flip:
            cmd.append("--rotate-and-flip")
        if extra_rotate_and_flip:
            cmd.append("--extra-rotate-and-flip")
        print(cmd)
        
        gpu_destripe(cmd)        

    else:
        print("Using CPU Destriper")
        from live_destriper.destripe.core import main as cpu_destripe
        cpu_destripe(["-i", str(input_path),
                        "-o", str(output_path), 
                        "--sigma1", str(sigma2),
                        "--sigma2", str(sigma2),
                        "--workers", str(num_workers)])
    
    if 'MIP' in input_path or not check_corrupt:
        # Finish up. 
        return

    corrupted = check_for_bad_images(output_path)
    if corrupted > 0:
        print('{} corrupt images found in {}.  This folder is being re-destriped'.format(corrupted, output_path))
        run_pystripe(input_path,
                     output_path,
                     sigma1 = sigma1,
                     sigma2 = sigma2,
                     use_gpu = use_gpu,
                     gpu_chunksize = gpu_chunksize,
                     ram_loadsize = ram_loadsize,
                     num_workers = num_workers,
                     rotate_and_flip = rotate_and_flip,
                     extra_rotate_and_flip = extra_rotate_and_flip,
                     log_path = log_path,
                     check_corrupt = check_corrupt)
        

def get_metadata(input_path):
    # builds metadata dict
    metadata_path = input_path / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def pair_key_value_lists(keys, values):
    # utility function for building metadata dict

    d = {}
    for i in range(0, len(keys)):
        key = keys[i]
        val = values[i]
        if key != '':
            d[key] = val
    return d

def get_target_number(z_block, z_step):
    # Calculates number of images per tile
    try:
        steps_per_tile = max(math.ceil(float(z_block) / float(z_step)), 1) # used to have -1 but this is incorrect
    except:
        steps_per_tile = 1
    return steps_per_tile

def parse_destripe_tag(tag):
    status = 'true'
    for i in 'NCDA':
        if i in tag:
            status = 'false'
            tag = tag.replace(i, '')
    return status, tag

def get_metadata_v5(input_path):
    # builds metadata dict
    metadata_path = input_path / 'metadata.txt'

    metadata_dict = {
        'channels': [],
        'tiles': []
    }
    sections = {
        'channel_vals': [],
        'tile_vals': []
    }
    with open(metadata_path, encoding="utf8", errors="ignore") as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        section_num = 0
        for row in reader:
            if section_num == 0:
                sections['gen_keys'] = row
                section_num += 1
                continue
            if section_num == 1:
                sections['gen_vals'] = row
                section_num += 1
                continue
            if section_num == 2:
                sections['channel_keys'] = row
                section_num += 1
                continue
            if section_num == 3:
                if row[0] != 'X':
                    sections['channel_vals'].append(row)
                    continue
                else:
                    sections['tile_keys'] = row
                    section_num += 2
                    continue
            if section_num == 5:
                sections['tile_vals'].append(row)

    d = pair_key_value_lists(sections['gen_keys'], sections['gen_vals'])
    target_number = get_target_number(d['Z_Block'], d['Z step (m)'])
    d['destripe_status'], d['destripe'] = parse_destripe_tag(d['Destripe'])
    metadata_dict.update({'sample metadata': d})

    for channel in sections['channel_vals']:
        d = pair_key_value_lists(sections['channel_keys'], channel)
        metadata_dict['channels'].append(d)

    for tile in sections['tile_vals']:
        d = pair_key_value_lists(sections['tile_keys'], tile)
        d['NumImages'] = 1
        if int(d['Skip']) == 1:
            d['NumImages'] = target_number
        metadata_dict['tiles'].append(d)
    return metadata_dict

def get_megaspim_metadata(input_path):
    metadata_path = input_path / "metadata.json" #
    # TODO: Fill this out
    pass

def count_tiles(input_path, output_path, metadata, metadata_version):
    tiles = []
    for tile in metadata['tiles']:
        expected = int(tile['NumImages'])
        laser = tile['Laser']
        filter = tile['Filter']
        x = tile['X']
        y = tile['Y']
        
        if metadata_version==0:
            print("MegaSPIM")
            pass 
        elif metadata_version==6:
            ch = tile['FilterChannel']
            tile_path = os.path.join('Ex_{}_Em_{}_Ch{}'.format(laser, filter, ch), x, '{}_{}'.format(x, y))
        else:
            tile_path = os.path.join('Ex_{}_Ch{}'.format(laser, filter), x, '{}_{}'.format(x, y))

        # input_images = len(os.listdir(os.path.join(input_path, tile_path))) # This gets all files which may include other stuff
        input_images = len(glob.glob(os.path.join(input_path, tile_path, "*.tif*")) +\
                           glob.glob(os.path.join(input_path, tile_path, "*.png")) +\
                           glob.glob(os.path.join(input_path, tile_path, "*.raw")))
        try:
            output_images = len(glob.glob(os.path.join(output_path, tile_path, "*.tif*")))
        except:
            output_images = 0      
        tiles.append({
            'path': tile_path,
            'input_images': input_images,
            'output_images': output_images,
            'expected': expected
        })
    return tiles

def show_output(tiles, input_path):
    headers = ['Tile', 'Images Expected', 'Images on Acquisition Drive', 'Images on Stitch Drive']
    data = []
    total_images = 0
    total_destriped = 0
    for tile in tiles: # list of dictionaries of tile dictionaries
        total_images += tile['expected']
        total_destriped += tile['output_images']
        data.append([
            tile['path'],
            tile['expected'],
            tile['input_images'],
            tile['output_images']
        ])
    print('Current Acquisition: {}\n'.format(input_path))
    print(tabulate(data, headers))
    pct = total_destriped / total_images
    bar_length = 72
    print('\nOVERALL DESTRIPING PROGRESS: {:.0%} [{}{}]'.format(pct, '#'*round(pct*bar_length), '-'*round((1-pct)*bar_length)))
    
def check_mips(raw_dir, destriped_dir, tiles, destripe_list_path, **kwargs):
    """
    Check MIP stacks in case we have to destripe these too. 

    **kwargs are kwargs for pystripe
    """
    # First check to see if MIPs are even created:
    has_MIPS = False
    for item in os.listdir(raw_dir):
        if "MIP" in item and os.path.isdir(os.path.join(raw_dir, item)):
            has_MIPS = True
            break

    if has_MIPS:
        for tile in tiles: # We get the names of which MIPs should have been created
            tile_path = tile["path"]
            channel_mip_name = tile_path.split("/")[0] + "_MIP"
            item = f'{channel_mip_name}/{"/".join(tile_path.split("/")[1:])}'

            input_path = os.path.join(raw_dir, item)
            output_path = os.path.join(destriped_dir, item)
            try:
                output_images = len(glob.glob(os.path.join(output_path, "*.tif*")))
            except:
                output_images = 0

            input_images = len(glob.glob(os.path.join(input_path, "*.tif*")) +\
                        glob.glob(os.path.join(input_path, "*.png")) +\
                        glob.glob(os.path.join(input_path, "*.raw")))
            if input_images != output_images:
                print('\nDestriping {}...\n'.format(item))
                
                run_pystripe(input_path, output_path, use_gpu = False,
                            **kwargs)
                update_destripe_list_txt(input_path, destripe_list_path)
    else:
        print("No MIP folders found in acquisition, skipping...")

def finish_directory(input_path, output_path):
    """
    Move the .txt .ini and .json files to the new directory.
    """
    for file in Path(input_path).iterdir():
        file_name = os.path.split(file)[1]
        if Path(file).suffix in ['.txt', '.ini', '.json']:
            output_file = os.path.join(Path(output_path), file_name)
            shutil.copyfile(file, output_file)

def create_destripe_list_txt(input_dir, destripe_list_path, tiles, metadata_version):
    """
    Based on the metadata, we create the list of folders that need to be destriped still. 
    """
    with open(destripe_list_path, "w") as f:
        for tile in tiles:
            relative_path = tile["path"]
            full_path = input_dir / relative_path
            f.write(str(full_path) + "\n")

            if metadata_version != 0:
                # That means it's SmartSPIM, which means we have MIPs
                channel_mip_name = relative_path.split("/")[0] + "_MIP"
                item = f'{channel_mip_name}/{"/".join(relative_path.split("/")[1:])}'
                full_mip_path = input_dir / item
                f.write(str(full_mip_path) + "\n")


def update_destripe_list_txt(input_path, destripe_list_path):
    # Now remove the folder from the destripe_list
    encoding = get_txt_encoding(destripe_list_path)
    with open(destripe_list_path, 'r+', encoding=encoding) as fp:
        lines = fp.readlines()  # Read all lines into a list
        lines = [line for line in lines if str(input_path) not in line]
        
        fp.seek(0)  # Move the file pointer to the beginning
        fp.writelines(lines)  # Write the updated lines back to the file
        fp.truncate()  # Remove any remaining content beyond the new end of file


def main():
    args = parse_args()
    raw_dir = Path(args.input)
    destriped_dir = Path(args.output)

    if args.log_path is None:
        log_path = os.path.join(destriped_dir, "logs", "postprocessing_logs_%s.txt"%(str(datetime.now()).replace(' ','_').replace(':',',')))
    else:
        log_path = args.log_path

    logger = get_logger(log_path)
    logger.info(f"Starting destriping-on-the-fly on {raw_dir}, saving to {destriped_dir}")
    tic = time.time()
        
    stall_counter = ['', 0, 0]

    metadata_version = args.metadata_version

    # Get the spim type ot determine if we have to rotate and flip the destriped images
    # Get whether we should do an extra rotation and flip if it's a Fire camera
    metadata_path = raw_dir / "metadata.txt"
    spim_type = get_spim_type(metadata_path)
    extra_rotate_and_flip = False # This is a different rotation and flipping for MegaSPIM Fire camera
    rotate_and_flip = False # This is for standard destriping
    if spim_type == "Fire":
        rotate_and_flip = True
        extra_rotate_and_flip = True
    elif spim_type == "MegaSPIM" or spim_type == "SacSPIM":
        rotate_and_flip = True

    if spim_type != "SmartSPIM":
        metadata_version = 0
        # Then it is Dali or MegaSPIM
        metadata = get_megaspim_metadata(raw_dir) # TODO: write this functino for reading megaspim_metadata
    elif args.metadata_version==6:
        metadata = get_metadata(raw_dir)
    else:
        metadata = get_metadata_v5(raw_dir)

    # we also want to write to a destripe_folder_list.txt to keep track of the un-destriped stacks in case this fails. Then, the normal destriping can
    # take care of this
    destripe_list_path = destriped_dir / "destripe_folder_list.txt"
    if not destripe_list_path.exists():
        tiles = count_tiles(raw_dir, destriped_dir, metadata, metadata_version)
        create_destripe_list_txt(raw_dir, destripe_list_path, tiles, metadata_version)

    
    # Make params dictionary
    destripe_params = {
            "sigma1": args.sigma1,
            "sigma2": args.sigma2,
            "gpu_chunksize": args.gpu_chunksize,
            "ram_loadsize": args.ram_loadsize,
            "num_workers": args.num_workers,
            "rotate_and_flip": rotate_and_flip,
            "extra_rotate_and_flip": extra_rotate_and_flip,
            "log_path": log_path,
            "check_corrupt": args.check_corrupt,
        }


    # Main loop for destriping tiles as they come up
    while True:
        # After getting metadata, we get the tiles
        tiles = count_tiles(raw_dir, destriped_dir, metadata, metadata_version)
        show_output(tiles, raw_dir)

        # Check to see if we've finished destriping the main tiles yet.
        finished = True
        for tile in tiles:
            if tile['output_images'] < tile['expected']:
                finished = False
                break # Don't need to keep going through
        if finished:
            print('\nAll tiles have been destriped.  Checking for Maximum Intensity Projections...')
            check_mips(raw_dir, destriped_dir, tiles, destripe_list_path, **destripe_params)
            finish_directory(raw_dir, destriped_dir)
            break

        destripe_tile = False
        waiting_tile = False

        # Get the next tile to destripe
        for tile in tiles:
            if tile['input_images'] >= tile['expected'] and tile['output_images'] < tile['expected']:
                destripe_tile = tile['path']
                break
        
        # If no tile is found that is needed to be destriped, then we wait for a tile that has input images but currently has not been finished
        # so destriping has not started yet
        if not destripe_tile:
            for tile in tiles:
                if tile['input_images'] > 0 and tile['output_images'] == 0:
                    waiting_tile = tile
                    break 
        
        # If there is a destripe tile, then we start destriping it
        if destripe_tile:
            input_path = os.path.join(raw_dir, destripe_tile)
            output_path = os.path.join(destriped_dir, destripe_tile)
            print('\nDestriping {}...\n'.format(destripe_tile))
            time.sleep(1)
            if tile["expected"] == 1:
                use_gpu = False
            else:
                use_gpu = args.use_gpu
            run_pystripe(input_path, output_path, use_gpu = use_gpu, **destripe_params)
            update_destripe_list_txt(input_path, destripe_list_path)
            
        elif waiting_tile:
            # Wait for the current tile to finish being acquired
            print('\nWaiting for current tile: {} to finish being acquired...'.format(waiting_tile['path']))
            if stall_counter[0] == waiting_tile['path'] and stall_counter[1] == waiting_tile['input_images']:
                stall_counter[2] += 1
            else:
                stall_counter[0] = waiting_tile['path']
                stall_counter[1] = waiting_tile['input_images']
                stall_counter[2] = 0

            if stall_counter and stall_counter[2] > args.stall_limit:
                raise ValueError(f"Stall limit of {args.stall_limit} reached, assuming aborted acquisition.")
            time.sleep(5)

        else:
            time.sleep(5)

    logger.info(f"Destriping all stacks took {time.time()-tic} seconds")

if __name__ == "__main__":
    main()
