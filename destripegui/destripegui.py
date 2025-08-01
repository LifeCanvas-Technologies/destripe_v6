import os, time, csv, json
import math
import configparser
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


from destripegui.destripe.utils import find_all_images
from destripegui.destripe import supported_extensions

DESTRIPE_TIME_STAMP_PATH = 'logs/destriping_time_stamps.txt'

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
    parser.add_argument("--is-smartspim", action="store_true", default=False, help="Whether it's smartspim or megaspim postprocessing, which affects metadata reading.")
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
                 log_path = None,
                 check_corrupt = False):
    if use_gpu:
        print("Using GPU Destriper")
        from destripegui.destripe.core_gpu import main as gpu_destripe
        cmd = ["-i", str(input_path),
                        "-o", str(output_path), 
                        "--sigma1", str(sigma1),
                        "--sigma2", str(sigma2),
                        "--cpu-readers", str(num_workers)]
        if ram_loadsize is not None:
            cmd.append("--ram-loadsize")
            cmd.append(str(ram_loadsize))
        if gpu_chunksize is not None:
            cmd.append("--gpu-chunksize")
            cmd.append(str(gpu_chunksize))
        if log_path is not None:
            cmd.append("--log-path")
            cmd.append(str(log_path))
        print(cmd)
        
        gpu_destripe(cmd)        

    else:
        print("Using CPU Destriper")
        from destripegui.destripe.core import main as cpu_destripe
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
    
def check_mips(raw_dir, destriped_dir, tiles, **kwargs):
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
                
                run_pystripe(input_path, output_path,
                            **kwargs)
    else:
        print("No MIP folders found in acquisition, skipping...")

def finish_directory(input_path, output_path, metadata_version):
    """
    Show and save time stamp when finished, then move the .txt .ini and .json files to the new directory.
    """
    time_stamp_finish(input_path, output_path, metadata_version)
    for file in Path(input_path).iterdir():
        file_name = os.path.split(file)[1]
        if Path(file).suffix in ['.txt', '.ini', '.json']:
            output_file = os.path.join(Path(output_path), file_name)
            shutil.copyfile(file, output_file)
            
def time_stamp_start(output_path):
    time_file = os.path.join(output_path, DESTRIPE_TIME_STAMP_PATH)
    try:
        # If it already exists
        with open(time_file, 'r') as f:
            pass
    except:
        os.makedirs(Path(time_file).parent, exist_ok=True)
        with open(time_file, 'w') as f:
            f.write('Destriper Start Time: {}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))

def time_stamp_finish(input_path, output_path, metadata_version):
    finish_time = datetime.now()
    time_file = os.path.join(output_path, DESTRIPE_TIME_STAMP_PATH)

    with open(time_file, 'r') as f:
        start_string = f.readlines()[0]
        start_time = datetime.strptime(start_string[22:].strip(), "%m/%d/%Y, %H:%M:%S")

    elapsed_time = finish_time - start_time
    s = elapsed_time.seconds
    hours = math.floor(s/3600)
    minutes = math.floor(s/60)%60
    seconds = s%60
    timer_text = "\nDestriper Finish Time: {}".format(finish_time.strftime("%m/%d/%Y, %H:%M:%S"))
    timer_text += "\nDestriper Elapsed Time: {:02}:{:02}:{:02}".format(hours, minutes, seconds)

    if metadata_version == 6:
        acq_file = os.path.join(input_path, 'acquisition log.txt')
        with open(acq_file, 'r') as f:
            lines = f.readlines()
        line = lines[5]
        acq_start = datetime.strptime(line[:line.index("\t")], "%Y-%m-%dT%H:%M:%S")
        line = lines[-1]
        acq_finish = datetime.strptime(line[:line.index("\t")], "%Y-%m-%dT%H:%M:%S")
    elif metadata_version == 5:
        acq_file = os.path.join(input_path, 'ASI_logging.txt')
        with open(acq_file, 'r') as f:
            lines = f.readlines()
        line = lines[0]
        acq_start = datetime.strptime(line[:line.index('M')+1], "%m/%d/%Y %I:%M:%S %p")
        line = lines[-1]
        acq_finish = datetime.strptime(line[:line.index('M')+1], "%m/%d/%Y %I:%M:%S %p")
    else: # megaspim
        # TODO: write the logic for megaspim metadata
        pass

    elapsed_time = acq_finish - acq_start
    s = elapsed_time.seconds
    hours = math.floor(s/3600)
    minutes = math.floor(s/60)%60
    seconds = s%60
    timer_text += "\n\nAcquisition Start Time: {}".format(acq_start.strftime("%m/%d/%Y, %H:%M:%S"))
    timer_text += "\nAcquisition Finish Time: {}".format(acq_finish.strftime("%m/%d/%Y, %H:%M:%S"))
    timer_text += "\nAcquisition Elapsed Time: {:02}:{:02}:{:02}".format(hours, minutes, seconds)

    with open(time_file, 'a') as f:
        f.write(timer_text)


def main():
    args = parse_args()
    raw_dir = Path(args.input)
    destriped_dir = Path(args.output)

        
    stall_counter = ['', 0, 0]

    metadata_version = args.metadata_version

    if not args.is_smartspim:
        metadata_version = 0
        # Then it is Dali or MegaSPIM
        metadata = get_megaspim_metadata(raw_dir) # TODO: write this functino for reading megaspim_metadata
    elif args.metadata_version==6:
        metadata = get_metadata(raw_dir)
    else:
        metadata = get_metadata_v5(raw_dir)
    
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
            check_mips(raw_dir, destriped_dir, tiles,
                       sigma1 = args.sigma1,
                        sigma2 = args.sigma2,
                        use_gpu = args.use_gpu,
                        gpu_chunksize = args.gpu_chunksize,
                        ram_loadsize = args.ram_loadsize,
                        num_workers = args.num_workers,
                        log_path = args.log_path,
                        check_corrupt = args.check_corrupt)
            finish_directory(raw_dir, destriped_dir, metadata_version)
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
            time_stamp_start(destriped_dir)
            print('\nDestriping {}...\n'.format(destripe_tile))
            time.sleep(1)
            run_pystripe(input_path, output_path,
                         sigma1 = args.sigma1,
                        sigma2 = args.sigma2,
                        use_gpu = args.use_gpu,
                        gpu_chunksize = args.gpu_chunksize,
                        ram_loadsize = args.ram_loadsize,
                        num_workers = args.num_workers,
                        log_path = args.log_path,
                        check_corrupt = args.check_corrupt)
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
    

if __name__ == "__main__":
    main()
