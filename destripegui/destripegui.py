import os, sys, time, csv, re, json
import math
import multiprocessing
import configparser
from pathlib import Path
from datetime import datetime
import traceback
import shutil
from win32event import CreateMutex
from win32api import GetLastError
from winerror import ERROR_ALREADY_EXISTS
from tqdm import tqdm
from sys import exit
#import torch
import subprocess
from pprint import pprint
import math
from tabulate import tabulate
from PIL import Image

from destripegui.destripe.core import main as cpu_destripe
from destripegui.destripe.utils import find_all_images
from destripegui.destripe import supported_extensions

def get_configs(config_path):
    reader = configparser.ConfigParser()   
    reader.read(config_path)
    configs = {}
    for section in reader.sections():
        for (key, val) in reader.items(section):
            if val.lower() == 'true': val = True
            elif val.lower() == 'false': val = False
            configs[key] = val
    return configs

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
    

def run_pystripe(input_path, output_path, current_dir):
    # print('test')
    # input_path = Path(dir['path'])
    # output_path = Path(dir['output_path'])
    sig_strs = current_dir['metadata']['sample metadata']['destripe'].split('/')
    sigma = list(int(sig_str) for sig_str in sig_strs)

    # sigma = [256, 0]
    workers = int(configs['workers'])
    chunks = int(configs['chunks'])
    use_gpu = int(configs["use_gpu"])
    cpu_readers = int(configs["cpu_readers"])
    gpu_chunksize = int(configs["gpu_chunksize"])
    ram_loadsize = int(configs["ram_loadsize"])

    contents = os.listdir(input_path)
    if len(contents) == 1:
        # input_path = os.path.join(input_path, contents[0])
        # output_path = os.path.join(output_path, contents[0])
        use_gpu = 0

    if 'MIP' in input_path:
        use_gpu = 0

    if use_gpu:
        print("Using GPU Destriper")
        from destripegui.destripe.core_gpu import main as gpu_destripe
        cmd = ["-i", str(input_path),
                        "-o", str(output_path), 
                        "--sigma1", str(sigma[0]),
                        "--sigma2", str(sigma[1]),
                        "--cpu-readers", str(workers), 
                        "--gpu-chunksize", str(gpu_chunksize),
                        "--extra-smoothing", "True"]
        if ram_loadsize > 0:
            cmd.append("--ram-loadsize")
            cmd.append(str(ram_loadsize))
        print(cmd)
        
        gpu_destripe(cmd)        

    else:
        print("Using CPU Destriper")
        cpu_destripe(["-i", str(input_path),
                        "-o", str(output_path), 
                        "--sigma1", str(sigma[0]),
                        "--sigma2", str(sigma[1]),
                        "--workers", str(workers),
                        "--chunks", str(chunks)])
    
    if 'MIP' in input_path or not configs['check_corrupt']:
        return

    corrupted = check_for_bad_images(output_path)
    if corrupted > 0:
        print('{} corrupt images found in {}.  This folder is being re-destriped'.format(corrupted, output_path))
        run_pystripe(input_path, output_path, current_dir)

def get_metadata(dir):
    # builds metadata dict
    metadata_path = os.path.join(dir['path'], 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        dir['metadata'] = metadata

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
        steps_per_tile = max(math.ceil(float(z_block) / float(z_step)) - 1, 1)
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

def get_metadata_v5(dir):
    # builds metadata dict
    metadata_path = os.path.join(dir['path'], 'metadata.txt')

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

    dir['metadata'] = metadata_dict

def search_directory(search_dir, ac_list, depth):
    # Recursive search function through input_dir to find directories with metadata.json.  Ignores no_list

    try:
        contents = os.listdir(search_dir)
    except:
        print('Could not access input directory: {}.'.format(configs['input_dir']))
        print('Make sure drive is accessible, and not open in another program.')
        x = input('Press Enter to retry...')
        search_loop()

    is_acquisition = True
    if 'metadata.txt' not in contents: is_acquisition = False
    if 'sequence.json' in contents and configs['version'] != '6': is_acquisition = False
    if 'sequence.json' not in contents and configs['version'] == '6': is_acquisition = False

    if is_acquisition:
        ac_list.append({
            'path': search_dir, 
            'output_path': os.path.join(configs['output_dir'], os.path.relpath(search_dir, configs['input_dir']))
        })
        # log("Adding {} to provisional Acquisition Queue".format(search_dir), False)
        return ac_list

    if depth == 0: return ac_list
    for item in contents:
        item_path = os.path.join(search_dir, item)
        if os.path.isdir(item_path) and item_path not in configs['no_list']:
            ac_list = search_directory(item_path, ac_list, depth-1)
    return ac_list
        
def get_acquisition_dirs():
    # run recursive search for new directories.  Build metadata dicts. Checks metadata flags and folder names to make
    # sure its actually new, and adds to no_list if not 

    search_dir = configs['input_dir']
    ac_dirs = search_directory(search_dir, list(), depth=3)
            
    for dir in ac_dirs:
        if configs['version'] == '6':
            get_metadata(dir)
        else:
            get_metadata_v5(dir)
   
    unfinished_dirs = []    
    for dir in ac_dirs:
        destripe_status = dir['metadata']['sample metadata']['destripe_status']
        if destripe_status == 'true':
            unfinished_dirs.append(dir)
        else:
            configs['no_list'].append(dir['path'])
    
    if len(unfinished_dirs) > 0: unfinished_dirs.sort(key=lambda x: x['path'])
    return unfinished_dirs

def count_tiles(dir):
    tiles = []
    for tile in dir['metadata']['tiles']:
        expected = int(tile['NumImages'])
        laser = tile['Laser']
        filter = tile['Filter']
        x = tile['X']
        y = tile['Y']
        
        if configs['version'] == '6':
            ch = tile['FilterChannel']
            tile_path = os.path.join('Ex_{}_Em_{}_Ch{}'.format(laser, filter, ch), x, '{}_{}'.format(x, y))
        else:
            tile_path = os.path.join('Ex_{}_Ch{}'.format(laser, filter), x, '{}_{}'.format(x, y))

        input_images = len(os.listdir(os.path.join(dir['path'], tile_path)))
        try:
            output_images = len(os.listdir(os.path.join(dir['output_path'], tile_path)))
        except:
            output_images = 0      
        tiles.append({
            'path': tile_path,
            'input_images': input_images,
            'output_images': output_images,
            'expected': expected
        })
    dir['tiles'] = tiles

def show_output(ac_dirs, current_dir):
    headers = ['Tile', 'Images Expected', 'Images on Acquisition Drive', 'Images on Stitch Drive']
    data = []
    total_images = 0
    total_destriped = 0
    for tile in current_dir['tiles']:
        total_images += tile['expected']
        total_destriped += tile['output_images']
        data.append([
            tile['path'],
            tile['expected'],
            tile['input_images'],
            tile['output_images']
        ])
    print('Current Acquisition: {}\n'.format(current_dir['path']))
    print(tabulate(data, headers))
    pct = total_destriped / total_images
    bar_length = 72
    print('\nOVERALL DESTRIPING PROGRESS: {:.0%} [{}{}]'.format(pct, '#'*round(pct*bar_length), '-'*round((1-pct)*bar_length)))

    if len(ac_dirs) > 1:
        print('\nAdditional Acquisitions in Destriping Queue:')
        for i in range(1, len(ac_dirs)):
            print(ac_dirs[i]['path'])
    
def check_mips(current_dir):
    for item in os.listdir(current_dir['path']):
        if 'MIP' in item:
            input_path = os.path.join(current_dir['path'], item)
            output_path = os.path.join(current_dir['output_path'], item)
            try:
                output_images = len(os.listdir(output_path))
            except:
                output_images = 0

            if len(os.listdir(input_path)) != output_images:
                print('\nDestriping {}...\n'.format(item))
                run_pystripe(input_path, output_path, current_dir)

def finish_directory(dir):
    configs['no_list'].append(dir['path'])
    if configs['time_stamp']:
        time_stamp_finish(dir)

    for file in Path(dir['path']).iterdir():
        file_name = os.path.split(file)[1]
        if Path(file).suffix in ['.txt', '.ini', '.json']:
            output_file = os.path.join(Path(dir['output_path']), file_name)
            shutil.copyfile(file, output_file)

    if configs['version'] == '6':
        change_status(dir, 'in', 'done')
        change_status(dir, 'out', 'done')
    else:
        prepend_tag(dir, 'in', 'D')
        prepend_tag(dir, 'out', 'D')

    # x = input('about to rename...')
    append_folder_name(dir, 'in', configs['input_done'])
    append_folder_name(dir, 'out', configs['output_done'])

    # log(' finishing {}'.format(dir['path']), True)

def append_folder_name(dir, drive, msg, attempts = 0):
    global reconnect
    if drive == 'in':
        path = dir['path'] 
    else:
        path = dir['output_path']

    try:
        split = os.path.split(path)
        if msg not in split[1]:
            new_dir_name = split[1] + msg
            new_path = os.path.join(split[0], new_dir_name)
            os.rename(path, new_path)
    except Exception as error:
        print(error)
        print('Cannot access {} to rename folder'.format(path))
        if configs['reconnect']:
            print('Retrying in 5 seconds')
            time.sleep(5)
        else:
            x = input('Make sure it is accessible and not open in another program, then press Enter to retry...\n')
        append_folder_name(dir, drive, msg)

def prepend_tag(dir, drive, msg):
    # prepend tag to metadata file
    
    if drive == 'in':
        metadata_path = os.path.join(dir['path'], 'metadata.txt')
    else:
        metadata_path = os.path.join(dir['output_path'], 'metadata.txt')
    try:
        with open(metadata_path, errors="ignore") as f:
            reader = csv.reader(f, dialect='excel', delimiter='\t')
            line_list = list(reader)
            
        destripe_position = line_list[0].index('Destripe')
        destripe = line_list[1][destripe_position]
        for char in 'ACDNacdn':
            destripe = destripe.replace(char, '')

        line_list[1][destripe_position] = msg + destripe
        # os.remove(metadata_path)
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f, dialect='excel', delimiter='\t')
            for row in line_list:
                writer.writerow(row)
    except:
        print('Cannot access {} to change destripe tag'.format(metadata_path))
        x = input('Make sure it is accessible and not open in another program, then press Enter to retry...\n')
        prepend_tag(dir, drive, msg)

def change_status(dir, drive, msg):
    # prepend tag to metadata file

    if drive == 'in':
        metadata_path = os.path.join(dir['path'], 'metadata.json')
    else:
        metadata_path = os.path.join(dir['output_path'], 'metadata.json')
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['sample metadata']['destripe_status'] = msg

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except:
        print('Cannot access {} to change destripe_status'.format(metadata_path))
        x = input('Make sure it is accessible and not open in another program, then press Enter to retry...\n')
        change_status(dir, drive, msg)

def abort(dir):
    # Perform tasks needed to respond to aborted acquisition
    
    print("\nAborting {}...\n".format(dir['path']))

    if configs['version'] == '6':
        change_status(dir, 'in', 'aborted')
    else:
        prepend_tag(dir, 'in', 'A')
    append_folder_name(dir, 'in', configs['input_abort'])

    if os.path.exists(dir['output_path']):
        if os.path.exists(os.path.join(dir['output_path'], 'metadata.json')):
            if configs['version'] == '6':
                change_status(dir, 'out', 'aborted')
            else:
                prepend_tag(dir, 'out', 'A')
        append_folder_name(dir, 'out', configs['output_abort'])
            
def time_stamp_start(current_dir):
    time_file = os.path.join(current_dir['output_path'], 'Time Stamps.txt')
    try:
        with open(time_file, 'r') as f:
            pass
    except:
        os.makedirs(current_dir['output_path'])
        with open(time_file, 'w') as f:
            f.write('Destriper Start Time: {}'.format(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))

def time_stamp_finish(current_dir):
    finish_time = datetime.now()
    time_file = os.path.join(current_dir['output_path'], 'Time Stamps.txt')

    with open(time_file, 'r') as f:
        start_string = f.readlines()[0]
        start_time = datetime.strptime(start_string[22:], "%m/%d/%Y, %H:%M:%S")

    elapsed_time = finish_time - start_time
    s = elapsed_time.seconds
    hours = math.floor(s/3600)
    minutes = math.floor(s/60)%60
    seconds = s%60
    timer_text = "\nDestriper Finish Time: {}".format(finish_time.strftime("%m/%d/%Y, %H:%M:%S"))
    timer_text += "\nDestriper Elapsed Time: {:02}:{:02}:{:02}".format(hours, minutes, seconds)

    if configs['version'] == '6':
        acq_file = os.path.join(current_dir['path'], 'acquisition log.txt')
        with open(acq_file, 'r') as f:
            lines = f.readlines()
        line = lines[5]
        acq_start = datetime.strptime(line[:line.index("\t")], "%Y-%m-%dT%H:%M:%S")
        line = lines[-1]
        acq_finish = datetime.strptime(line[:line.index("\t")], "%Y-%m-%dT%H:%M:%S")
    else:
        acq_file = os.path.join(current_dir['path'], 'ASI_logging.txt')
        with open(acq_file, 'r') as f:
            lines = f.readlines()
        line = lines[0]
        acq_start = datetime.strptime(line[:line.index('M')+1], "%m/%d/%Y %I:%M:%S %p")
        line = lines[-1]
        acq_finish = datetime.strptime(line[:line.index('M')+1], "%m/%d/%Y %I:%M:%S %p")

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

def search_loop():
    while True:
        print('\n-------------\n\n')
        ac_dirs = get_acquisition_dirs()

        if len(ac_dirs) == 0:
            print("Waiting for new acquisitions...")
            time.sleep(5)
            continue
        if len(ac_dirs) > 0:
            current_dir = ac_dirs[0]
            count_tiles(current_dir)
            
            show_output(ac_dirs, current_dir)
            if configs['safe_mode']:
                x = input('Press Enter to exit program...')
                exit()

            finished = True
            for tile in current_dir['tiles']:
                if tile['output_images'] < tile['expected']:
                    finished = False
            if finished:
                print('\nAll tiles have been destriped.  Checking for Maximum Intensity Projections...')
                check_mips(current_dir)
                finish_directory(current_dir)
                continue

            destripe_tile = False
            waiting_tile = False

            for tile in current_dir['tiles']:
                if tile['input_images'] >= tile['expected'] and tile['output_images'] < tile['expected']:
                    destripe_tile = tile['path']
                    break

            if not destripe_tile:
                for tile in current_dir['tiles']:
                    if tile['input_images'] > 0 and tile['output_images'] == 0:
                        waiting_tile = tile
                        break 
            
            if destripe_tile:
                input_path = os.path.join(current_dir['path'], destripe_tile)
                output_path = os.path.join(current_dir['output_path'], destripe_tile)
                if configs['time_stamp']:
                    time_stamp_start(current_dir)
                print('\nDestriping {}...\n'.format(destripe_tile))
                time.sleep(1)
                run_pystripe(input_path, output_path, current_dir)

            elif waiting_tile:
                print('\nWaiting for current tile: {} to finish being acquired...'.format(waiting_tile['path']))
                if configs['stall_counter'][0] == waiting_tile['path'] and configs['stall_counter'][1] == waiting_tile['input_images']:
                    configs['stall_counter'][2] += 1
                else:
                    configs['stall_counter'][0] = waiting_tile['path']
                    configs['stall_counter'][1] = waiting_tile['input_images']
                    configs['stall_counter'][2] = 0

                if configs['stall_limit'] and configs['stall_counter'][2] > int(configs['stall_limit']):
                    x = input('\nThis acquisition ({}) seems to be incomplete.  Mark as aborted (y/n)?\n'.format(current_dir['path']))
                    if x in 'yesYesyeahsure':
                        abort(current_dir)
                        continue
                time.sleep(5)

            else:
                time.sleep(5)

def main():
    # print('testing')
    if 'configs' not in globals():
        double_test = CreateMutex(None, 1, 'A unique mutex name')
        if GetLastError(  ) == ERROR_ALREADY_EXISTS:
            # Take appropriate action, as this is the second
            # instance of this script; for example:
            print('Another instance of destripegui is already running')
            exit(1)

    global configs
    
    print('Reading config file...\n')

    config_path = Path(__file__).parent / 'data/config.ini'
    configs = get_configs(config_path)

    configs['input_dir'] = Path(configs['input_dir'])
    configs['output_dir'] = Path(configs['output_dir'])

    configs['safe_mode'] = False
    try:
        if sys.argv[1] == '-s':
            configs['safe_mode'] = True
            print('\nRunning in Safe Mode.  No changes will be made to any files.\n')
    except:
        pass
        
    configs['stall_counter'] = ['', 0, 0]
    configs['no_list'] = []

    if 'reconnect' not in configs.keys(): configs['reconnect'] = False
    if 'check_corrupt' not in configs.keys(): configs['check_corrupt'] = False
    if 'stall_limit' not in configs.keys(): configs['stall_limit'] = False
    if 'time_stamp' not in configs.keys(): configs['time_stamp'] = False  

    try:
        x = os.listdir(configs['input_dir'])
    except:
        print('Could not access input directory: {}.'.format(configs['input_dir']))
        print('Make sure drive is accessible, or change drive location in config file: {}'.format(config_path))
        x = input('Press Enter to retry...')
        main()
    try:
        x = os.listdir(configs['output_dir'])
    except:
        print('Could not access output directory: {}.'.format(configs['output_dir']))
        print('Make sure drive is accessible, or change drive location in config file: {}'.format(config_path))
        x = input('Press Enter to retry...')
        main()
    
    print('\nScanning {} for new acquisitions...\n'.format(configs['input_dir']))
    search_loop()
    

if __name__ == "__main__":
    main()
