import chardet

def get_txt_encoding(txt_path):
    with open(txt_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    return encoding

def read_metadata(metadata_path):
    # Get the encoding so we can properly read it, Windows or Linux
    encoding = get_txt_encoding(metadata_path)

    # Read with proper encoding
    with open(metadata_path, "r", encoding=encoding) as f:
        lines = f.readlines() 
    array_of_lines = []
    for line in lines:
        array_of_lines.append(line.split('\t'))
    return array_of_lines

def get_spim_type(metadata_path):
    metadata = read_metadata(metadata_path)

    if  "SacSPIM" in metadata[0]:
        return "SacSPIM"
    elif "Fire" in metadata[0]:
        return "Fire"
    elif "MegaSPIM" in metadata[0]:
        return "MegaSPIM"
    return "SmartSPIM" # if we can't find any of those in the first line, then assume it's SmartSPIM

# Utils for SharedMemory
from multiprocessing.shared_memory import SharedMemory

def cleanup_shared_memory(name):
    """make sure all shared memories are closed to prevent any orphand objects"""
    print("Cleaning up shared memory..")
    try:
        shm = SharedMemory(name=name)
        shm.close()
        shm.unlink()
        print("Orphaned shared memory unlinked.")
    except FileNotFoundError as e:
        # print("No orphaned shared memory found.")
        pass
    except Exception as e:
        print(f"Error cleaning up shared memory {name}: {e}")