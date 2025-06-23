import os 
from pathlib import Path
import argparse

def revert_images(dir):
    # revert images back from .orig

    revert_count = 0
    for (root,dirs,files) in os.walk(dir):
        for file in files:
            if Path(file).suffix == '.orig':
                file_path = os.path.join(root, file)
                try:
                    os.rename(file_path, os.path.splitext(file_path)[0])
                    revert_count += 1
                except Exception as e:
                    print('    Error reverting image from .orig after acquisition:', e)
    print('    Reverted {} images back from .orig'.format(revert_count))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Path to images that need to be converted back from .orig", type=str, required=True)
    args = parser.parse_args()
    revert_images(args.dir)

if __name__ == "__main__":
    main()