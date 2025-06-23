
Live Command Line Destriping with GPU-accelerated destriping
==============

**Installation instructions**
Instructions for installing Command Line Live Destriping:
1. Ensure Anaconda is installed on the computer.
2. Go to Go to https://github.com/LifeCanvas-Technologies/destripegui/tree/command-line-destripe and click the green "<> Code" button. 
3. Download it by clicking "Download Zip"
4. Extract the folder somewhere
5. Navigate to the folder (named "destripegui") in File Explorer, and then double click "install.bat". This will do all the necessary installations
   * This should have created a Desktop shortcut called "Destripe_CL", which is a clickable shortcut that will run the Destriper.

**Post-installation setup**
1. Modify the config file located at ```destripegui/destripegui/data/config.ini```
   * Change the ```input_dir``` and ```output_dir``` to the raw and destriped directories (e.g. ```D:\SmartSPIM_Data``` and ```S:\SmartSPIM_Data```)
   * Under ```[params]```, the parameters should be good for our default workstation configurations of i9 processor and 4090 GPU. Leave the gpu_chunksize as 64 for a 4090 (scale based on the memory on the workstation GPU). Shouldn't need to actually modify these.
2. For Windows 11 machines, open the Windows Terminal, go to Settings and change the Launch Size to 120 Columns and 60 Rows.  Click Save and close the terminal.

**Running destriping from command line**
1. You can see the help for CPU destriping by first activating the environment with ```conda activate command_line_destripe``` typing in ```cpu-destripe --help```, and GPU destriping by typing in ```gpu-destripe --help```
2. CPU destriping options (should be the same as pystripe)
   * ```cpu-destripe --input <Input Directory> --output <Output Directory> --sigma1 <SIGMA1> --sigma2 <SIGMA2>```
3. GPU destriping options
   * ```gpu-destripe --input <Input Directory> --output <Output Directory> --sigma1 <SIGMA1> --sigma2 <SIGMA2> --extra-smoothing True --gpu-chunksize 64```
   * Leave the ```extra-smoothing``` option to be True, and use 64 for gpu-chunksize unless the GPU is not a 4090 (which has 24GB vRAM. You may have to make it less for GPUs with less than that, and you can increase it for GPUs with larger vRAM)
4. Safe Mode - To run the command line destriper in safe mode, activate the environment with ```conda activate command_line_destripe```, then run ```command_line_destripe -s```.  This will display any acquisitions that the destriper interprets as ready to destripe, without taking any action.  If you do not wish any of these acquisitions to be destriped, change the destripe tag in that acquisition's metadata tag accordingly before running the destriper in standard mode.
