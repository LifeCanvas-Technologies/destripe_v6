@echo on
cd /D "%~dp0"
call conda env remove -n cl_destripe_6
call conda env remove -p C:\ProgramData\Anaconda3\envs\cl_destripe_6

call conda env create -n cl_destripe_6 -f environment.yml

call conda activate cl_destripe_6

call pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

call pip install -e .
