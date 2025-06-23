call conda env remove -n cl_destripe_6

call conda env create -f environment.yml

call conda activate cl_destripe_6

call pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

call pip install -e .


@echo off

set SCRIPT="%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%.vbs"

echo Set WshShell = CreateObject("Wscript.shell") >> %SCRIPT%
echo Set oLink = WshShell.CreateShortcut("%USERPROFILE%\Desktop\Destripe v6.lnk") >> %SCRIPT%
echo oLink.TargetPath = "%~dp0destripegui\data\Command_Line_Destripe.bat" >> %SCRIPT%
echo oLink.IconLocation = "%~dp0destripegui\data\lct.ico" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%

cscript %SCRIPT%
del %SCRIPT%
