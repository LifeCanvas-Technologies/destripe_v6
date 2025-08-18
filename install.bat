@echo off
echo %cd%
if /I %cd% neq C:\Windows\system32 (
	echo You need to run this installer as an Administrator
	pause
	exit
)
@echo on
cd /D "%~dp0"
call conda env remove -n cl_destripe_6
call conda env remove -p C:\ProgramData\Anaconda3\envs\cl_destripe_6

call conda env create -p C:\ProgramData\Anaconda3\envs\cl_destripe_6 -f environment.yml
icacls C:\ProgramData\Anaconda3\envs\cl_destripe_6 /grant Everyone:(RX) /T

call conda activate cl_destripe_6

call pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

call pip install -e .


@echo off

set SCRIPT="%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%.vbs"

echo Set WshShell = CreateObject("Wscript.shell") >> %SCRIPT%
echo Set oLink = WshShell.CreateShortcut("C:\Users\Public\Desktop\Destripe v6.lnk") >> %SCRIPT%
@REM echo oLink.WindowStyle = 7 >> %SCRIPT%
echo oLink.TargetPath = "%~dp0destripegui\data\Destripe v6.exe" >> %SCRIPT%
echo oLink.IconLocation = "%~dp0destripegui\data\lct.ico" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%

cscript %SCRIPT%
del %SCRIPT%
pause
