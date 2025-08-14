@REM @echo off
@REM echo %cd%
@REM if %cd% neq C:\Windows\system32 (
@REM 	echo You need to run this installer as an Administrator
@REM 	pause
@REM 	exit
@REM )
@REM @echo on
@REM cd /D "%~dp0"
@REM call conda env remove -n cl_destripe_6
@REM call conda env remove -p C:\ProgramData\Anaconda3\envs\cl_destripe_6

@REM call conda env create -p C:\ProgramData\Anaconda3\envs\cl_destripe_6 -f environment.yml
@REM icacls C:\ProgramData\Anaconda3\envs\cl_destripe_6 /grant Everyone:(RX) /T

@REM call conda activate cl_destripe_6

@REM call pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

@REM call pip install -e .


@echo off

set SCRIPT="%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%.vbs"

echo Set WshShell = CreateObject("Wscript.shell") >> %SCRIPT%
echo Set oLink = WshShell.CreateShortcut("C:\Users\Public\Desktop\Destripe v6.lnk") >> %SCRIPT%
@REM echo oLink.WindowStyle = 7 >> %SCRIPT%
echo oLink.TargetPath = "%~dp0destripegui\data\Command_Line_Destripe.bat" >> %SCRIPT%
echo oLink.IconLocation = "%~dp0destripegui\data\lct.ico" >> %SCRIPT%
echo oLink.Save >> %SCRIPT%

cscript %SCRIPT%
del %SCRIPT%
pause
