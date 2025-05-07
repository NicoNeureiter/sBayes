@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
  echo Usage: run_sbayes.bat path\to\config.yaml [additional sBayes CLI arguments...]
  exit /b 1
)

REM Split path and filename
for %%F in (%~f1) do (
  set "CONFIG_PATH=%%~fF"
  set "CONFIG_DIR=%%~dpF"
  set "CONFIG_FILE=%%~nxF"
)

REM Shift off the config argument
shift

REM Convert Windows path to Docker-compatible format
set "CONFIG_DIR_UNIX=%CONFIG_DIR:\=/%"
set "CONFIG_DIR_UNIX=%CONFIG_DIR_UNIX:C:=/c%"

REM Reconstruct the remaining CLI args
set "CLI_ARGS="
:buildargs
if "%~1"=="" goto run
set CLI_ARGS=!CLI_ARGS! %~1
shift
goto buildargs

:run
docker run --rm -v "%CONFIG_DIR%:/data" sbayes /data/%CONFIG_FILE% %CLI_ARGS%
