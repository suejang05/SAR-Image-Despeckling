@REM @echo off
@REM setlocal enabledelayedexpansion

@REM if not exist "%userprofile%\.kaggle\kaggle.json" (
@REM   set /p USERNAME=Kaggle username: 
@REM   echo.
@REM   set /p APIKEY=Kaggle API key: 

@REM   mkdir "%userprofile%\.kaggle"
@REM   echo {"username":"!USERNAME!","key":"!APIKEY!"} > "%userprofile%\.kaggle\kaggle.json"
@REM   attrib +R "%userprofile%\.kaggle\kaggle.json"
@REM )

@REM pip install kaggle --upgrade

@REM kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip
@REM powershell Expand-Archive train_hq.zip -DestinationPath data\imgs
@REM move data\imgs\train_hq\* data\imgs\
@REM rmdir /s /q data\imgs\train_hq
@REM del /q train_hq.zip

@REM kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
@REM powershell Expand-Archive train_masks.zip -DestinationPath data\masks
@REM move data\masks\train_masks\* data\masks\
@REM rmdir /s /q data\masks\train_masks
@REM del /q train_masks.zip

@REM exit /b 0
