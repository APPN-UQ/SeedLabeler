@echo off
setlocal

:: Build SeedLabeler into a distributable folder + zip.
:: Run from the SeedLabeler repo root with the seedlabeler conda env active.
:: Output: dist\SeedLabeler\ (folder) and dist\SeedLabeler-win-VERSION.zip

where pyinstaller >nul 2>&1 || (
    echo Installing PyInstaller ...
    pip install pyinstaller
)

echo.
echo === Building SeedLabeler ===
echo.
pyinstaller SeedLabeler.spec --noconfirm
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

:: Read version from the built app
for /f %%v in ('python -c "from seedlabeler import __version__; print(__version__)"') do set VER=%%v
set ZIPNAME=SeedLabeler-win-%VER%.zip

echo.
echo === Zipping dist\SeedLabeler as %ZIPNAME% ===
echo.
cd dist
if exist "%ZIPNAME%" del "%ZIPNAME%"
powershell -Command "Compress-Archive -Path 'SeedLabeler' -DestinationPath '%ZIPNAME%' -Force"
cd ..

echo.
echo === Done ===
echo   Folder: dist\SeedLabeler\SeedLabeler.exe
echo   Zip:    dist\%ZIPNAME%
echo.
echo To create a GitHub release:
echo   gh release create v%VER% dist\%ZIPNAME% --title "SeedLabeler %VER%" --notes "Release %VER%"
echo Or upload dist\%ZIPNAME% at https://github.com/APPN-UQ/SeedLabeler/releases/new
echo.
