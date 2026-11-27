@echo off
rem Double-click to update SonyRX0, ColourCal and SeedLabeler from GitHub and reinstall (see install.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
