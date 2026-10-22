@echo off
rem Double-click to set up SeedLabeler on this PC (see setup.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause
