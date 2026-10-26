<#
SeedLabeler one-step setup for a Windows PC.

    Double-click setup.bat, or:  powershell -ExecutionPolicy Bypass -File setup.ps1 [-Cpu] [-EnvName name] [-NoShortcut]

Expects SonyRX0, ChickpeaSeed and SeedLabeler side by side in one folder (as in
OneDrive or after cloning the three GitHub repos). Creates or updates a conda
environment, installs PyTorch for this PC's GPU (CUDA 12.8 / 12.6, or CPU),
installs the three packages with the tested versions, runs a self-test and puts
a SeedLabeler shortcut on the desktop. Safe to run again (e.g. after git pull).
#>
param(
    [switch]$Cpu,                       # force the CPU build of PyTorch
    [string]$EnvName = "seedlabeler",
    [switch]$NoShortcut
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$code = Split-Path $here -Parent

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`n$msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- repos
Step "Checking the three repos"
foreach ($r in @("SonyRX0", "ChickpeaSeed", "SeedLabeler")) {
    if (-not (Test-Path (Join-Path $code "$r\pyproject.toml"))) {
        Fail "Missing $code\$r. Put SonyRX0, ChickpeaSeed and SeedLabeler side by side in one folder."
    }
    Write-Host "  found $r"
}

# ---------------------------------------------------------------- conda
Step "Finding conda"
$conda = $null
$cmd = Get-Command conda -ErrorAction SilentlyContinue
if ($cmd -and $cmd.Source -like "*.exe") { $conda = $cmd.Source }
if (-not $conda) {
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe", "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe", "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe", "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniforge3\Scripts\conda.exe")
    $conda = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $conda) {
    Write-Host "  conda not found."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        $ans = Read-Host "  Install Miniconda for this user now with winget? (y/n)"
        if ($ans -eq "y") {
            winget install -e --id Anaconda.Miniconda3 --scope user --accept-package-agreements --accept-source-agreements
            $conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
            if (-not (Test-Path $conda)) { $conda = "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe" }
        }
    }
    if (-not $conda -or -not (Test-Path $conda)) {
        Fail "Install Miniconda (https://docs.conda.io/projects/miniconda), then run setup again."
    }
}
Write-Host "  $conda"

# ---------------------------------------------------------------- environment
Step "Conda environment '$EnvName'"
function CondaEnvs {
    # Only the "envs" list: newer conda also emits per-env details keyed by path, which can
    # differ only in case (Anaconda3 vs anaconda3) and break Windows PowerShell's ConvertFrom-Json
    $json = (& $conda env list --json) -join "`n"
    if ($json -notmatch '(?s)"envs"\s*:\s*(\[[^\]]*\])') { return @() }
    $list = $Matches[1] | ConvertFrom-Json  # a variable, so returning it unrolls the array (PS 5.1)
    return $list
}
$envDir = CondaEnvs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
if (-not $envDir) {
    # conda-forge only: avoids the default channel's terms-of-service prompt
    & $conda create -y -n $EnvName -c conda-forge --override-channels python=3.11 pip
    if ($LASTEXITCODE) { Fail "Could not create the environment." }
    $envDir = CondaEnvs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
}
$py = Join-Path $envDir "python.exe"
if (-not (Test-Path $py)) { Fail "No python.exe in $envDir" }
Write-Host "  $envDir"

function Pip {
    $a = @($args)
    if ($a[0] -eq "install") { $a += "--no-warn-script-location" }  # the env's Scripts dir needn't be on PATH
    & $py -m pip @a
    if ($LASTEXITCODE) { Fail "pip $($args -join ' ') failed." }
}

# ---------------------------------------------------------------- PyTorch for this GPU
Step "PyTorch"
$index = "https://download.pytorch.org/whl/cpu"
$what = "CPU (no NVIDIA GPU found)"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi -and -not $Cpu) {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
    $driver = [double](($gpu -split ",")[1].Trim().Split(".")[0])
    if ($driver -ge 570) { $index = "https://download.pytorch.org/whl/cu128"; $what = "CUDA 12.8 for $gpu" }
    else { $index = "https://download.pytorch.org/whl/cu126"; $what = "CUDA 12.6 for $gpu (update the driver to 570+ for CUDA 12.8)" }
} elseif ($Cpu) { $what = "CPU (forced)" }
Write-Host "  $what"
Pip install --upgrade pip
Pip install torch torchvision --index-url $index

# ---------------------------------------------------------------- packages
Step "SonyRX0, ChickpeaSeed, SeedLabeler"
Pip install -c (Join-Path $here "constraints.txt") -e (Join-Path $code "SonyRX0") -e (Join-Path $code "ChickpeaSeed") -e "$here[train,dev,oak]"
# Ultralytics pulls in opencv-python, which shadows opencv-contrib (needed for ArUco boards)
& $py -m pip uninstall -y opencv-python opencv-python-headless | Out-Null
Pip install --force-reinstall --no-deps "opencv-contrib-python==4.10.0.84"

# ---------------------------------------------------------------- self-test
Step "Self-test"
$test = @'
import cv2, torch, PySide6, ultralytics
import sonyrx0, chickpea, seedlabeler.shell
from sonyrx0.camera.wpd_mtp import find_devices
assert hasattr(cv2, "aruco"), "opencv-contrib missing (no aruco)"
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
print(f"  python ok | torch {torch.__version__} | GPU: {gpu} | OpenCV {cv2.__version__} | PySide6 {PySide6.__version__}")
print(f"  RX0 cameras visible now: {len(find_devices())} (needs both on, PC Remote mode, data cables)")
import depthai
print(f"  OAK cameras visible now: {len(depthai.Device.getAllAvailableDevices())} (depthai {depthai.__version__})")
'@
# via a file: Windows PowerShell strips double quotes from arguments passed to programs
$testFile = Join-Path $env:TEMP "seedlabeler_selftest.py"
Set-Content -Path $testFile -Value $test -Encoding UTF8
& $py $testFile
$rc = $LASTEXITCODE
Remove-Item $testFile -ErrorAction SilentlyContinue
if ($rc) { Fail "Self-test failed (see above)." }

# ---------------------------------------------------------------- shortcut
if (-not $NoShortcut) {
    Step "Desktop shortcut"
    $lnkPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "SeedLabeler.lnk"
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut($lnkPath)
    $lnk.TargetPath = Join-Path $envDir "pythonw.exe"
    $lnk.Arguments = "-m seedlabeler"
    $lnk.WorkingDirectory = $here
    $lnk.Description = "SeedLabeler"
    $lnk.Save()
    Write-Host "  $lnkPath"
}

Write-Host "`nDone. Start SeedLabeler from the desktop shortcut (or: $py -m seedlabeler)." -ForegroundColor Green
Write-Host "SAM weights download on first use; projects open from wherever OneDrive puts them."
