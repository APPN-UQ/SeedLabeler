<#
Get or update SonyRX0, ColourCal and SeedLabeler from GitHub, then install (setup.ps1).

Fresh PC, nothing cloned yet (PowerShell; needs Git and Miniconda/Anaconda):

    irm https://raw.githubusercontent.com/APPN-UQ/SeedLabeler/main/install.ps1 | iex

  puts the three repos in %USERPROFILE%\code\seedlabeler (or set $env:SEEDLABELER_ROOT
  first to choose the folder).

Already installed: double-click install.bat in the SeedLabeler folder, or

    powershell -ExecutionPolicy Bypass -File install.ps1 [-Cpu] [-EnvName name] [-NoShortcut] [-NoSetup]

What "clean up" means here, per repo:
  - missing: cloned fresh
  - folder there but not a git clone (e.g. copied from another PC): renamed to
    <name>.old-<date> and cloned fresh, so nothing is lost
  - git clone: local edits are stashed (git stash list to get them back), it is put
    on the default branch and brought level with GitHub; if it had local commits
    GitHub doesn't, they are kept on a backup-<date> branch first
  - __pycache__ and *.egg-info are removed so the reinstall starts clean
Projects and data are never touched (they live outside these folders).
#>
param(
    [string]$Root,
    [switch]$Cpu,
    [string]$EnvName = "seedlabeler",
    [switch]$NoShortcut,
    [switch]$NoSetup
)
$ErrorActionPreference = "Stop"
$Repos = @("SonyRX0", "ColourCal", "SeedLabeler")
$Org = "https://github.com/APPN-UQ"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Note($msg) { Write-Host "  $msg" -ForegroundColor Yellow }
# throw, not exit: run through "irm | iex", exit would close the PowerShell window
function Fail($msg) { Write-Host "`n$msg" -ForegroundColor Red; throw $msg }
function Invoke-Git {
    & git @args
    if ($LASTEXITCODE) { Fail "git $($args -join ' ') failed." }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "Git is not installed. Get it from https://git-scm.com/download/win (defaults are fine), then run this again."
}

# where the three repos live side by side
if (-not $Root) {
    if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "setup.ps1"))) {
        $Root = Split-Path $PSScriptRoot -Parent          # run from inside SeedLabeler
    } elseif ($env:SEEDLABELER_ROOT) {
        $Root = $env:SEEDLABELER_ROOT
    } else {
        $Root = Join-Path $env:USERPROFILE "code\seedlabeler"
    }
}
New-Item -ItemType Directory -Force $Root | Out-Null
$Root = (Resolve-Path $Root).Path
Write-Host "Repos folder: $Root"

foreach ($r in $Repos) {
    Step $r
    $dir = Join-Path $Root $r
    $url = "$Org/$r.git"
    if ((Test-Path $dir) -and -not (Test-Path (Join-Path $dir ".git"))) {
        $old = "$dir.old-$stamp"
        Rename-Item $dir (Split-Path $old -Leaf)
        Note "not a git clone: moved to $old (delete it once you've checked nothing in it is needed)"
    }
    if (-not (Test-Path $dir)) {
        Invoke-Git clone --quiet $url $dir
        Write-Host "  cloned"
    } else {
        Push-Location $dir
        try {
            Invoke-Git remote set-url origin $url
            Invoke-Git fetch --quiet --prune origin
            Invoke-Git remote set-head origin --auto | Out-Null
            $branch = (git symbolic-ref --short refs/remotes/origin/HEAD) -replace '^origin/', ''
            if (git status --porcelain --untracked-files=no) {
                Invoke-Git stash push --quiet -m "install.ps1 $stamp"
                Note "local edits stashed ('git stash list' in $dir to see them)"
            }
            Invoke-Git checkout --quiet $branch
            $ahead = [int](git rev-list --count "origin/$branch..HEAD")
            if ($ahead -gt 0) {
                Invoke-Git branch "backup-$stamp"
                Note "$ahead local commit(s) not on GitHub kept on branch backup-$stamp"
            }
            Invoke-Git reset --quiet --hard "origin/$branch"
            Write-Host "  up to date with origin/$branch ($(git log -1 --format='%h %s'))"
        } finally {
            Pop-Location
        }
    }
    Get-ChildItem $dir -Recurse -Directory -Force -Include "__pycache__", "*.egg-info" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\\.git\\' } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

if ($NoSetup) {
    Write-Host "`nRepos ready. Install with: $Root\SeedLabeler\setup.bat" -ForegroundColor Green
    return
}
Step "Installing (setup.ps1)"
$setupArgs = @{ EnvName = $EnvName }
if ($Cpu) { $setupArgs.Cpu = $true }
if ($NoShortcut) { $setupArgs.NoShortcut = $true }
& (Join-Path $Root "SeedLabeler\setup.ps1") @setupArgs
