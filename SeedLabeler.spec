# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SeedLabeler.

Build with:  pyinstaller SeedLabeler.spec
Output:      dist/SeedLabeler/SeedLabeler.exe
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Data files that PyInstaller won't find automatically
datas = []
datas += collect_data_files("ultralytics")
datas += collect_data_files("depthai")
datas += collect_data_files("colourcal")

# Board layouts shipped with the app
datas += [("boards", "boards")]

# Hidden imports: packages loaded dynamically or via lazy imports
hiddenimports = []
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("skimage")
hiddenimports += collect_submodules("PySide6")
hiddenimports += [
    "chickpea", "chickpea.pipeline", "chickpea.stage", "chickpea.segment",
    "chickpea.measure", "chickpea.summary", "chickpea.ortho",
    "sonyrx0", "sonyrx0.stereo", "sonyrx0.stereo.depth", "sonyrx0.stereo.common",
    "sonyrx0.camera", "sonyrx0.camera.sony", "sonyrx0.camera.wpd_mtp",
    "sonyrx0.camera.settings",
    "colourcal", "colourcal.correct",
    "comtypes", "comtypes.stream",
    "torch", "torchvision",
    "pandas", "pandas.io.formats.csvs",
    "scipy", "scipy.ndimage", "scipy.spatial",
    "matplotlib", "matplotlib.backends.backend_agg",
]

a = Analysis(
    ["launch.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SeedLabeler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="SeedLabeler",
)
