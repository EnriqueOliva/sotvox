# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files

PROJECT_ROOT = os.path.dirname(SPECPATH)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

datas = [
    (os.path.join(PROJECT_ROOT, 'assets'), 'assets'),
    (os.path.join(PROJECT_ROOT, 'sounds'), 'sounds'),
]
binaries = []
hiddenimports = ['constants', 'engine', 'gpu_pack', 'ui']

for package in ('faster_whisper', 'ctranslate2', 'av', 'tkinterdnd2'):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

datas += collect_data_files('faster_whisper', include_py_files=False)

a = Analysis(
    [os.path.join(SRC_DIR, 'main.py')],
    pathex=[SRC_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'nvidia',
        'torch',
        'matplotlib',
        'numpy.distutils',
        'pytest',
        'setuptools',
        'pip',
        'PyInstaller',
    ],
    noarchive=False,
    optimize=0,
)

CUDA_LIBRARY_PREFIXES = (
    'cudnn', 'cublas', 'nvblas', 'cudart', 'cufft', 'curand',
    'cusolver', 'cusparse', 'nvrtc', 'nvjitlink',
)


def _is_cuda_payload(entry):
    source_path = str(entry[1]).lower()
    if 'nvidia' in source_path:
        return True
    else:
        basename = os.path.basename(str(entry[0])).lower()
        return basename.startswith(CUDA_LIBRARY_PREFIXES)


a.binaries = TOC([entry for entry in a.binaries if not _is_cuda_payload(entry)])
a.datas = TOC([entry for entry in a.datas if not _is_cuda_payload(entry)])

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Sotvox',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PROJECT_ROOT, 'assets', 'sotvox.ico'),
    version=os.path.join(SPECPATH, 'version_info.txt'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Sotvox',
)
