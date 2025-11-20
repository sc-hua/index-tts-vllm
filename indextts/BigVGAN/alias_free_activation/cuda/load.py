# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import pathlib
import subprocess

import torch
from torch.utils import cpp_extension

"""
Setting this param to a list has a problem of generating different compilation commands (with diferent order of architectures) and leading to recompilation of fused kernels. 
Set it to empty stringo avoid recompilation and assign arch flags explicity in extra_cuda_cflags below
"""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


import re
import shutil
import tempfile

# 补丁修复：sources 路径含中文字符时，生成 build.ninja 乱码导致编译失败
# 使用临时目录来规避 ninja 编译失败（比如中文路径）
def chinese_path_compile_support(sources, buildpath):
    pattern = re.compile(r'[\u4e00-\u9fff]')  
    if not bool(pattern.search(str(sources[0].resolve()))):
        return buildpath # 检测非中文路径跳过
    # Create build directory
    resolves = [ item.name for item in sources]
    ninja_compile_dir = os.path.join(tempfile.gettempdir(), "BigVGAN", "cuda")
    os.makedirs(ninja_compile_dir, exist_ok=True)
    new_buildpath = os.path.join(ninja_compile_dir, "build")
    os.makedirs(new_buildpath, exist_ok=True)
    print(f"ninja_buildpath: {new_buildpath}")
    # Copy files to directory
    sources.clear()
    current_dir = os.path.dirname(__file__)
    ALLOWED_EXTENSIONS = {'.py', '.cu', '.cpp', '.h'}
    for filename in os.listdir(current_dir):
        item = pathlib.Path(current_dir).joinpath(filename)
        tar_path = pathlib.Path(ninja_compile_dir).joinpath(item.name)
        if not item.suffix.lower() in ALLOWED_EXTENSIONS:continue
        pathlib.Path(shutil.copy2(item, tar_path))
        if tar_path.name in resolves:sources.append(tar_path)
    return new_buildpath



def load():
    arches = _resolve_cuda_arch_list()
    arch_flags = _build_arch_flags(arches)
    if not arch_flags:
        arch_flags = _build_arch_flags(["70"])  # safety fallback

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
            ]
            + arch_flags
            + extra_cuda_flags,
            verbose=True,
        )

    extra_cuda_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    
    # 兼容方案：ninja 特殊字符路径编译支持处理（比如中文路径）
    buildpath = chinese_path_compile_support(sources, buildpath)
    
    anti_alias_activation_cuda = _cpp_extention_load_helper(
        "anti_alias_activation_cuda", sources, extra_cuda_flags
    )

    return anti_alias_activation_cuda


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")


def _resolve_cuda_arch_list():
    env_value = os.environ.get("INDEXTTS_CUDA_ARCH_LIST")
    if env_value:
        parsed = _parse_arches(env_value)
        if parsed:
            return parsed

    # Try to detect from the current device
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            detected = [f"{major}{minor}"]
            parsed = _parse_arches(detected)
            if parsed:
                return parsed
    except Exception:
        pass

    # Fall back to legacy defaults (sm70 + optional sm80 if CUDA >= 11)
    defaults = ["70"]
    cuda_home = cpp_extension.CUDA_HOME
    if cuda_home:
        try:
            _, bare_metal_major, _ = _get_cuda_bare_metal_version(cuda_home)
            if int(bare_metal_major) >= 11:
                defaults.append("80")
        except Exception:
            pass
    return _parse_arches(defaults)


def _parse_arches(values):
    if isinstance(values, str):
        values = re.split(r"[\s,]+", values)
    parsed = []
    for value in values:
        if value is None:
            continue
        cleaned = value.strip().lower()
        if not cleaned:
            continue
        for prefix in ("sm_", "sm", "compute_", "compute"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
        cleaned = cleaned.replace(".", "")
        if not cleaned.isdigit():
            continue
        if cleaned not in parsed:
            parsed.append(cleaned)
    return parsed


def _build_arch_flags(arches):
    flags = []
    if not arches:
        return flags
    for arch in arches:
        flags.extend([
            "-gencode",
            f"arch=compute_{arch},code=sm_{arch}",
        ])
    return flags
