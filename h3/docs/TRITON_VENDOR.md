# 海光 Triton 厂商包获取与安装

本工程使用 triton 3.5.1+das.opt1.dtk2604.torch290，对应 Linux x86_64、CPython 3.11、glibc >=2.28、海光 Torch 2.9.0+das.opt1.dtk2604 和 DTK 26.04。Mac 可以下载、校验，不能安装或运行此 Linux wheel。

## 官方地址与校验值

- [海光 DAS PyPI](https://pypi.sourcefind.cn/)
- [das_release/dtk2604-rc1 的 Triton 索引](https://pypi.sourcefind.cn/das_release/dtk2604-rc1/+simple/triton/)
- [本工程精确 cp311 wheel 下载](https://pypi.sourcefind.cn/das_release/dtk2604-rc1/+f/e39/7b16365da2250/triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl)

文件名：triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl

SHA256：e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581

2026-11-10 已实际下载并校验，文件为 105484167 字节，哈希与 env/vendor-wheels.sha256 及此前成功推理的原包一致。不要用通用 PyPI Triton、cp310 或其他构建号替换。

## 自动下载并校验

在仓库 h3/ 目录执行：

    python3 download_vendor_triton.py

默认写入 downloads/vendor/，只下载，不安装、不导入 Torch、不使用 DCU。可在联网 CPU 节点或本地下载后复制到离线节点：

    python3 download_vendor_triton.py --output-dir /path/to/wheels

脚本复用哈希正确的已有文件；已有文件哈希不符则停止并保留原文件。新下载内容必须通过 SHA256 才成为最终 wheel。人工核验命令为：

    sha256sum downloads/vendor/triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl

## 新环境安装与兼容补丁

已有验证通过、打过补丁的个人 .venv 应继续复用，跳过重新安装以免覆盖补丁。以下用于新环境，匹配的厂商 Torch 和 DTK 需预先由平台提供或另行准备：

    cd /path/to/songshanbei/h3
    python3.11 -m venv --system-site-packages .venv
    .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11); assert sys.prefix != sys.base_prefix; print(sys.executable)'
    .venv/bin/python -m pip install --no-deps --no-index downloads/vendor/triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
    .venv/bin/python apply_triton_compat_memory.py
    .venv/bin/python scripts/prepare_triton_compat.py --check
    .venv/bin/python -c 'from importlib.metadata import version; import triton, triton.backends.amd.compiler; print(version("triton")); print(triton.__file__)'

元数据应显示完整厂商版本，模块路径应在本工程 .venv/ 内，补丁检查应显示 compatibility_patch_present: true。triton.__version__ 可能仅显示 3.5.1，以包元数据为准。

校验的是安装前的原 wheel；LLVM 17 GEP 兼容补丁只写个人 .venv 中的 triton/backends/amd/compiler.py，不修改原 wheel 或系统环境。运行时由工程激活脚本启用 DPA4_TRITON_LLVM17_COMPAT=1。导入与静态补丁检查不等于设备代码生成验证；后者仍在获分配的 DCU 节点执行。

若厂商将来移动文件，从上述索引查找同一完整文件名并坚持校验固定 SHA256，不静默替换版本。本说明补齐 Triton 获取，不包含 Torch/DTK 安装，也不代表全新环境整套重建已经验收。
