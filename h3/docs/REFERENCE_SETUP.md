# 独立 PyTorch 参考环境

在同一项目个人 `.venv` 中使用匹配的海光 Torch2.9/DTK26.04 栈。以下准备不修改系统 Torch，厂商注意力只用于参考子进程。

1. 从 README 中海光官方地址取得 cp311 wheel，验证 SHA256 为 `24d4021bf4ace04cf3c417f60affebe134128c0626a5b136230989812e66c18a`。不要重复下载已验证文件。
2. 将 wheel 解压到项目 `.reference-runtime/flash-attn`。扩展约2GiB，先确保个人存储空间。
3. 执行引用视图工具，之后由实测运行脚本仅向参考子进程传入视图路径。

```bash
cd "$IFAB_AFS_HOME"
sha256sum /path/to/flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
.venv/bin/python -m zipfile -e \
  /path/to/flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl \
  .reference-runtime/flash-attn
.venv/bin/python setup_reference_torch_view.py
```

工具针对实际平台 Torch 安装位置 `/usr/local/lib/python3.11/site-packages/torch` 建立引用视图。所有 Torch 数值代码仍指向系统原文件；仅将厂商 FlashAttention 扩展链接到加载器寻找的相邻目录。完整生成时 `run_unified_dog.py` 自动给参考进程设置 PYTHONPATH 和 LD_LIBRARY_PATH，正式 InfiniLM/九齿进程不添加这些参考目录。

本目录工具是对已验证平台准备步骤的脚本化记录；未替换 SDPA 算法，也未把九齿待评实现注入参考。
