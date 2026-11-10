# MiniMax H3 FL2VA 初赛复现代码

本目录主体代码直接导出自 InfiniFabric 上实际完成统一小狗样例的个人工程，保留原文件字节。适配模块的 SHA256 已与成功运行记录核对，完整导出记录见 `PLATFORM-SOURCE-MANIFEST.json` 和 `docs/PLATFORM_PROVENANCE.json`。

## 实测结果

输入为报告模板中的小狗 PNG（800×965），提示词精确为 `A dog falls asleep`。八卡 gfx936，seed 42、50 步、124 帧、768×928、24fps。

```text
[result] MiniMax-H3 video comparison

  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 93.485%
```

正式推理、独立 legacy 参考和固定上游比较脚本均退出 0。完整指标及运行日志在 `evidence/unified-dog/`；输出示例在 `examples/`。SSIM 最低86.331%、最高98.290%。该脚本未内置赛事通过阈值，结果不代表主办方已经评审通过。

先执行 `python3 verify_source.py` 可核对仓库源码与平台成功运行记录的哈希。

## 文件与适配

- `songshan_h3/attention_kernel.py`：NineToothed 在线 softmax 注意力、GQA、头维72等布局适配。
- `songshan_h3/attention_bridge.cc`、`attention_bridge.py`：InfiniCore HYGON dense/varlen 调度与原生流交接。
- `songshan_h3/diffusers_attention.py`：H3 视频 VAE 注意力接入同一九齿实现。
- `scripts/launch_h3.py`：注册九齿调度后执行官方入口。
- `patches/InfiniLM/0002-*`、`0003-*`：VAE 精度、引擎生命周期和 PyTorch/原生流同步。
- `patches/InfiniCore/`、`patches/InfiniLM/0001-*`：Hygon 构建规则及 Python/pybind11 头文件接入。
- `patches/triton/0001-*`：九齿生成 LLVM IR 与 DTK LLVM17 的 GEP 标志兼容。
- `runtime_entry/`：实际运行的已打补丁 `minimax_h3.py` 与未修改的 `minimax_h3_compare.py`。
- `tests/unified-dog.json`：统一输入哈希和固定运行参数。模型权重、统一输入数据不在仓库重复打包。

上述源码保留原工程和上游来源。九齿内核、主机桥及模型流交接已经过八卡执行验证；`run.sh` 和 `prepare_runtime_layout.py` 是本次发布新增的薄包装，不改变设备数值计算。

## 已验证环境

Python 3.11.9，海光 PyTorch `2.9.0+das.opt1.dtk2604`，Triton `3.5.1+das.opt1.dtk2604.torch290`，DTK26.04，GCC/G++11.4，xmake2.9.8，原生八卡 BW1000 gfx936。

源码完整 commit 见 `env/source-pins.json`。模型固定内容版本 `42ed227ee7df40d41602854ae760620d6eb651fe`，运行使用平台只读路径 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`。厂商 Torch/Triton wheel 校验值在 `env/vendor-wheels.sha256`。

## 复用已部署平台环境

实际验证时，`scripts/`、`songshan_h3/`、`.venv/`、`.infini/` 都位于个人 `$IFAB_AFS_HOME` 工程根目录。已有成功环境不应从头重装。将同一版本源码置于该工程布局、准备模板原图后，在获分配的八卡 DCU 节点执行：

```bash
cd "$IFAB_AFS_HOME"
source scripts/unified_dog_env.sh
python -u scripts/run_unified_dog.py
```

`run_unified_dog.py` 就是此次平台实测脚本：检查原图哈希及八卡 gfx936，先运行 InfiniLM/九齿，再独立运行 legacy PyTorch，最后调用固定上游比较程序；自动保存两个视频、日志、参数、媒体解码结果和 SSIM JSON。

脚本需要 `H3_EXPORT_DIR` 指向一个现有可写导出目录（用来复制最终产物）：

```bash
export H3_EXPORT_DIR="$IFAB_AFS_HOME/exports/unified-dog"
mkdir -p "$H3_EXPORT_DIR"
bash run.sh
```

不要在共享 CPU 登录节点运行模型。程序会检查8卡与gfx936后才启动。

## 在新工程目录构建

以下步骤依据实际环境和原构建脚本整理，尚未在全新干净目录完成整套重建验收。需要预先提供匹配的厂商 wheel、DTK 和共享模型；不能仅凭下载本仓库就省略这些运行依赖。

```bash
git clone https://github.com/lmf624953-blip/songshanbei.git
cd songshanbei/h3
# 本子进程兼容原运行脚本的工程根目录变量
export IFAB_AFS_HOME="$PWD"
python3.11 -m venv --system-site-packages .venv
cp env/machine.local.example.sh env/machine.local.sh
```

在 `machine.local.sh` 填写实际 DTK_ROOT、SONGSHAN_GCC_ROOT，并将 SONGSHAN_STORAGE 配为本工程的 `$IFAB_AFS_HOME/h3-runtime`。厂商 Torch 可复用同一 Python 的系统安装；Triton 必须安装到个人 `.venv`，兼容补丁只能写个人环境。依赖准备需要网络，离线重建则需先完整准备固定源码、子模块和依赖。

```bash
source scripts/activate.sh
export PATH="$IFAB_AFS_HOME/.venv/bin:$PATH"
export SONGSHAN_STORAGE="$IFAB_AFS_HOME/h3-runtime"
bash scripts/prepare_dependencies.sh
# 检查/应用个人 Triton LLVM17 兼容补丁
python scripts/prepare_triton_compat.py --check
python apply_triton_compat_memory.py
python scripts/prepare_triton_compat.py --check
export SONGSHAN_BUILD_JOBS=96
bash scripts/build_frameworks.sh --check
bash scripts/build_frameworks.sh
python scripts/build_attention_bridge.py
python scripts/check_attention_bridge.py --devices=0,1,2,3,4,5,6,7
python prepare_runtime_layout.py --first-frame=/path/to/template-dog.png
export H3_EXPORT_DIR="$IFAB_AFS_HOME/exports/unified-dog"
mkdir -p "$H3_EXPORT_DIR"
bash run.sh
```

构建脚本应用全部 Core/LM 补丁。Core 的 `--ninetoothed=n` 关闭该上游提交的可选内置集成，本工程经自有主机桥向 HYGON MHA/varlen dispatcher 注册 NineToothed；不能据此判断运行时未用九齿。注意力数值代码位于 `attention_kernel.py`。

## 独立参考依赖

正式九齿路径不使用厂商 FlashAttention 作为注意力实现。独立 PyTorch 参考需要海光 FlashAttention `2.8.3+das.opt1.dtk2604.torch290` 的 cp311 wheel，SHA256：

`24d4021bf4ace04cf3c417f60affebe134128c0626a5b136230989812e66c18a`

官方来源：`https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.8/flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`。

当前平台参考进程通过 `.reference-runtime/python` 中指向原系统 Torch 的符号链接视图，在 Torch 库相邻位置加载该扩展；`run_unified_dog.py` 仅对 reference 子进程添加视图路径及动态库路径。视图和 wheel 解压目录需提前准备，不修改系统 Torch。参考准备步骤见 `docs/REFERENCE_SETUP.md`。

## 已知范围

本工程仍复用 Diffusers/PyTorch 的既有 VAE 等模型数值组件。视频 VAE 注意力已接九齿，其余解码计算未全部重写。模型权重、计算配置与比较脚本未修改。单次公开输入验证不覆盖所有输入，也不等于干净环境重建和长时间稳定性已经通过。

原始导出清单记录平台文件哈希；发布时仅把 `evidence/` 文本中的个人绝对工程路径规范为 `$PROJECT_ROOT`，数值及代码保持原样。新增包装和文档另列于 `RELEASE-MANIFEST.json`。上游许可应随所用固定源码保留。


## 海光 Triton 厂商包获取

[官方精确下载链接、SHA256 与个人环境安装步骤](docs/TRITON_VENDOR.md) 已补齐。在 h3/ 执行 `python3 download_vendor_triton.py`，会下载并校验实际使用的 cp311 厂商 wheel；该工具只下载，不自动安装或覆盖现有补丁。新环境需先按该文档准备 Triton，再执行依赖准备。
