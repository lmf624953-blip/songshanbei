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

## 海光 Triton 厂商包获取

已补齐 [官方精确下载链接、SHA256 及个人环境安装步骤](docs/TRITON_VENDOR.md)。在 `h3/` 运行 `python3 download_vendor_triton.py` 即可下载并校验实际使用的 cp311 厂商 wheel；该工具不自动安装或覆盖现有补丁。新环境需先按该文档准备 Triton，再执行依赖准备。

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

## 在全新目录重建

使用平台原生八卡 BW1000/gfx936 镜像：CPython 3.11、海光 Torch 2.9.0+das.opt1.dtk2604、DTK 26.04、GCC/G++ 11.4。必须可读共享 Diffusers 模型 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`。本工程不包含权重，不覆盖系统 Torch，不向共享模型写文件。不是面向任意 Linux/CUDA 镜像的安装包。

重建分为五个阶段：创建个人虚拟环境；下载和安装校验过的依赖；应用 Triton 和框架补丁；从原始源码编译 InfiniCore/InfiniLM；进行九齿算子检查、正式生成、独立参考和比较。新增的 `rebuild.py` 串联这些阶段。下载与 Python 准备可在登录节点完成；设备编译和推理必须在分配的 DCU 节点完成。

原始输入缓存包含六个公开 Release 文件及一份海光原厂 Triton wheel，精确 URL、字节数、SHA256 和目标路径见 `rebuild/inputs.lock.json`。Triton 从海光原厂获取，优先复用已经校验的原包缓存。缓存只用于原始源码、wheel 和模板图片，不复用旧框架二进制或旧虚拟环境。

先在 `h3/` 设置路径：

```bash
export AFS_BASE="$IFAB_AFS_HOME"
export H3_CODE="$PWD"
export H3_CACHE="$AFS_BASE/h3-input-cache"
export H3_DEST="$AFS_BASE/clean-rebuilds/official-$(date +%Y%m%d-%H%M%S)"
printf 'export H3_CODE=%q\nexport H3_CACHE=%q\nexport H3_DEST=%q\n' \
  "$H3_CODE" "$H3_CACHE" "$H3_DEST" > "$AFS_BASE/h3-rebuild-paths.sh"
```

CPU 登录节点只准备依赖：

```bash
python3.11 rebuild.py --destination "$H3_DEST" \
  --cache-root "$H3_CACHE" --download-inputs --phase prepare
```

目标目录必须尚不存在。成功输出 `PREPARE_COMPLETE`，并写入目标目录的 `artifacts/rebuild-prepare-status.json`，其中 `complete=true`。准备程序通过 Python 自带 bundled pip 引导个人 pip；pip 的平台兼容只在个人安装进程中生效。NineToothed、ntops、Diffusers 从固定源码重新安装；不读取旧安装树。

在已分配的八卡计算终端中加载先前保存的路径，编译框架：

```bash
source "$IFAB_AFS_HOME/h3-rebuild-paths.sh"
python3.11 "$H3_CODE/rebuild.py" --destination "$H3_DEST" \
  --cache-root "$H3_CACHE" --phase build
```

成功输出 `BUILD_COMPLETE` 后执行推理验收；如果当前实例剩余时间不足，可先释放资源，再在新八卡实例中执行同一条命令：

```bash
source "$IFAB_AFS_HOME/h3-rebuild-paths.sh"
python3.11 "$H3_CODE/rebuild.py" --destination "$H3_DEST" \
  --cache-root "$H3_CACHE" --phase accept
```

验收入口会先核对本次新编译包和持久原生库的哈希，再启动设备验证，不会复制历史成功环境的框架二进制。

资源使用时限足够时，也可以用 `--phase compute` 合并编译与验收，或在计算节点用 `--phase all --download-inputs` 从空目录执行全程。当前平台每次最长 60 分钟，推荐保留可分配到不同实例的 prepare、build、accept 三阶段入口。计算阶段在节点提供的可写 `TMPDIR` 中解包原始源码，应用全部 Core/LM 补丁，构建 xmake 2.9.8，再以 96 路构建框架。Python 扩展必须匹配 cp311 和当前 Torch C++ ABI；完整 Python 包树安装到新 `.venv`，包含 `infinilm.multimodal`。原生库保存到新 `.infini`，不能留下指向已销毁节点目录的链接。推理前按仓库模型哈希清单只读核对全部 63 个模型文件，将结果写入个人 artifacts。随后构建注意力桥、执行九齿 lowering/六组 gfx936 代码生成/八卡数值检查，再依次生成正式视频、参考视频并调用官方比较程序。

最终必须输出 `CLEAN_REBUILD_ACCEPTED`，并且 `artifacts/clean-rebuild-acceptance.json` 中 `complete=true`、两侧推理与比较退出码均为 0。只看到构建 100%、导入成功或旧样例证据不算完整验收。检查输出目录中的两个视频都为 124 帧、768×928、24fps，并播放确认内容。SSIM 保存为本次真实结果，不预填历史的 93.485%。

若平台无法访问 GitHub Release，可先在可联网机器运行 `rebuild/download_inputs.py --destination 本地缓存目录`，将完整缓存按相同相对目录上传个人 AFS，再省略 `--download-inputs`。现有文件仍逐一验证 SHA256，遇到不一致立即停止，不覆盖。完整缓存已包含匹配的 Triton 原包，准备阶段可复用它而无需访问厂商下载站。

平台网页单文件上传上限为 128MiB，参考 FlashAttention wheel 超过此限制。在下载机器将它分为 112MiB 片段：

```bash
FLASH=flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
mkdir -p h3-input-cache/downloads/flash-parts
split -b 112M -d -a 2 "h3-input-cache/downloads/$FLASH" \
  h3-input-cache/downloads/flash-parts/flash.part
```

将六个片段及其他六个原始输入按锁定清单的相对目录上传个人 AFS，不上传超限的完整 wheel。Linux 计算或登录终端中合并：

```bash
FLASH=flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
cat "$H3_CACHE"/downloads/flash-parts/flash.part?? \
  > "$H3_CACHE/downloads/$FLASH"
python3.11 "$H3_CODE/rebuild/download_inputs.py" --destination "$H3_CACHE"
```

最后一条命令会验证所有现有完整输入的字节数和 SHA256，并输出 `RAW_INPUT_CACHE_READY`；验证失败的文件会保留并停止，不会被默认为可用。所有输入齐备时不会再请求网络。其他原始输入均小于网页单文件上限。

诊断文件位于目标目录 `artifacts/`：`rebuild-prepare.log`、`rebuild-prepare-status.json`、`clean-native.log`、`clean-native-status.json`、`fresh-module-paths.json`、两个 `*-fresh-install.json`、`attention-bridge-device.json`、`clean-rebuild-acceptance.json`。视频和比较结果位于 `outputs/unified_fl2va/时间戳/`，导出副本在 `artifacts/clean-rebuild-exports/`。

验收状态更新：2026-11-11 正在新计算实例执行本轮重建；此候选版本尚未取得端到端通过结果，不能先行声明新环境验收成功。

## 独立参考依赖

正式九齿路径不使用厂商 FlashAttention 作为注意力实现。独立 PyTorch 参考需要海光 FlashAttention `2.8.3+das.opt1.dtk2604.torch290` 的 cp311 wheel，SHA256：

`24d4021bf4ace04cf3c417f60affebe134128c0626a5b136230989812e66c18a`

官方来源：`https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.8/flash_attn-2.8.3+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`。

当前平台参考进程通过 `.reference-runtime/python` 中指向原系统 Torch 的符号链接视图，在 Torch 库相邻位置加载该扩展；`run_unified_dog.py` 仅对 reference 子进程添加视图路径及动态库路径。视图和 wheel 解压目录需提前准备，不修改系统 Torch。参考准备步骤见 `docs/REFERENCE_SETUP.md`。

## 已知范围

本工程仍复用 Diffusers/PyTorch 的既有 VAE 等模型数值组件。视频 VAE 注意力已接九齿，其余解码计算未全部重写。模型权重、计算配置与比较脚本未修改。单次公开输入验证不覆盖所有输入，也不等于干净环境重建和长时间稳定性已经通过。

原始导出清单记录平台文件哈希；发布时仅把 `evidence/` 文本中的个人绝对工程路径规范为 `$PROJECT_ROOT`，数值及代码保持原样。新增包装和文档另列于 `RELEASE-MANIFEST.json`。上游许可应随所用固定源码保留。
