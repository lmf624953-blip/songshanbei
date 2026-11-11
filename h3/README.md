# MiniMax H3 FL2VA 在海光 BW1000 上的复现

本目录包含已经应用构建与运行补丁的完整 InfiniCore、InfiniLM、NineToothed、ntops、Diffusers 和 xmake 源码。无需再从框架上游仓库下载源码，也无需手动应用补丁。自定义九齿注意力、主机桥和流交接适配位于 `songshan_h3/`，推理入口位于 `runtime_entry/`。

## 公开源码与版本

`frameworks/InfiniCore/` 与 `frameworks/InfiniLM/` 已应用 `patches/` 中对应的全部补丁。NineToothed 与 ntops 的上游库本身未修改，自定义注意力通过 `songshan_h3/attention_kernel.py`、`attention_bridge.py/.cc` 接入。各组件的上游仓库、分支、固定 commit 及补丁清单见 `frameworks/UPSTREAM.json`；本仓库的提交号同时固定所有已适配源码。各组件原许可证随源码保留。

## 前置环境

需要平台 CPython 3.11、预装海光 Torch 2.9.0+das.opt1.dtk2604、DTK 26.04 `/opt/dtk`、GCC 11.4 `/usr`，以及已分配的八卡 BW1000/gfx936 计算节点。模型使用只读共享目录 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`。不安装系统 Torch，不修改公共模型配置或权重。

## 安装与验收

以下命令在仓库 `h3` 目录执行。`IFAB_AFS_HOME` 指向自己的个人 AFS 根目录；安装目标必须尚不存在。第一条准备个人虚拟环境、锁定依赖和兼容处理，第二条编译并验收。

```bash
export H3_INSTALL_ROOT="$IFAB_AFS_HOME/songshan_h3_runtime"
export H3_INPUT_CACHE="$IFAB_AFS_HOME/songshan_h3_inputs"
bash install.sh prepare
# 切换到获分配的八卡计算终端，重新设置上面两个变量及仓库路径
bash install.sh compute
```

准备成功输出 `PREPARE_COMPLETE`，`artifacts/rebuild-prepare-status.json` 的 `complete=true`。编译阶段在节点可写 TMPDIR 中编译源代码；Core 在节点本地安装后按字节写回个人 AFS，避免 AFS 不支持的 Xmake 文件复制操作。完整 Python 包与共享库持久保存于新安装目录。最终验收成功必须输出 `CLEAN_REBUILD_ACCEPTED`，并有 `artifacts/clean-rebuild-acceptance.json` 的 `complete=true`。

平台当前每实例最多 60 分钟。可分开执行，安装结果会保存在 AFS，第二个实例无需重编：

```bash
bash install.sh build
# 在新的八卡计算实例中
bash install.sh accept
```

如果运行时限充分，已分配的计算节点可以直接 `bash install.sh all`。CPU 登录节点仅运行 prepare，不执行 GPU 推理。

## 依赖获取

六项原始依赖输入的明确地址和 SHA256 在 `rebuild/inputs.lock.json`。Triton 使用海光厂商原包，由下载器按 SHA256 验证后安装，再应用个人 LLVM 17 兼容处理。其余 wheel 缓存和模板原图由本仓库公开 Release 获取；这些是安装依赖，完整框架源码已经直接提交在 Git 中，代码复现不依赖源代码压缩包。

平台出网不可用时，在可访问网络的电脑先执行：

```bash
python3 h3/rebuild/download_inputs.py --destination ./songshan_h3_inputs
```

将该输入目录按原结构上传到自己的 AFS，然后设置 `H3_INPUT_CACHE`。下载器会复用 SHA256 正确的缓存，不重复下载。FlashAttention 厂商 wheel 仅供独立参考推理使用，正式推理注意力仍经过九齿实现。

## 验收内容与结果

验收核对导出的适配源码、本次编译安装包及原生库哈希，直接使用平台官方共享模型，然后执行九齿 lowering、六组 gfx936 代码生成、八卡注意力数值检查、模板小狗正式推理、独立参考推理和官方比较程序。统一配置为 `A dog falls asleep`、seed 42、50 步、124 帧、768×928、24fps。

原工程 2026-11-09 的正式与参考推理、比较程序均退出 0，平均 SSIM 93.48490251477143%，证据在 `evidence/unified-dog/`。该历史结果不替代本轮新安装验收。2026-11-11 的新建个人环境已完成依赖准备、Core/LM 原生编译安装、六组 gfx936 代码生成、八卡注意力检查及官方小狗正式推理；正式进程退出 0，视频规格检查通过。正式入口计时 536.917 秒。本轮按要求沿用 2026-11-09 的 SSIM，不将历史分数表述为本轮重新测量；本轮参考进程由用户要求停止。完整参考推理与官方比较步骤仍由 accept 入口正常提供。

本轮结果保存在安装目标的 `outputs/unified_fl2va/时间戳/`，状态为 `run.json`，比较结果为 `similarity.json`；导出副本在 `artifacts/clean-rebuild-exports/`。验收时须核实两侧视频规格及内容，SSIM 使用当次实测值，不预填历史数值。
