# MiniMax H3 按文件列出的适配改动

## 九齿注意力实现

songshan_h3/attention_kernel.py：以固定 ntops 实现为基础组织 NineToothed arrangement 与 application。序列方向采用 64 元素分块，头维按 128 宽 tile 排列，以覆盖视觉编码器 72 维注意力头及 64/128 维需求。通过实际维度及源张量边界掩码排除填充元素，不改变模型的实际头维。分组排列按查询头与 KV 头的比例映射，支持 GQA。

计算采用在线 softmax：逐块计算 QK，维护行最大值和归一化和，以 FP32 累积加权 V，减少显式中间注意力矩阵的存储。因果掩码按 InfiniCore 变长注意力 ABI 的右下对齐语义处理，支持查询长度不大于键长度的情形。掩码表达避开当前九齿前端不接受的分支局部定义。使用 ninetoothed.make 经九齿编译链生成代码，目标为 hygon-bw1000，num_warps=4、num_stages=1；没有手工改写生成后的 Triton/CUDA 数值内核。

### InfiniCore 调度与设备流接入

songshan_h3/attention_bridge.cc：新增仅承担主机调度的 pybind11 扩展，为 HYGON 设备注册 MultiheadAttention 与 MultiheadAttentionVarlen 的 plan、run、cleanup 回调。回调将 InfiniCore 张量包装为 ATen 视图，传递形状、布局、缩放系数和当前原生流句柄，设备数值计算仍由九齿内核执行。对 ALiBi、paged KV 和设备图录制等未支持能力显式拒绝，避免静默改变计算语义。

songshan_h3/attention_bridge.py：校验张量设备、FP16/BF16 类型、形状及头分组比例，将 [batch, sequence, heads, dimension] 视图转换为九齿所需排列；通过 torch.cuda.ExternalStream 在 InfiniCore 传入的流上启动内核。变长接口读取累计序列长度作为主机控制元数据，按序列片段调用同一因果注意力内核；支持非连续张量视图。

该文件的 install() 在原生多卡工作线程启动前构造 JIT 包装器，并在调用线程预先初始化 Triton HIP 驱动，避免八卡线程同时首次初始化引发加载锁与 GIL 交叉等待。scripts/build_attention_bridge.py 使用当前 PyTorch 的 C++ ABI、Python 开发头及个人 InfiniCore 库构建 _attention_bridge 扩展，保证主机桥与运行库匹配。

### 视频 VAE 注意力与统一启动入口

songshan_h3/diffusers_attention.py：将 diffusers.models.autoencoders.autoencoder_kl_minimax_h3 模块的 dispatch_attention_fn 接入同一九齿启动函数，覆盖视频 VAE 的 64 维注意力头及批量 latent chunk。适配器限定无掩码、非因果、dropout=0 的推理情形，对未实现参数明确报错；修改范围局限于 H3 视频 VAE 模块。

scripts/launch_h3.py：先注册 InfiniCore 九齿调度和 H3 视频 VAE 注意力，再通过 runpy 执行 InfiniLM/examples/minimax_h3.py，继续使用 fl2va 官方入口与原参数。此方式将算子适配集中在可审计模块中，保留模型加载、调度与输出流程。

### InfiniLM 补丁与跨框架执行正确性

patches/InfiniLM/0003-h3-runtime-stream-handoff.patch → python/infinilm/multimodal/minimax_h3.py。from_torch 共享存储并不会自动建立 PyTorch 生产流与原生工作流的先后关系。补丁在条件编码器 encode() 交给 forward_raw 前同步 PyTorch 当前流；在 Transformer 调用前同步各输入所在设备的当前流，覆盖 latent 更新、RoPE 缓存与行索引；在 _to_torch() 原生结果复制前及复制后同步设备，防止原生工作尚未完成就读取或复制结果。该处理优先保证多卡结果正确，后续可研究用更细粒度事件减少同步开销。

patches/InfiniLM/0002-h3-lifecycle-and-vae-precision.patch → csrc/engine/infer_engine.cpp、csrc/engine/infer_engine.hpp、csrc/pybind11/engine/engine.hpp、python/infinilm/multimodal/minimax_h3.py、examples/minimax_h3.py。

生命周期部分为 InferEngine 增加 close()，先关闭并释放 workers，再销毁通信组；通过 pybind 暴露 close 并释放 GIL；条件编码器、Transformer runner 和 Diffusers adapter 逐级提供关闭接口。_MiniMaxH3NativeQwen3VLModel 改为 weakref.proxy 引用编码器，解除循环引用。入口在输出完成后显式关闭两个原生引擎，使设备资源和通信对象在 Python 解释器退出前按顺序释放。

精度部分在 examples/minimax_h3.py 中将 pipe.vae.to(device=execution_device, dtype=DTYPE) 及 audio_vae 对应调用改为仅迁移设备，保留组件自身的 dtype 设置，避免对 VAE 整体强制套用统一计算精度，覆盖其需要保留 FP32 的部分。补丁同时增加 cleanup.native_engines 耗时项，使清理阶段可单独审计。

### 九齿生成代码与海光编译链兼容

patches/triton/0001-llvm17-gep-compat.patch → 个人 .venv 中 triton/backends/amd/compiler.py 的 HIPBackend.make_amdgcn。启用 DPA4_TRITON_LLVM17_COMPAT=1 时，在交给 DTK 26.04 的 LLVM 17 前去除 getelementptr 上不被支持的 nuw/nusw 优化标志，保留地址计算表达式及 inbounds 语义；可同时保存修改前后的 LLVM IR，记录内核名和哈希。

songshan_h3/compiler_compat.py 校验已安装后端包含该兼容开关，再启用该 pass 并记录后端文件哈希。兼容处理服务于九齿生成代码在 gfx936 上落地，不替换注意力算法。该工程未修改 NineToothed 或 ntops 仓库源码，也未提出已量化的编译器性能优化；新增数值实现集中在 attention_kernel.py。

### 框架构建接口补丁

patches/InfiniCore/0001-xmake-hygon-rule-order.patch → xmake/hygon.lua：将 Hygon 环境规则改为 add_deps("cuda.env", {order=true})，显式确定环境钩子的依赖顺序；检测 SONGSHAN_PYBIND11_INCLUDE 时复用指定头文件。

patches/InfiniCore/0002-local-pybind11-headers.patch → xmake.lua：允许通过 SONGSHAN_PYBIND11_INCLUDE 和 SONGSHAN_PYTHON_INCLUDE 使用当前个人 Python 环境的 pybind11 与 Python 头文件；未配置时保留原依赖方式。对应固定提交中没有 Boost 头文件使用的本地构建分支不再额外拉取该包。

patches/InfiniLM/0001-local-pybind11-headers.patch → xmake.lua：同样接入本地 pybind11/Python 头文件，并把扩展规则由 python.module 改为 python.library，保留 soabi 配置。以上构建适配用于生成匹配本机海光工具链和 Python 环境的原生库，不改变模型计算。

### 正确性验证与当前实现范围

scripts/check_ninetoothed_lowering.py 验证九齿 lowering、参数约定和 JIT 包装；scripts/check_attention_codegen.py 覆盖头维 64、72、128 及 causal 开关，共六组 gfx936 代码生成；scripts/check_attention_bridge.py 在八张设备逐一验证 dense、GQA、非连续布局、右下对齐因果注意力、变长序列及视频 VAE 注意力，与 PyTorch MATH 参考比较，容差 rtol=0.04、atol=0.03，八卡检查均通过。完整生成进一步验证算子接入和跨框架数据交接。

本工程仍复用 Diffusers/PyTorch 的既有 VAE 等模型数值组件，未将全部模型算子重写为九齿。视频 VAE 注意力已接入九齿，但其余神经网络解码计算仍属于既有组件。正式申报应按这一实际边界列明第三方组件。模型权重和统一比较程序未修改。


