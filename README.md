# songshanbei

天气宝宝嵩山杯

MiniMax-H3 FL2VA 在 InfiniFabric 海光 BW1000 八卡平台的复现代码、补丁及验证记录位于 [h3/](h3/README.md)。

- [复现步骤与已验证环境](h3/README.md)
- [InfiniLM / InfiniCore / NineToothed 适配说明](h3/docs/ADAPTATION.md)
- [模板小狗样例的官方比较输出](h3/evidence/unified-dog/official-compare.log)
- [实测视频](h3/examples/unified-dog-infinilm.mp4)

统一输入：模板原始小狗图片，提示词 `A dog falls asleep`；124 帧、768×928、24fps、50 步、seed 42。官方脚本平均 SSIM 为 **93.485%**。源码直接取自成功运行的个人工程，提供文件哈希校验；全新干净环境的完整重建尚未复验。共享模型权重不在仓库重复打包。
