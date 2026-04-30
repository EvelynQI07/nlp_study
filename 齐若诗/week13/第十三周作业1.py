【比较deepseek2和3】
一、架构创新 1. 无辅助损失的负载均衡策略（Auxiliary-Loss-Free Load Balancing）
- V2 ：依赖辅助损失（auxiliary loss）来鼓励专家负载均衡，但可能损害模型性能
- V3 ：创新性地采用无辅助损失策略，通过动态调整专家偏置项（bias term）来实现负载均衡，避免了辅助损失对模型性能的负面影响 2. 多token预测（Multi-Token Prediction, MTP）
- 新增功能 ：在训练中预测多个未来token，增强数据效率和模型的前瞻规划能力
- 可用于推理加速（speculative decoding）
二、模型规模扩展
模型 总参数 每token激活参数 
V2    236B   21B 
V3    671B    37B

三、训练效率优化 1. FP8混合精度训练框架
- 首次在超大规模模型上验证FP8训练的可行性
- 精细量化策略（tile-wise/block-wise）提升精度
- 低精度存储和通信减少内存占用 2. DualPipe算法
- 创新的流水线并行算法，减少流水线气泡
- 实现计算-通信重叠，降低跨节点MoE通信开销 3. 高效跨节点全对全通信
- 充分利用InfiniBand和NVLink带宽
- 仅需20个SM即可完全利用通信带宽

 四、训练数据改进
- 数据量 ：从V2扩展到 14.8万亿tokens
- 质量提升 ：增加数学和编程样本比例
- 多语言扩展 ：超越中英双语，扩展更多语言覆盖
- FIM策略 ：采用Prefix-Suffix-Middle框架增强代码理解能力

五、上下文长度扩展
- 通过YaRN方法分两阶段扩展：
  1. 4K → 32K
  2. 32K → 128K
- 在"Needle In A Haystack"测试中表现出色

六、后训练优化 1. 推理能力蒸馏
- 从DeepSeek-R1模型蒸馏推理能力
- 将R1的验证和反思模式融入V3 2. 改进的RL流程
- 采用Group Relative Policy Optimization (GRPO)
- 规则+模型混合奖励机制

七、显著的性能提升
从论文中的对比数据可以看出，V3在多个基准测试中大幅超越V2：
基准	V2	V3	提升
MMLU	78.4	87.1	+8.7
MMLU-Pro	51.4	64.4	+13.0
HumanEval	43.3	65.2	+21.9
MATH	43.4	61.6	+18.2
GSM8K	81.6	89.3	+7.7

八、训练成本优化
- 总训练成本 ：仅278.8万H800 GPU小时（约557.6万美元）
- 训练效率 ：每万亿token仅需18万H800 GPU小时
- 稳定性 ：全程无损失尖峰或回滚
