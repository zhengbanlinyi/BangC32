# openoperator-start-kit

赛事官网->https://openoperator.cn

快捷监控->http://43.143.241.66:13000

此仓库是openoperator赛事举办方提供的模板仓库。选手可以直接Fork此仓库作为自己队伍的仓库。

## Quick Start

1. Fork此仓库，将仓库可见范围设置为private，将bot帐号加入collaborators
2. 点击`settings`->`webhooks`->`Add webhook`，配置webhook
   1. `Payload URL`填写`http://43.143.241.66:8000/webhook`
   2. `Content type`选择`application/json`
   3. `Secret`填写分配到的密钥（参赛信息收集完成后会分发随机的`webhook secret`）
   4. `SSL verification`选择`Disable`
   5. `Which events...`选择`Just the push event`
   6. 勾选`Active`
   7. 点击`Add webhook`
3. Clone仓库，随便在README.md中写点什么
4. 使用`git push`将修改推送到`github`，恭喜你完成了第一次代码提交！
5. 点开`xx Commits`提交记录页面，边刷新边等待一会，如果系统此时不太忙碌，大约1～3分钟后你就可以在该次提交的评论处看到系统反馈的结果和运行日志。

## 提交说明

### 流程说明

配置好webhook后，当仓库发生push操作，github会向远程服务器发送提交信息。远程仓库检查webhook secret有效性后，拉取仓库更新，执行评估脚本。无论结果如何，执行结束后该次commit评论区会收到日志。如果该次提交的某道题目跑分优于你在该道题目上的历史最好成绩，排行榜的该道题目成绩会更新（排行榜检查是否有新的最好成绩的间隔为5分钟）。

### 仓库结构

提交时仓库根目录需要包含`config`文件和题目的`mlu`代码文件。文件组织结构如下

```bash
.
├── config			# 配置文件，用于指定要评估的题目
├── LeakyReLU.mlu	# bangc代码文件，必须包含kernel函数定义和用于外部程序调用的函数定义
├── ...				# 其他题目的bangc代码文件
└── README.md		# 可选的代码说明
```

> [!NOTE]
>
> 通过`config`文件可以指定本次提交想要评估的题目范围
> config文件的每行代表一个题目，应按照题目序号的三位数字给出
> 例如，LeakyReLU的序号是001，为了评估LeakyReLU题目，config中必须包含一行`001`

> [!TIP]
>
> 每道题目的评估耗时预计不少于30s，评估系统评估完所有题目后才会返回结果，请合理安排评估请求，尽量不要一次性评估太多题目。

> [!CAUTION]
>
> 如果提交中不包含config文件，则会默认评估所有题目！

### 代码要求

1. 代码文件必须以题目名称命名，这是评估脚本能找到你代码的关键要求。
2. 代码中要覆盖头文件引用，核函数定义和用于外部调用的函数定义。
3. 用于外部调用的函数名必须设置为bang_func，bang_func的返回值为`torch::Tensor`，输入参数包含`torch::Tensor input`和参考代码中`__init__`部分定义的其他参数，请参考LeakyReLU示例进行理解。

## 题目&打分

题目按照类别分为`basic`，`easy`，`medium`，`hard`。其中`basic`是必做题，其他类为挑战题。

打分有两个指标：

- 算子结果必须与参考结果误差不大于1e-2，精度达标后性能评估结果才有效
- 性能分数按照`bangc`代码硬件时间相对于`torch`的执行时间赋值

## 最佳实践

1. 每次只评估少量题目
2. 尽量在调试服务器debug，远程评估时通过阅读commit评论中的报错进行debug
3. 系统只接收`main`分支的提交，所以请分时开发或者做好分支管理
4. github评论是执行结束第一时间更新的，快捷监控可以查看目前评估进度，排行榜是周期性更新的，且只会记录团队历史最好成绩
<!-- retrigger 131-139 evaluation 2026-06-05T01:25Z -->
<!-- retrigger 070 evaluation after tls fetch failure 2026-06-05T03:34Z -->

## 正确性修复与回归验证

2026-11-13 修复了运行时修改 PyTorch 参考函数、返回预置答案及忽略实际输入的实现。
编号文件与同名算子文件同步更新。回归测试覆盖输入相关性、模型权重、数值边界、
非连续布局、掩码广播、重复索引、空输入及 MoE 容量截断。

本轮对照了同一赛事的两个公开实现，核对显式权重接口、路由索引和掩码语义，
并在本仓库中实现修复。

- [AICS-2025-63H/openoperator，74efc75](https://github.com/AICS-2025-63H/openoperator/tree/74efc75d4b14ae7bae66cf0e54a7d2b6b4b581ef)
- [zfginucas/openoperator-start-kit，7ace04b](https://github.com/zfginucas/openoperator-start-kit/tree/7ace04bf7f1265a0a06040a04b32caf7ec8bf649)

请求中截图列出的 PRO-2684/AICS-openoperator、JerryLiu369/openoperator-start-kit-theory
及 cotangents/openoperator-start-kit-team25 在本轮访问时返回 404。

### 参数接口

带学习参数的算子使用调用方提供的权重。以下旧接口只有维度信息，调用时会报告缺少参数，
可以改用表中的显式参数形式。其他新增重载保留在对应源码中。

| 算子 | 显式参数形式 |
| --- | --- |
| Linear | `bang_func(x, weight, bias)`，无偏置时传入空 Tensor |
| Embedding_lookup、Sparse_embedding | `bang_func(indices, weight)` |
| Embedding_bag_mean | `bang_func(indices, offsets, weight)` |
| Relative_position_encoding | `bang_func(weight, seq_len, max_seq)`，表形状为 `[2 * max_seq - 1, d_model]`，索引为 `i - j + max_seq - 1` |
| Cross_Attention | `bang_func(query, key_value, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, num_heads)`，输入采用 batch-first 布局 |
| Conditional_LayerNorm | 使用已有的完整 gamma/beta 权重及偏置重载，按 `LayerNorm(x) * gamma(cond) + beta(cond)` 计算 |
| LSTM_forward、GRU_forward | `bang_func(x, parameters, input_size, hidden_size, num_layers)`，输入为 `[batch, sequence, features]`，初始状态为零 |
| Weight_standardization | 使用已有的显式卷积权重重载 |

循环网络的 `parameters` 按 PyTorch 每层的 `weight_ih`、`weight_hh`、`bias_ih`、`bias_hh`
顺序排列。无偏置网络每层传入前两个权重。GRU 原有的两层完整参数重载继续用于两层网络。

`DropPath(x, drop_prob)` 按推理模式执行。三参数重载的 `training=true` 启用按样本随机丢弃，
并使用 PyTorch 当前随机数生成器。`Grid_sample(input, grid)` 使用双线性插值、零填充及
`align_corners=true`，三参数重载可显式选择 `align_corners=false`。

`Batch_norm_1D` 的六参数重载使用训练模式并更新传入的运行均值与方差。八参数重载可指定
`training` 和 `eps`。`cross_entropy` 读取传入的平滑系数、是否求平均和忽略索引，
本地词表计算接受空的进程组占位 Tensor 或标量零。

矩阵求逆、Cholesky、QR 和 SVD 使用真实的 ATen 线性代数例程。FP16 输入提升到 FP32
求解，再转换回输入精度。这四个算子的 MLU 后端支持与性能需要在目标环境验证。

### CPU 回归

测试依赖 Python 3.12、C++17 编译器、PyTorch CPU、pytest 和 ninja。可在仓库根目录安装并运行。

```bash
uv venv --python 3.12 .venv-test
uv pip install --python .venv-test/bin/python 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-test/bin/python pytest ninja numpy==1.26.4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-test/bin/python -m pytest -q tests/test_correctness.py
```

CPU 模式将修复后的核函数以单任务串行方式执行，验证设备算法的数学结果、宿主检查和
输入状态变化，并检查加载算子前后 PyTorch 参考函数的身份。本轮 94 项回归通过。
CNCC 编译、MLU 设备同步和真实硬件性能通过目标 MLU 环境验证。
