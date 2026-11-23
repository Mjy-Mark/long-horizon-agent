# Long-Horizon Agent

这个项目在本地 WorkBench 子集上比较自由 ReAct 和受约束的任务执行流程。

## 当前方法

Baseline 使用自由的 ReAct 工具调用流程。

Improved 1 使用以下流程。

```text
结构化 Planner
    ↓
确定性事件选择
    ↓
TaskMemory
    ↓
ExecutionMonitor
    ↓
一次受保护的邮件发送
    ↓
最终状态验证
```

Improved 1 的准确名称是 `Structured Planner + Deterministic Executor + Task Memory + Execution Monitor`。当前实现由 Python 控制器执行搜索、事件选择、邮件准备、发送和状态验证。LLM 只生成结构化计划，当前流程没有让 LLM 根据工具观测进行多轮 ReAct 决策。

Improved 2 尚未实现。发送异常后的不确定状态、状态查询和有限恢复重试属于下一阶段。

当前 Planner 针对这个 WorkBench 子集任务提供 2023 年日期解释。`target_date` 会经过格式校验，任务专用的年份约束仍然属于当前基准适配逻辑，不能作为通用日期解析器的结论。

## 项目文件

- `workbench_subset_agent.py` 保存 Baseline 适配器。
- `workbench_planner_agent.py` 保存 Improved 1 控制器和本地 Qwen 计划入口。
- `planning_memory.py` 保存事件选择器和 `TaskMemory`。
- `execution_monitor.py` 保存发送指纹、参数检查和单次发送保护。
- `verify_workbench_subset.py` 根据最终邮箱状态验证任务结果。
- `experiments/run_task_set.py` 为固定任务集创建独立数据副本并运行 Baseline 与 Improved 1。
- `snapshots/baseline/` 保存 Baseline 文件、轨迹和 Git 状态。

## 本地测试

使用项目环境运行全部测试。

```bash
/home/mark/micromamba/envs/longagent/bin/python -m unittest discover \
  -s tests \
  -p "test_*.py" \
  -v
```

当前共有 29 项测试，覆盖无序候选、候选截断边界、TaskMemory 状态、错误收件人、错误标题、重复发送、查询重试、发送后写入再抛异常、最终状态为零封或多封邮件，以及固定任务集加载、验证和 Baseline 解析错误步骤中的工具调用。

## 离线验证

Improved 1 的离线记录位于 `experiments/manifests/improved_1_offline.json`。该记录使用本地测试夹具，不调用模型，结果为选择事件 `00000200`，新增 1 封目标邮件，验证器判定任务成功，额外副作用为空。

离线夹具结果仍然保存在独立 manifest 中，没有被标记为真实模型结果。真实 Qwen 运行结果保存在 `experiments/manifests/improved_1_qwen.json`。

## 真实 Qwen 单任务验证

真实 Qwen 运行使用主机侧 CUDA 和本地 `llama-server` 完成。服务配置为 Qwen3-4B Q4_K_M、llama.cpp build `b12534`、`temperature=0`、`max_tokens=256`，模型别名为 `longagent-qwen`。

本次单任务验证使用任务 `multi_domain_030`。控制器生成了 `target_date=2023-11-30` 的结构化计划，搜索返回四个候选活动，Python 选择事件 `00000200`，随后只执行一次受保护的 `email.send_email`。

验证结果如下。

- 新增邮件数量为 1。
- 收件人为 `fatima.khan@atlas.com`。
- 标题为 `Annual Budget Planning Session`。
- 正文为 `Remember to attend this event.`。
- 现有验证器返回 `task_success=true`。
- `harmful_side_effects` 为空。

完整轨迹位于 [runs/workbench-improved1-7b0c8169ff324a6da77ba541945ea194.json](runs/workbench-improved1-7b0c8169ff324a6da77ba541945ea194.json)，运行 manifest 位于 [experiments/manifests/improved_1_qwen.json](experiments/manifests/improved_1_qwen.json)。轨迹同时记录模型文件、量化版本、`smolagents` 版本、本地代码提交、WorkBench 源提交和采样参数。

## 本地 Qwen 入口

启动兼容 OpenAI API 的本地 `llama-server` 后，可以运行以下入口。

```bash
/home/mark/micromamba/envs/longagent/bin/python workbench_planner_agent.py
```

运行轨迹会写入 `runs/workbench-improved1-<uuid>.json`。真实运行记录包含模型文件、量化版本、llama.cpp 版本、采样参数、任务文本、初始沙箱状态和代码提交。

当前已完成一次真实 Qwen 验证。最新轨迹为 `runs/workbench-improved1-7b0c8169ff324a6da77ba541945ea194.json`，使用主机侧 CUDA 运行 Qwen3-4B Q4_K_M。

## 候选集完整性边界

当前 Baseline 适配器的 `search_events()` 返回 `events[:5]`。`select_first_event()` 只能从工具实际返回的候选集中选择最早活动。当前 `2023-11-30` 有四场活动，因此任务选择了 `00000200`。当某一天有六场或更多活动时，最早活动可能位于工具返回上限之外。

当前结论应表述为：

> 当前实现能够选择工具返回候选集中的最早活动。对于超过工具返回上限的日期，尚未实现候选结果完整性保证。

后续可以增加分页、完整性标志或适配器级别的完整日期查询接口。

当初始邮箱已经存在相同发送指纹时，当前 Improved 1 会阻止发送并将任务记录为失败。`already_satisfied`、`duplicate_blocked`、`send_uncertain` 和 `send_failed` 的区分留给 Improved 2 的状态恢复设计。

## 固定对照任务集

固定任务集位于 `experiments/tasks.jsonl`。当前包含三条任务。

- `email_first_event_001` 验证 2023-11-30 的首事件发送。
- `email_first_event_late_date_001` 验证 2023-12-21 的首事件发送。
- `email_no_matching_event_001` 验证没有匹配活动时不发送邮件。

每条任务都定义了 `target_date`、期望行为、期望事件、ground truth 动作和 `fresh_workbench_subset` 初始状态。运行器每次将 `benchmarks/workbench_subset/` 复制到临时目录，记录日历和邮件 CSV 的 SHA-256，只把 `instruction` 传给 Agent 或 Planner，最后才读取 ground truth 进行验证。

在本地 `llama-server` 就绪后运行固定对照。

```bash
/home/mark/micromamba/envs/longagent/bin/python experiments/run_task_set.py
```

每个方法和任务都会生成独立轨迹，批量结果写入 `experiments/results/task-set-<batch_id>.json`，汇总 manifest 位于 `experiments/manifests/task_set_20261123T052134Z-377b3633.json`。

最终三任务真实 Qwen 对照结果如下。

- Baseline 的验证器任务成功率为 `2/3`，框架成功率为 `3/3`，平均环境工具调用数为 `11/3=3.67`，总解析工具调用平均数为 `5.0`。
- Improved 1 的验证器任务成功率为 `3/3`，框架成功率为 `2/3`，平均环境工具调用数为 `5/3=1.67`，总解析工具调用平均数同为 `1.67`。
- Baseline 的 Invalid Action Rate 为 `4/15`，其中 `15` 是所有解析工具调用，Invalid Send Rate 为 `4/6`，其中 `6` 是发送调用。Improved 1 的两项指标均为 `0`，发送调用数为 `2`。
- Baseline 的 `final_answer` 调用数为 `4`，Improved 1 没有 `final_answer` 工具调用。
- Baseline 的额外副作用出现在 `email_first_event_001`，最终新增 5 封邮件，其中 3 封是错误目标邮件，1 封是正确目标邮件的重复发送，总额外邮件数为 4。
- Improved 1 在两个发送任务中各新增 1 封匹配邮件，在无匹配任务中保持零发送。该任务的控制器框架状态为失败，最终任务验证结果为成功，因为预期行为是安全终止且不发送。

批量结果中的 `average_environment_tool_calls` 排除 `final_answer`，`average_tool_calls` 包含全部解析工具调用。`invalid_action_rate` 使用全部解析工具调用作为分母，`invalid_send_rate` 使用发送调用作为分母。结果可以通过以下命令从已有轨迹重算，不需要再次调用模型。

```bash
/home/mark/micromamba/envs/longagent/bin/python experiments/run_task_set.py \
  --tasks experiments/tasks.jsonl \
  --recompute-results experiments/results/task-set-20261123T052134Z-377b3633.json
```

## WorkBench 范围

仓库中的 `benchmarks/workbench_subset/` 只保留一条日历和邮件任务所需的数据。它用于本地适配和控制器验证，不代表完整 WorkBench 数据集的正式成绩。`verify_workbench_subset.py` 检查最终新增邮件数量和邮件字段，不等同于 WorkBench 官方评分流程。

## Baseline 记录

Baseline 轨迹显示原始自由 ReAct 流程新增了 5 封邮件，其中 1 封匹配目标邮件，其他邮件属于额外副作用。Baseline manifest 位于 `experiments/manifests/baseline.json`。

## 当前状态

Improved 1 的本地控制器、TaskMemory、ExecutionMonitor、29 项测试、固定任务集运行器、离线记录、一次真实单任务验证、三任务真实批量对照和指标重算已经完成。Recovery 实现暂不在当前阶段范围内。
