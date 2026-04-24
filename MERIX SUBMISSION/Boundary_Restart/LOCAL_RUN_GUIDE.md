# 本地实验运行指南

这些脚本帮助你在本地计算机上逐个运行SBATCH脚本中的所有实验。

## 📋 脚本说明

### 0. **test_quick_cpu.sh** (CPU专用 - 首先运行！)
快速测试脚本，在30-60分钟内验证全部pipeline是否正常工作。

```bash
bash test_quick_cpu.sh
```

功能：
- 单个seed快速运行（epochs=10，不是完整的60）
- 跳过BiLSTM（在CPU上特别慢）
- 只测试TCN和Baselines
- 检查结果完整性

**推荐：先运行这个脚本！** 可以快速确认环境配置是否正确。

### 1. **run_interactive.sh** (推荐用于首次使用)
交互式脚本，提示你选择要运行的实验配置。

```bash
bash run_interactive.sh
```

选项：
- 选择要运行的seed (42-46 或自定义)
- 选择要运行的模型组件 (TCN/Baselines/BiLSTM)
- 选择计算设备 (CPU/GPU)

### 2. **run_experiments_locally.sh** (直接运行或命令行调用)
主运行脚本，支持完整的命令行配置。

#### 基本用法

```bash
# 默认运行所有seed和所有组件（使用CPU）
bash run_experiments_locally.sh

# 只运行特定seed
bash run_experiments_locally.sh --seed 42

# 跳过某些组件
bash run_experiments_locally.sh --skip-tcn --skip-summary

# 从特定seed恢复（如果中途中断）
bash run_experiments_locally.sh --resume-from 44

# 组合使用
bash run_experiments_locally.sh --seed 42 --skip-bilstm
```

#### 命令行选项

```
--seed SEED              仅运行指定的seed（default: 42 43 44 45 46）
--skip-tcn              跳过TCN实验
--skip-baselines        跳过基线实验
--skip-bilstm           跳过BiLSTM实验
--skip-summary          跳过摘要生成
--resume-from SEED      从指定seed恢复运行
```

#### 环境变量配置（CPU优化）

由于你使用CPU，脚本已自动优化：
- TCN: 仍然使用 epochs=60（核心模型）
- BiLSTM: 自动降低到 epochs=30, batch_size=32（加快速度）

你还可以通过环境变量进一步优化：

```bash
# 减少epoch数（快速测试）
EPOCHS=20 BILSTM_EPOCHS=15 bash run_experiments_locally.sh --seed 42

# 使用自定义配置文件
CONFIG="configs/my_config.yaml" bash run_experiments_locally.sh

# 不重用现有结果（从零开始）
REUSE_EXISTING=0 bash run_experiments_locally.sh

# 限制内部交叉验证折数（加快速度）
MAX_INNER_FOLDS=3 bash run_experiments_locally.sh

# 减少batch size（节省内存）
BATCH_SIZE=16 bash run_experiments_locally.sh
```

完整的可配置环境变量：
- `DEVICE`: 计算设备 (cpu/cuda，default: cpu)
- `EPOCHS`: 训练epoch数 (default: 60)
- `EARLY_STOP_PATIENCE`: 早停耐心值 (default: 10)
- `BATCH_SIZE`: 批大小 (default: 自动)
- `MAX_INNER_FOLDS`: 最大内部折数 (default: 无限制)
- `REUSE_EXISTING`: 是否重用现有结果 (default: 1)
- `CONFIG`: 配置文件路径 (default: configs/salience_grouped3_hi8_score_only_xml_curated.yaml)

## 🚀 使用示例（CPU优化）

### 示例 0: 快速验证（30-60分钟）⭐ 推荐首先运行
```bash
bash test_quick_cpu.sh
```
验证pipeline是否正常工作，节省时间。

### 示例 1: 单个seed完整运行（2-8小时）
```bash
bash run_experiments_locally.sh --seed 42
```
完整运行所有模型和组件，但只用一个seed测试。

### 示例 2: 后台运行，同时查看日志（推荐）
```bash
# 在后台运行
nohup bash run_experiments_locally.sh --seed 42 > my_exp.log 2>&1 &

# 在另一个终端查看进度
tail -f my_exp.log

# 或查看CPU使用
watch -n 2 'top -b -n 1 | head -n 15'
```

### 示例 3: 多seed快速运行（跳过BiLSTM）
```bash
bash run_experiments_locally.sh --skip-bilstm
# 约 3 seeds * 4 hours = 12 hours
```

### 示例 4: 减少epoch数进行快速原型测试
```bash
EPOCHS=10 BILSTM_EPOCHS=5 bash run_experiments_locally.sh --seed 42 --skip-summary
# 约 1-2 小时
```

### 示例 5: 仅运行特定组件
```bash
# 只运行TCN
bash run_experiments_locally.sh --skip-baselines --skip-bilstm --seed 42

# 只运行Baselines
bash run_experiments_locally.sh --skip-tcn --skip-bilstm --seed 42
```

### 示例 6: 从中断的地方继续（seed 44）
```bash
bash run_experiments_locally.sh --resume-from 44
```

### 示例 7: 使用交互式菜单
```bash
bash run_interactive.sh
# 选择seed、模块和设备
```

## ⏱️ 预计运行时间（CPU）

基于CPU性能的估计（MacBook Air/Intel i5及以上）：

| 配置 | 运行时间 | 说明 |
|------|--------|------|
| `test_quick_cpu.sh` | 0.5-1h | 快速验证 (1 seed, 10 epochs) |
| 单个seed完整版 | 2-8h | TCN + Baselines + BiLSTM |
| TCN only | 1-2h | 单个seed |
| Baselines only | 0.5-1h | 单个seed |
| 1 seed, 跳过BiLSTM | 1-2h | TCN + Baselines |
| 所有5个seeds | 10-40h | 完整运行 |

**建议策略**：
1. 先运行 `test_quick_cpu.sh` (~1小时) 验证环境
2. 然后运行完整的单个seed (`--seed 42`) (~4小时) 确认结果质量
3. 再在后台运行其他seeds

### 原有的 📊 输出和日志

运行时产生的文件：

- **experiment_run.log**: 详细的运行日志，包含所有output和error
- **.experiments_completed**: 追踪已完成的seeds（用于简化恢复）
- **reports/**: 各种模型和seed的结果目录

查看实时日志：
```bash
tail -f experiment_run.log
```

## ⚠️ 注意事项

### CPU 运行优化（重要！）

由于你没有GPU，请注意：

1. **运行时间较长**:
   - 单个seed的完整运行：**2-8小时**（取决于数据大小和CPU性能）
   - 所有5个seeds：**10-40小时**
   - 建议优先运行单个seed进行测试

2. **CPU优化策略**:
   ```bash
   # 方案 A: 快速测试单个seed（推荐首先尝试）
   bash run_experiments_locally.sh --seed 42

   # 方案 B: 在后台运行所有seeds，查看进度
   nohup bash run_experiments_locally.sh > experiment_run.log 2>&1 &
   tail -f experiment_run.log

   # 方案 C: 跳过耗时模块进行快速运行
   bash run_experiments_locally.sh --skip-bilstm --seed 42

   # 方案 D: 减少epoch进行快速验证
   EPOCHS=20 BILSTM_EPOCHS=10 bash run_experiments_locally.sh --seed 42
   ```

3. **监控资源占用**:
   ```bash
   # 在另一个终端窗口监控CPU使用
   watch -n 2 'top -b -n 1 | head -n 15'
   ```

4. **避免过载**:
   - 不要同时运行多个实验脚本
   - 关闭其他耗CPU的程序
   - 脚本会自动使用所有可用CPU核心

5. **如果内存不足**:
   ```bash
   # 减少batch size
   BATCH_SIZE=8 bash run_experiments_locally.sh

   # 或跳过某些组件
   bash run_experiments_locally.sh --skip-bilstm
   ```

### 原有的注意事项

1. **虚拟环境**: 脚本会自动激活 `$PROJECT_DIR/../MIREX_Model/.venv`。确保路径正确。

2. **计算资源**:
   - CPU模式：较慢但开销小
   - GPU模式：需要CUDA/GPU可用，速度快10倍+

3. **中断和恢复**:
   - 按 Ctrl+C 中断当前运行
   - 使用 `--resume-from SEED` 继续断点运行
   - 脚本在 `.experiments_completed` 中追踪进度

4. **重用结果**:
   - 默认情况下 (`REUSE_EXISTING=1`)，脚本会跳过已有的结果
   - 若要重新运行，设置 `REUSE_EXISTING=0`

5. **配置文件**: 确保配置文件存在：
   ```bash
   ls configs/salience_grouped3_hi8_score_only_xml_curated.yaml
   ```

## 🔧 故障排除

**问题**: 运行非常慢（CPU默认）
```bash
# 这是正常的！跑快速测试先验证环境
bash test_quick_cpu.sh

# 或在后台运行，同时做其他事
nohup bash run_experiments_locally.sh --seed 42 > exp.log 2>&1 &
tail -f exp.log
```

**问题**: "Missing virtualenv activate script"
```bash
# 解决: 检查虚拟环境路径
echo $VENV_PATH
# 或手动指定
VENV_PATH=/your/path/to/.venv bash run_experiments_locally.sh
```

**问题**: 运行很慢
```bash
# 使用GPU
DEVICE="cuda" bash run_experiments_locally.sh
```

**问题**: 运行很少的CPU核心
```bash
# 检查可用的CPU核心
sysctl -n hw.ncpu

# 脚本会自动使用所有核心，但可以手动限制（如需要）
taskset -c 0-3 bash run_experiments_locally.sh  # 仅使用4个核
```

**问题**: 想在后台运行并监控
```bash
# 启动后台任务
nohup bash run_experiments_locally.sh --seed 42 > exp_42.log 2>&1 &
JOB_PID=$!

# 检查进程
ps aux | grep $JOB_PID

# 实时查看日志
tail -f exp_42.log

# 查看CPU和内存使用
top -p $JOB_PID -n 1
```

**原有的 问题**: 想查看每个命令的详细信息
```bash
# 脚本已经通过日志文件保存所有output
tail -f experiment_run.log | grep "Command:"
```

## 📈 进度追踪

脚本会显示清晰的进度指示：

```
╔══════════════════════════════════════╗
║  Processing Seed: 42
╚══════════════════════════════════════╝

>>> Running TCN for seed 42
  ├─ Target: level1plus_boundary (min_precision=0.85)
  ├─ Target: level2plus_boundary (min_precision=0.85)
  ...

>>> Running Baselines for seed 42
  ├─ LogReg + weighted_topdown (all features)
  ...

✓ Seed 42 completed
```

## ℹ️ 更多信息

- 原始SBATCH脚本: `run_outer_all_baselines_seed42_46.sbatch`
- 项目目录: 自动检测为当前目录或 `$SLURM_SUBMIT_DIR`
