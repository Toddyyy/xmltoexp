# 💻 CPU-only 快速开始指南

你的系统没有GPU，这里是为CPU优化的运行指南。

## 🎯 第一步：快速验证（强烈推荐！）

```bash
cd "MERIX SUBMISSION/Boundary_Restart"
bash test_quick_cpu.sh
```

**耗时**: ~30-60分钟
**作用**: 验证环境配置是否正确，pipeline是否能正常运行

如果这个通过了，说明你的环境没问题，可以继续完整运行。

---

## ⏱️ 完整运行时间预期

| 任务 | 耗时 | 推荐时机 |
|-----|------|--------|
| 快速测试 (`test_quick_cpu.sh`) | 30-60分钟 | **首次运行** |
| 单个seed完整版 (42) | 2-8小时 | 验证质量 |
| 所有5个seeds | 10-40小时 | 最终结果 |

---

## 🚀 建议执行计划

### 第一天
```bash
# 1. 快速验证（30-60分钟）
bash test_quick_cpu.sh

# 2. 完整运行单个seed（2-8小时，可在后台）
nohup bash run_experiments_locally.sh --seed 42 > exp_42.log 2>&1 &
tail -f exp_42.log  # 查看进度
```

### 验证通过后
```bash
# 后台运行其他seeds，同时做其他事
nohup bash run_experiments_locally.sh --seed 43 > exp_43.log 2>&1 &
nohup bash run_experiments_locally.sh --seed 44 > exp_44.log 2>&1 &
# ...依次运行其他seeds
```

---

## 🔥 常用命令快速参考

```bash
# ✅ 推荐！快速测试
bash test_quick_cpu.sh

# ✅ 单个seed完整版
bash run_experiments_locally.sh --seed 42

# ✅ 后台运行（推荐长期运行）
nohup bash run_experiments_locally.sh --seed 42 > exp.log 2>&1 &
tail -f exp.log

# ✅ 跳过慢的BiLSTM模块
bash run_experiments_locally.sh --seed 42 --skip-bilstm

# ✅ 交互式菜单选择
bash run_interactive.sh

# ✅ 减少epoch快速原型（1-2小时）
EPOCHS=10 bash run_experiments_locally.sh --seed 42

# ✅ 仅运行TCN（~1-2小时/seed）
bash run_experiments_locally.sh --seed 42 --skip-baselines --skip-bilstm

# ✅ 从中断处恢复
bash run_experiments_locally.sh --resume-from 44
```

---

## 📊 查看进度和结果

```bash
# 查看实时日志（后台运行时）
tail -f experiment_run.log

# 检查CPU使用率
watch -n 2 'top -b -n 1 | head -n 15'

# 检查生成的结果
ls -lh reports/
```

---

## 🛠️ 性能优化建议

### 如果感觉太慢：
```bash
# 1. 关闭其他程序（浏览器、编辑器等）
# 2. 减少epoch数
EPOCHS=30 bash run_experiments_locally.sh --seed 42

# 3. 跳过BiLSTM（最慢的部分）
bash run_experiments_locally.sh --seed 42 --skip-bilstm

# 4. 限制内部交叉验证折数
MAX_INNER_FOLDS=3 bash run_experiments_locally.sh --seed 42
```

### 如果内存不足：
```bash
# 减少batch size
BATCH_SIZE=8 bash run_experiments_locally.sh --seed 42
```

---

## ❓ 常见问题

**Q: 需要多长时间？**
A: 单个seed 2-8小时，所有5个seeds 10-40小时。先跑`test_quick_cpu.sh`快速验证。

**Q: 可以在后台运行吗？**
A: 可以！使用`nohup bash ... > log.txt 2>&1 &`在后台运行。

**Q: 运行中断了怎么办？**
A: 使用`--resume-from SEED`从断点继续。

**Q: 结果保存在哪里？**
A: `reports/`目录，按seed和模型组织。

---

## 📚 详细文档

查看完整的使用指南：
```bash
cat LOCAL_RUN_GUIDE.md
```

---

## 🎬 现在开始！

```bash
bash test_quick_cpu.sh
```

祝运行顺利！如有问题，查看 `LOCAL_RUN_GUIDE.md` 的故障排除部分。
