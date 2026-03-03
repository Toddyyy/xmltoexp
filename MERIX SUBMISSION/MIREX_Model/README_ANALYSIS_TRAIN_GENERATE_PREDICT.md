# MIREX_Model：分析 / 训练 / 生成 / 预测脚本说明

本说明覆盖 `MERIX SUBMISSION/MIREX_Model` 目录内**涉及分析、训练、生成、预测**的程序与常用命令。内容按任务流组织，方便你从数据准备到训练、推理与分析快速定位脚本。

## 一、数据准备（Beat/断句相关）

### 1) 生成基础 Beat 级训练数据
**`build_mazurka_beat_npz.py`**
- 作用：由 `*_boundary_prob.csv` + Mazurka XML 生成 `*.npz`（`note_feats + beat_ids + boundary_probs + num_beats`）
- 默认输入/输出：
  - `boundary_dir`: `<repo>/out/mazurka_boundary_probs`
  - `xml_dir`: `<repo>/MazurkaBL-master/xml_scores`
  - `out_dir`: `./beat_data_mazurka`
- 关键参数：
  - `--beat_unit`（默认 `1.0`，四分音符单位）

示例：
```bash
python build_mazurka_beat_npz.py \
  --boundary_dir /path/to/mazurka_boundary_probs \
  --xml_dir /path/to/MazurkaBL-master/xml_scores \
  --out_dir ./beat_data_mazurka \
  --beat_unit 1.0
```

### 2) 生成 per-performer 断句数据
**`build_mazurka_beat_npz_performer.py`**
- 作用：在 `beat_data_mazurka` 基础上，结合 PELT 变点（`cp_list_from_R_PELT_*.txt`）生成每位演奏者的断句标签
- 默认输入/输出：
  - `base_npz_dir`: `./beat_data_mazurka`
  - `cp_dir`: `<repo>/MazurkaBL-master/change_points_data/cp_per_maz_rec_PELT`
  - `sones_dir`: `<repo>/MazurkaBL-master/sones`
  - `out_dir`: `./beat_data_mazurka_performer`

示例：
```bash
python build_mazurka_beat_npz_performer.py \
  --base_npz_dir ./beat_data_mazurka \
  --cp_dir /path/to/cp_per_maz_rec_PELT \
  --sones_dir /path/to/sones \
  --out_dir ./beat_data_mazurka_performer
```

### 3) 生成 per-performer + 多层级断句数据
**`build_mazurka_beat_npz_performer_levels.py`**
- 作用：基于 `*beat_time.csv` 的 tempo curve 自动分层断句，输出：
  - per-level `*.npz` 到 `./beat_data_mazurka_performer_levels`
  - 对应 boundary CSV 到 `./beat_data_mazurka_performer`
- 关键参数：
  - `--str_vec`：层级步长，例如 `3,2,2,2,2,2`
  - `--smooth_window`：tempo 平滑窗口
  - `--append_beat_features` / `--no_append_beat_features`

示例：
```bash
python build_mazurka_beat_npz_performer_levels.py \
  --beat_time_dir /path/to/MazurkaBL-master/beat_time \
  --xml_dir /path/to/MazurkaBL-master/xml_scores \
  --out_dir ./beat_data_mazurka_performer_levels \
  --csv_dir ./beat_data_mazurka_performer \
  --str_vec 3,2,2,2,2,2 \
  --smooth_window 3
```

### 4) 快速 smoke 数据（调试用）
**`prepare_beat_smoke.py`**
- 作用：从 `smoke_data/sample.json` 生成极小样本 `beat_data_smoke/sample.npz`

### 5) 单文件定制拼接（一次性脚本）
**`build_train_npz.py`**
- 作用：读取固定路径的 note npz 与 boundary CSV 生成训练 npz
- 注意：路径硬编码，使用前需改脚本内部路径

## 二、训练

### 1) Score-to-Performance 主模型训练
**`train.py`**
- 作用：训练 `ScorePerformer`，支持 tempo prediction
- 关键参数：
  - `--config`（必需）
  - `--resume` / `--pretrained`
  - `--experiment_name`
  - `--wandb`
  - `--device`

示例：
```bash
python train.py --config config.yaml
python train.py --config config.yaml --resume ./check/xxx/checkpoint.pt
```

### 2) Beat 级断句模型训练
**`train_beat.py`**
- 作用：训练 `BeatBoundaryModel`
- 关键参数：
  - `--config`（必需，通常是 `config_beat_mazurka_level*.yaml`）
  - `--level`（只用某一层级 `_L{level}.npz`）
  - `--sanity_batch`（打印批次统计后退出）
  - `--bias_only` / `--freeze_base`
  - `--pos_weight`

示例：
```bash
python train_beat.py --config config_beat_mazurka_level3.yaml --level 3
python train_beat.py --config config_beat_mazurka_level6.yaml --sanity_batch
```

### 3) 训练调度脚本
**`train.sbatch` / `train_level6.sbatch` / `train_level6_dist.sbatch`**
- 作用：HPC/Slurm 训练提交脚本（根据集群配置使用）

## 三、生成（推理/合成）

### 1) XML -> MIDI（一键生成）
**`generate_from_xml.py`**
- 输入：MusicXML
- 输出：MIDI
- 关键参数：`--xml_path` `--model_path` `--output_midi` `--composer_id` `--sequence_length` `--overlap_length` `--no_tempo_prediction` `--temperature` `--top_p`

示例：
```bash
python generate_from_xml.py \
  --xml_path input.xml \
  --model_path ./check/model.pt \
  --output_midi output.mid \
  --composer_id 2 \
  --sequence_length 512 \
  --overlap_length 256
```

### 2) Score tokens -> Performance tokens
**`generate_overlap.py`**
- 输入：`score_tokens.json`
- 输出：`performance_tokens.json`
- 常用于研究/调试中间结果

**`generate_new.py`**
- 与 `generate_overlap.py` 类似的生成脚本（ScorePerformer + tempo prediction）

## 四、预测（Beat 断句概率）

### `infer_beat.py`
- 输入：`note-level npz`（`note_feats` + `beat_ids`）
- 输出：`boundary probability CSV`
- 关键参数：
  - `--config`（模型与特征配置）
  - `--model_path`（checkpoint）
  - `--input_npz` / `--output_csv`
  - `--window_beats` / `--window_stride`（滑窗推理）
  - `--head`（`dist` 或 `prob`）
  - `--performer_id`（条件模型）

示例：
```bash
python infer_beat.py \
  --config config_beat_mazurka_level3.yaml \
  --model_path ./check/beat_mazurka/best.pt \
  --input_npz ./beat_data_mazurka/M06-1.npz \
  --output_csv ./out/pred_M06-1.csv \
  --head prob
```

## 五、分析与诊断

### 1) 数据统计与分布分析
**`dataset.py`**
- CLI：分析 onset/duration deviation 分布并保存统计与图
```bash
python dataset.py \
  --data_dir /path/to/json_data \
  --sequence_length 512 \
  --stride 256 \
  --batch_size 16 \
  --save_fig deviation_analysis.png \
  --num_workers 4
```

### 2) 对齐检查
**`check_beat_alignment.py`**
- 检查 `note npz` 与 `boundary_prob_by_beat.csv` 的对齐关系

**`check_mazurka_alignment.py`**
- 统一检查 `beat_data_mazurka`、boundary CSV、`beat_time.csv` 的对齐情况

### 3) Notebook 分析
- `analysis.ipynb` / `analysis copy.ipynb`：调用 `infer_beat.py` + 统计分布
- `peak_detection.ipynb`：从 boundary 概率中提取峰值与生成 CSV
- `rebuild.ipynb`：重建/调试用的工作流记录

## 六、常见输出目录
- 训练 checkpoint：`./check/*`
- Beat 数据：`./beat_data_mazurka*`
- 推理结果：`./out/*`
- 生成 MIDI：`./generate_results/*`

---

如果你需要，我可以根据你当前的工作流程再写一个更精简的“按任务步骤”版本，或者把上述内容合并进 `README.md`。
