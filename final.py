import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from concurrent.futures import ThreadPoolExecutor
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import scipy.io

# 配置类，定义模型和训练的超参数
class Config:
    BASE_DIR = r"C:\Users\Administrator\PycharmProjects\6.23"  # 项目基础目录
    DATA_OUTPUT_DIR = os.path.join(BASE_DIR, 'data')  # 数据输出目录
    DATA_INPUT_DIR = os.path.join(BASE_DIR, 'data', 'model_data')  # 数据输入目录
    PROCESS_DATA_DIR = os.path.join(BASE_DIR, 'process_data')  # 预处理数据目录
    RESULT_DIR = os.path.join(BASE_DIR, 'result final3')  # 结果保存目录
    MODEL_DIR = os.path.join(BASE_DIR, 'models3')  # 模型保存目录
    CLUSTER_ANALYSIS_DIR = os.path.join(BASE_DIR, 'cluster_analysis final3')  # 聚类分析目录

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择（GPU或CPU）
    BATCH_SIZE = 8  # 批次大小
    NUM_WORKERS = 2  # 数据加载的线程数
    MAX_SEQ_LEN = 256  # 最大序列长度
    INPUT_DIM = 64  # 输入维度
    HIDDEN_DIM = 256  # 隐藏层维度
    NUM_LAYERS = 2  # LSTM层数
    DROPOUT_RATE = 0.4  # Dropout率
    ATTN_HEADS = 4  # 注意力头数
    COMMITMENT_LOSS_WEIGHT = 2  # 承诺损失权重
    KMEANS_INIT = True  # 是否使用K-Means初始化码本
    KMEANS_MAX_SAMPLES = 200000  # K-Means最大样本数
    KMEANS_BATCH_SIZE = 3072  # K-Means批次大小
    QUANTIZE_BLOCK_SIZE = 1000  # 量化块大小
    LEARNING_RATE = 0.0005  # 学习率
    EPOCHS = 40# 训练轮数
    WEIGHT_DECAY = 0.01  # 权重衰减
    ANALYSIS_FREQ = 5  # 分析频率
    CODEBOOK_MERGE_THRESHOLD = 1e-4  # 码本合并阈值（适配归一化后）
    GRADIENT_ACCUM_STEPS = 4  # 梯度累积步数
    WARMUP_EPOCHS = 8  # 学习率预热轮数
    CODEBOOK_DIVERSITY_WEIGHT = 0.15  # 码本多样性损失权重
    QUANTIZE_TEMP_INITIAL = 1.0  # 初始量化温度
    QUANTIZE_TEMP_FINAL = 0.5  # 最终量化温度

    def __init__(self):
        # 创建所有必要的目录
        for directory in [self.DATA_OUTPUT_DIR, self.DATA_INPUT_DIR, self.PROCESS_DATA_DIR,
                          self.RESULT_DIR, self.MODEL_DIR, self.CLUSTER_ANALYSIS_DIR]:
            os.makedirs(directory, exist_ok=True)

# 预处理CSV文件，生成BPM序列和填充掩码
def preprocess_csv(csv_file, output_dir, max_seq_len=256):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_file}")
    data = pd.read_csv(csv_file)
    if 'padding_mask' not in data.columns:
        raise ValueError(f"CSV 文件缺少 'padding_mask' 列: {csv_file}")
    bpms, padding_masks = [], []

    for idx in range(len(data)):
        bpm = [float(x) for x in data.iloc[idx]['bpm'].split(';')]  # 解析BPM序列
        padding_mask = [float(x) for x in data.iloc[idx]['padding_mask'].split(';')]  # 解析填充掩码
        if len(bpm) != len(padding_mask):
            raise ValueError(f"序列长度不匹配，索引 {idx}")
        if len(bpm) < max_seq_len:
            bpm = bpm + [0.0] * (max_seq_len - len(bpm))  # 填充BPM序列
            padding_mask = padding_mask + [0.0] * (max_seq_len - len(padding_mask))  # 填充掩码
        bpms.append(bpm)
        padding_masks.append(padding_mask)

    bpms = np.array(bpms, dtype=np.float32)  # 转换为NumPy数组
    padding_masks = np.array(padding_masks, dtype=np.float32)

    base_name = os.path.basename(csv_file).replace('.csv', '')  # 获取文件名
    np.save(os.path.join(output_dir, f'{base_name}_bpm.npy'), bpms)  # 保存BPM数据
    np.save(os.path.join(output_dir, f'{base_name}_padding_mask.npy'), padding_masks)  # 保存掩码
    return data

# 数据集类，加载预处理数据并进行数据增强
class PhraseDataset(Dataset):
    def __init__(self, npy_dir, split, csv_file):
        # 检查必要的npy文件是否存在
        for file in [f'pre_data_{split}_bpm.npy', f'pre_data_{split}_padding_mask.npy']:
            if not os.path.exists(os.path.join(npy_dir, file)):
                raise FileNotFoundError(f"未找到文件: {os.path.join(npy_dir, file)}")
        self.bpms = np.load(os.path.join(npy_dir, f'pre_data_{split}_bpm.npy'), mmap_mode='r')  # 加载BPM数据
        self.padding_masks = np.load(os.path.join(npy_dir, f'pre_data_{split}_padding_mask.npy'), mmap_mode='r')  # 加载掩码
        self.metadata = pd.read_csv(csv_file)  # 加载元数据

    def __len__(self):
        return len(self.bpms)

    def __getitem__(self, idx):
        bpm = self.bpms[idx]  # 获取BPM序列
        padding_mask = self.padding_masks[idx]  # 获取填充掩码
        seq_len = int(np.sum(padding_mask))  # 计算有效序列长度
        if np.random.rand() < 0.5:  # 50%概率进行数据增强
            bpm_orig = bpm[:seq_len]  # 仅处理有效序列
            noise_scale = np.random.uniform(0.05, 0.15)  # 噪声尺度
            noise = np.random.normal(0, noise_scale, len(bpm_orig))  # 添加正态噪声
            bpm_orig = np.clip(bpm_orig + noise, 0.0, np.max(bpm_orig))  # 裁剪确保非负
            scale = np.random.uniform(0.8, 1.2)  # 缩放因子
            bpm_orig = bpm_orig * scale  # 缩放序列
            if len(bpm_orig) > 10:
                perturb_idx = np.random.randint(0, len(bpm_orig) - 10)  # 随机选择扰动起始点
                perturb_size = np.random.uniform(-0.2, 0.2)  # 扰动大小
                bpm_orig[perturb_idx:perturb_idx + 10] += perturb_size  # 添加局部扰动
                bpm_orig = np.clip(bpm_orig, 0.0, np.max(bpm_orig))  # 裁剪
            if len(bpm_orig) > 20:
                freq = np.random.uniform(0.5, 1.5)  # 正弦波频率
                t = np.arange(len(bpm_orig))
                bpm_orig += 0.05 * np.sin(2 * np.pi * freq * t / len(bpm_orig))  # 添加正弦调制
                bpm_orig = np.clip(bpm_orig, 0.0, np.max(bpm_orig))  # 裁剪
            valid_bpm = bpm_orig[padding_mask[:seq_len] > 0]  # 获取有效BPM值
            if len(valid_bpm) > 0:
                bpm_mean = np.mean(valid_bpm)  # 计算均值
                if bpm_mean > 0:
                    bpm_orig = bpm_orig / bpm_mean  # 归一化
            bpm = np.concatenate([bpm_orig, [0.0] * (len(bpm) - len(bpm_orig))])  # 填充到原始长度
        performer = self.metadata.iloc[idx]['Performer']  # 获取表演者
        phrase_index = self.metadata.iloc[idx]['Phrase_Index']  # 获取乐句索引
        try:
            performer = int(float(performer))
            phrase_index = int(float(phrase_index))
        except (ValueError, TypeError):
            raise ValueError(f"无效的 Performer 或 Phrase_Index，索引 {idx}: {performer}, {phrase_index}")
        return (torch.tensor(bpm, dtype=torch.float32),
                torch.tensor(padding_mask, dtype=torch.float32),
                torch.tensor(seq_len, dtype=torch.long),
                self.metadata.iloc[idx]['Song'],
                performer,
                phrase_index)

# 批处理函数，填充到批次最大长度
def collate_fn(batch):
    bpm, padding_mask, seq_len, songs, performers, phrase_indices = zip(*batch)
    max_len = max(max(seq_len), 256)  # 确定最大长度
    def pad_and_stack(tensors, max_len):
        return torch.stack([F.pad(seq[:max_len], (0, max_len - min(len(seq), max_len)), value=0.0) for seq in tensors])
    return (pad_and_stack(bpm, max_len),
            pad_and_stack(padding_mask, max_len),
            torch.tensor(list(seq_len), dtype=torch.long),
            songs,
            performers,
            phrase_indices)

# 创建数据加载器
def create_dataloaders(config):
    splits = ['train', 'val', 'test']  # 数据集划分
    datasets, loaders = {}, {}
    total_phrases = 0
    expected_songs = {'174', '242', '302', '633', '683', 'Islamey'}  # 预期歌曲
    all_songs = set()

    for split in splits:
        csv_file = os.path.join(config.DATA_INPUT_DIR, f'pre_data_{split}_split.csv')  # 加载CSV文件
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"未找到 CSV 文件: {csv_file}")
        df = pd.read_csv(csv_file)
        required_cols = ['Song', 'Performer', 'Phrase_Index', 'bpm', 'padding_mask']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            col_map = {col.lower(): col for col in df.columns}
            for col in missing_cols[:]:
                if col.lower() in col_map:
                    df = df.rename(columns={col_map[col.lower()]: col})
                    missing_cols.remove(col)
            if missing_cols:
                raise ValueError(f"{split} 数据集缺少必要列: {missing_cols}")
        all_songs.update(df['Song'].astype(str))
        if df[required_cols[:3]].isna().any().any():
            raise ValueError(f"{split} 数据集元数据缺失")
        preprocess_csv(csv_file, config.DATA_OUTPUT_DIR, config.MAX_SEQ_LEN)  # 预处理数据
        datasets[split] = PhraseDataset(config.DATA_OUTPUT_DIR, split, csv_file)  # 创建数据集
        loaders[split] = DataLoader(datasets[split], batch_size=config.BATCH_SIZE, shuffle=(split == 'train'),
                                    collate_fn=collate_fn, num_workers=config.NUM_WORKERS, drop_last=False)
        print(f"{split.capitalize()} 数据集大小: {len(datasets[split])}")
        total_phrases += len(datasets[split])

    if all_songs != expected_songs:
        print(f"警告: 数据集包含歌曲 {all_songs}，预期 {expected_songs}")
    if total_phrases != 5756:
        print(f"警告: 总乐句数 {total_phrases}，预期 5756")
    print(f"总乐句数: {total_phrases}")
    return loaders['train'], loaders['val'], loaders['test']

# VQ-VAE模型
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, attn_heads, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        # 编码器
        self.embedding = nn.Linear(1, input_dim)  # 输入嵌入层
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers, batch_first=True, bidirectional=True)  # 双向LSTM
        self.attention = nn.MultiheadAttention(hidden_dim, attn_heads, batch_first=True)  # 多头注意力
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层
        self.encoder_fc = nn.Linear(hidden_dim, embedding_dim)  # 编码器全连接层
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)  # 码本
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)  # 初始化码本权重
        self.stats_embedding = nn.Linear(4, embedding_dim)  # 统计特征嵌入
        self.pooling_projection = nn.Linear(3 * embedding_dim, embedding_dim)  # 池化投影层
        # 解码器
        self.decoder_fc = nn.Linear(embedding_dim, hidden_dim)  # 解码器初始投影
        self.decoder_attention = nn.MultiheadAttention(hidden_dim, attn_heads, batch_first=True)  # 解码器注意力
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers, batch_first=True, bidirectional=True)  # 解码器LSTM
        self.decoder_output = nn.Linear(hidden_dim, 1)  # 输出层
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.register_buffer('codebook_usage', torch.zeros(num_embeddings))  # 码本使用统计

    def quantize(self, z_e, block_size=1000, temperature=0.7):
        if z_e.dim() == 2:
            z_e = z_e.unsqueeze(1)  # 调整维度
        batch_size, seq_len, _ = z_e.shape
        z_e_flat = z_e.reshape(-1, self.embedding_dim)
        encoding_indices = torch.zeros(z_e_flat.shape[0], dtype=torch.long, device=z_e.device)
        if self.num_embeddings == 1:
            encoding_indices = torch.zeros_like(encoding_indices)
        else:
            for i in range(0, z_e_flat.shape[0], block_size):
                z_e_block = z_e_flat[i:i + block_size]
                with torch.no_grad():
                    distances = torch.cdist(z_e_block, self.codebook.weight, p=2)  # 计算距离
                    probs = F.softmax(-distances / temperature, dim=1)  # 转换为概率
                    encoding_indices[i:i + block_size] = torch.multinomial(probs, 1).squeeze(-1).to(z_e.device)
                torch.cuda.empty_cache()  # 清理GPU缓存
        encoding_indices = encoding_indices.to(z_e.device)
        z_q_flat = self.codebook(encoding_indices)  # 获取量化向量
        z_q = z_q_flat.reshape(batch_size, seq_len, self.embedding_dim)
        encoding_indices = encoding_indices.reshape(batch_size, seq_len)
        indices_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        self.codebook_usage += indices_one_hot.sum(dim=(0, 1))  # 更新码本使用统计
        return z_q, encoding_indices

    def encode(self, x, padding_mask, seq_len):
        embedded = self.embedding(x.unsqueeze(-1))  # 输入嵌入
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, seq_len.cpu(), batch_first=True, enforce_sorted=False)  # 打包序列
        packed_lstm_out, _ = self.lstm(packed_embedded)  # LSTM处理
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out, batch_first=True, total_length=x.size(1))  # 解包序列
        lstm_out = self.dropout(lstm_out)  # 应用Dropout
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=~padding_mask.bool())  # 注意力机制
        attn_out = attn_out * padding_mask.unsqueeze(-1)  # 再次应用掩码强化有效时间步
        z_e = self.encoder_fc(attn_out)  # 编码器输出
        x_masked = x * padding_mask
        std = torch.std(x_masked, dim=1, keepdim=True).clamp(min=1e-6)  # 计算标准差
        std_expanded = std.expand(-1, x.size(1))
        peaks = torch.sum((x_masked > (1.0 + 2 * std_expanded)), dim=1, keepdim=True) / torch.sum(padding_mask, dim=1,
                                                                                                  keepdim=True).clamp(
            min=1e-6)  # 计算峰值
        skew = torch.mean(((x_masked - 1.0) ** 3) * padding_mask, dim=1, keepdim=True) / (std ** 3 + 1e-6)  # 计算偏度
        fft = torch.fft.rfft(x_masked, dim=1)
        energy = torch.abs(fft[:, :5]).mean(dim=1, keepdim=True)  # 计算频域能量
        stats = torch.cat([std, peaks, skew, energy], dim=-1)  # 合并统计特征
        stats_embed = self.stats_embedding(stats)  # 嵌入统计特征
        mask_weights = padding_mask / (torch.sum(padding_mask, dim=1, keepdim=True).clamp(min=1e-6))  # 动态掩码权重
        z_e_mean = torch.sum(z_e * mask_weights.unsqueeze(-1), dim=1)  # 加权均值池化
        z_e_max, _ = torch.max(z_e * padding_mask.unsqueeze(-1), dim=1)  # 最大池化
        z_e_phrase = torch.cat([z_e_mean, z_e_max, stats_embed], dim=-1)  # 合并特征
        z_e_phrase = self.pooling_projection(z_e_phrase)  # 投影到嵌入维度
        if torch.rand(1).item() < 0.001:
            print(f"z_e_phrase mean: {z_e_phrase.mean().item()}, std: {z_e_phrase.std().item()}")
        z_q_phrase, encoding_indices_phrase = self.quantize(z_e_phrase)  # 量化
        z_q_expanded = z_q_phrase.expand(-1, x.size(1), -1)  # 扩展量化向量
        return z_e, z_q_expanded, encoding_indices_phrase

    def decode(self, z_q, seq_len, padding_mask):
        h = self.decoder_fc(z_q)  # 解码器初始投影
        h = F.relu(h)  # 添加非线性激活
        h = self.dropout(h)  # 应用Dropout
        attn_out, _ = self.decoder_attention(h, h, h, key_padding_mask=~padding_mask.bool())  # 解码器注意力
        packed_attn_out = torch.nn.utils.rnn.pack_padded_sequence(
            attn_out, seq_len.cpu(), batch_first=True, enforce_sorted=False)  # 打包序列
        packed_lstm_out, _ = self.decoder_lstm(packed_attn_out)  # 解码器LSTM
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out, batch_first=True, total_length=z_q.size(1))  # 解包序列
        lstm_out = self.dropout(lstm_out)  # 应用Dropout
        return self.decoder_output(lstm_out).squeeze(-1)  # 输出重建序列

    def forward(self, src, padding_mask, seq_len):
        z_e, z_q, encoding_indices = self.encode(src, padding_mask, seq_len)  # 编码
        recon = self.decode(z_q, seq_len, padding_mask)  # 解码
        return recon, z_e, z_q, encoding_indices

# 合并码本
def merge_codebook(codebook, threshold=1e-4):
    codebook_np = codebook.detach().cpu().numpy()
    num_clusters = codebook_np.shape[0]
    distances = np.linalg.norm(codebook_np[:, np.newaxis] - codebook_np[np.newaxis, :], axis=2)
    keep = np.ones(num_clusters, dtype=bool)
    for i in range(num_clusters):
        if keep[i]:
            for j in range(i + 1, num_clusters):
                if distances[i, j] < threshold:
                    keep[j] = False
    merged_clusters = np.sum(keep)
    return merged_clusters

# 使用K-Means初始化码本（移除数据标准化）
def initialize_codebook_with_kmeans(model, data_loader, device, num_embeddings, max_samples):
    model.eval()
    z_e_list = []
    total_samples = 0
    with torch.no_grad():
        for bpm, padding_mask, seq_len, _, _, _ in data_loader:
            bpm = bpm.to(device)
            padding_mask = padding_mask.to(device)
            seq_len = seq_len.to(device)
            z_e, _, _ = model.encode(bpm, padding_mask, seq_len)
            z_e_list.append(z_e.cpu().numpy())
            total_samples += z_e.shape[0] * z_e.shape[1]
            if total_samples > max_samples * 2:
                break
    z_e_all = np.concatenate([z_e.reshape(-1, z_e.shape[-1]) for z_e in z_e_list], axis=0)
    if len(z_e_all) > max_samples:
        indices = np.random.choice(len(z_e_all), max_samples, replace=False)
        z_e_all = z_e_all[indices]
    kmeans = MiniBatchKMeans(n_clusters=num_embeddings, n_init=50, random_state=42,
                             batch_size=config.KMEANS_BATCH_SIZE, max_iter=2000).fit(z_e_all)
    centers = kmeans.cluster_centers_
    model.codebook.weight.data = torch.tensor(centers, dtype=torch.float32).to(device)

# 分析码本使用情况
def analyze_codebook_usage(model, data_loader, config, device, experiment_name, epoch):
    model.eval()
    all_indices = []
    all_songs = []
    all_performers = []
    all_phrase_indices = []
    with torch.no_grad():
        for bpm, padding_mask, seq_len, songs, performers, phrase_indices in data_loader:
            bpm = bpm.to(device)
            padding_mask = padding_mask.to(device)
            seq_len = seq_len.to(device)
            _, _, encoding_indices = model.encode(bpm, padding_mask, seq_len)
            all_indices.append(encoding_indices.cpu().numpy())
            all_songs.extend(songs)
            all_performers.extend(performers)
            all_phrase_indices.extend(phrase_indices)
    all_indices = np.concatenate(all_indices, axis=0)
    metadata_df = pd.DataFrame({
        'Song': all_songs,
        'Performer': all_performers,
        'Phrase_Index': all_phrase_indices,
        'Cluster_Index': all_indices.flatten()
    })
    experiment_dir = os.path.join(config.CLUSTER_ANALYSIS_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    def save_files():
        np.save(os.path.join(experiment_dir, f'indices_epoch_{epoch}.npy'), all_indices)
        np.save(os.path.join(experiment_dir, f'codebook_epoch_{epoch}.npy'),
                model.codebook.weight.detach().cpu().numpy())
        usage = model.codebook_usage.cpu().numpy()
        np.save(os.path.join(experiment_dir, f'usage_epoch_{epoch}.npy'), usage)
        metadata_df.to_csv(os.path.join(experiment_dir, f'metadata_epoch_{epoch}.csv'), index=False)
        plt.figure(figsize=(10, 5))
        plt.hist(usage, bins=50)
        plt.title(f'第 {epoch} 轮码本使用情况')
        plt.xlabel('使用次数')
        plt.ylabel('频率')
        plt.savefig(os.path.join(experiment_dir, f'usage_epoch_{epoch}.png'))
        plt.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(save_files)

# 生成乐句聚类分配
def generate_phrase_cluster_assignments(model, train_loader, val_loader, test_loader, config, device, experiment_name, num_embeddings, embedding_dim):
    model.eval()
    all_indices = []
    all_songs = []
    all_performers = []
    all_phrase_indices = []
    loaders = [train_loader, val_loader, test_loader]
    split_names = ['train', 'val', 'test']
    with torch.no_grad():
        for loader, split_name in zip(loaders, split_names):
            print(f"处理 {split_name} 数据集...")
            for bpm, padding_mask, seq_len, songs, performers, phrase_indices in loader:
                bpm = bpm.to(device)
                padding_mask = padding_mask.to(device)
                seq_len = seq_len.to(device)
                _, _, encoding_indices = model.encode(bpm, padding_mask, seq_len)
                all_indices.append(encoding_indices.cpu().numpy())
                all_songs.extend(songs)
                all_performers.extend(performers)
                all_phrase_indices.extend(phrase_indices)
    all_indices = np.concatenate(all_indices, axis=0)
    print(f"处理的总乐句数: {len(all_indices)}")
    if len(all_indices) != 5756:
        print(f"警告: 预期 5756 个乐句，实际得到 {len(all_indices)}")
    df = pd.DataFrame({
        'cluster_index': all_indices.flatten(),
        'song': all_songs,
        'performer': all_performers,
        'phrase_index': all_phrase_indices
    })
    song_order = ['174', '242', '302', '633', '683', 'Islamey']  # 歌曲排序
    df['song_order'] = df['song'].map({song: idx for idx, song in enumerate(song_order)})
    df = df.sort_values(by=['song_order', 'performer', 'phrase_index']).reset_index(drop=True)
    sorted_indices = df['cluster_index'].values
    if not np.all((sorted_indices >= 0) & (sorted_indices < num_embeddings)):
        print(f"警告: NumEmb_{num_embeddings}_EmbDim_{embedding_dim} 的簇索引超出范围 [0, {num_embeddings - 1}]")
    experiment_dir = os.path.join(config.CLUSTER_ANALYSIS_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    output_file = os.path.join(experiment_dir, f'numEmb_{num_embeddings}_embDim_{embedding_dim}_phrase_assignments.mat')
    scipy.io.savemat(output_file, {'cluster_indices': sorted_indices})
    print(f"乐句簇分配保存至 {os.path.basename(output_file)}")

# 训练VQ-VAE模型
def train_vqvae(model, train_loader, val_loader, config, device, experiment_name):
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)  # 优化器
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS - config.WARMUP_EPOCHS, eta_min=5e-7)  # 学习率调度
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 5
    train_patience_counter = 0
    val_patience_counter = 0
    min_delta = 0.001

    if config.KMEANS_INIT:
        initialize_codebook_with_kmeans(model, train_loader, device, model.num_embeddings, config.KMEANS_MAX_SAMPLES)  # 初始化码本

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        model.codebook_usage.zero_()  # 重置码本使用统计
        optimizer.zero_grad()
        # 计算动态量化温度
        temp = config.QUANTIZE_TEMP_INITIAL - (config.QUANTIZE_TEMP_INITIAL - config.QUANTIZE_TEMP_FINAL) * (epoch / config.EPOCHS)
        for batch_idx, (bpm, padding_mask, seq_len, _, _, _) in enumerate(train_loader):
            bpm = bpm.to(device)
            padding_mask = padding_mask.to(device)
            seq_len = seq_len.to(device)
            recon, z_e, z_q, _ = model(bpm, padding_mask, seq_len)  # 前向传播
            # 计算动态损失权重，基于有效时间步比例
            valid_ratio = torch.sum(padding_mask, dim=1) / padding_mask.size(1)
            loss_weights = torch.clamp(1.0 / valid_ratio, min=1.0, max=10.0).unsqueeze(-1).to(device)
            recon_loss = F.mse_loss(recon * padding_mask * loss_weights, bpm * padding_mask * loss_weights)  # 加权重构损失
            vq_loss = F.mse_loss(z_e.detach(), z_q)  # 量化损失
            commitment_loss = F.mse_loss(z_e, z_q.detach())  # 承诺损失
            codebook_diversity_loss = -torch.var(model.codebook.weight).mean()  # 码本多样性损失
            loss = (recon_loss + vq_loss + config.COMMITMENT_LOSS_WEIGHT * commitment_loss +
                    config.CODEBOOK_DIVERSITY_WEIGHT * codebook_diversity_loss)  # 总损失
            loss = loss / config.GRADIENT_ACCUM_STEPS  # 梯度累积
            loss.backward()
            if (batch_idx + 1) % config.GRADIENT_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * config.GRADIENT_ACCUM_STEPS
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bpm, padding_mask, seq_len, _, _, _ in val_loader:
                bpm = bpm.to(device)
                padding_mask = padding_mask.to(device)
                seq_len = seq_len.to(device)
                recon, z_e, z_q, _ = model(bpm, padding_mask, seq_len)
                valid_ratio = torch.sum(padding_mask, dim=1) / padding_mask.size(1)
                loss_weights = torch.clamp(1.0 / valid_ratio, min=1.0, max=10.0).unsqueeze(-1).to(device)
                recon_loss = F.mse_loss(recon * padding_mask * loss_weights, bpm * padding_mask * loss_weights)
                vq_loss = F.mse_loss(z_e.detach(), z_q)
                commitment_loss = F.mse_loss(z_e, z_q.detach())
                loss = recon_loss + vq_loss + config.COMMITMENT_LOSS_WEIGHT * commitment_loss
                val_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        effective_codebook_size = torch.sum(model.codebook_usage > 0).item()  # 有效码本大小
        print(f"第 {epoch + 1} 轮，有效码本大小: {effective_codebook_size}")
        if effective_codebook_size < model.num_embeddings // 4:
            print(f"第 {epoch + 1} 轮重启未使用码本")
            unused_indices = torch.where(model.codebook_usage == 0)[0]
            if len(unused_indices) > 0:
                model.codebook.weight.data[unused_indices] = torch.randn(len(unused_indices), model.embedding_dim).to(
                    device) * 0.1
                model.codebook_usage.zero_()
        if epoch < config.WARMUP_EPOCHS:
            lr = config.LEARNING_RATE * (epoch + 1) / config.WARMUP_EPOCHS  # 学习率预热
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
        if train_loss < best_train_loss - min_delta:
            best_train_loss = train_loss
            train_patience_counter = 0
        else:
            train_patience_counter += 1
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            val_patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f'{experiment_name}_best.pth'))  # 保存最佳模型
            print(f"新的最佳验证损失: {best_val_loss:.4f}，模型保存于第 {epoch + 1} 轮")
        else:
            val_patience_counter += 1
        if train_patience_counter >= patience and val_patience_counter >= patience:
            print(f"第 {epoch + 1} 轮提前停止：训练和验证损失连续 {patience} 轮未下降")
            break
        print(
            f"{experiment_name} 第 {epoch + 1} 轮，训练损失: {train_loss:.4f}，验证损失: {val_loss:.4f}，时间: {time.time() - start_time:.2f}秒")
        if (epoch + 1) % config.ANALYSIS_FREQ == 0 or epoch == config.EPOCHS - 1:
            analyze_codebook_usage(model, val_loader, config, device, experiment_name, epoch + 1)  # 分析码本

    # 训练结束后对码本进行L2归一化
    with torch.no_grad():
        model.codebook.weight.data = F.normalize(model.codebook.weight.data, p=2, dim=1)
        print(f"{experiment_name} 训练结束，码本已归一化")

    effective_codebook_size = torch.sum(model.codebook_usage > 0).item()
    print(f"{experiment_name} 最终有效码本大小: {effective_codebook_size}")

    experiment_dir = os.path.join(config.CLUSTER_ANALYSIS_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    analyze_codebook_usage(model, val_loader, config, device, experiment_name, 'final')  # 最终分析
    torch.save(model.state_dict(), os.path.join(experiment_dir, 'final_model.pth'))  # 保存最终模型

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失')
    plt.title('训练和验证损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.RESULT_DIR, f'{experiment_name}_loss_curve.png'))
    plt.close()
    return model, effective_codebook_size

# 评估VQ-VAE模型
def evaluate_vqvae(model, test_loader, config, device, experiment_name):
    model.eval()
    all_recon, all_true, all_indices = [], [], []
    all_songs, all_performers, all_phrase_indices = [], [], []
    test_mse_loss = 0.0
    vis_samples = 3
    with torch.no_grad():
        for batch_idx, (bpm, padding_mask, seq_len, songs, performers, phrase_indices) in enumerate(test_loader):
            bpm = bpm.to(device)
            padding_mask = padding_mask.to(device)
            seq_len = seq_len.to(device)
            recon, _, _, encoding_indices = model(bpm, padding_mask, seq_len)
            all_recon.append(recon.cpu().numpy())
            all_true.append(bpm.cpu().numpy())
            all_indices.append(encoding_indices.cpu().numpy())
            all_songs.extend(songs)
            all_performers.extend(performers)
            all_phrase_indices.extend(phrase_indices)
            valid_ratio = torch.sum(padding_mask, dim=1) / padding_mask.size(1)
            loss_weights = torch.clamp(1.0 / valid_ratio, min=1.0, max=10.0).unsqueeze(-1).to(device)
            mse = F.mse_loss(recon * padding_mask * loss_weights, bpm * padding_mask * loss_weights, reduction='mean')
            test_mse_loss += mse.item() * bpm.size(0)
            if batch_idx == 0:
                for i in range(min(vis_samples, bpm.size(0))):
                    plt.figure(figsize=(10, 4))
                    plt.plot(bpm[i].cpu().numpy()[:seq_len[i]], label='原始 BPM', alpha=0.7)
                    plt.plot(recon[i].cpu().numpy()[:seq_len[i]], label='重构 BPM', alpha=0.7)
                    plt.plot((bpm[i] * padding_mask[i]).cpu().numpy()[:seq_len[i]], label='掩码 BPM', alpha=0.7)
                    plt.title(f'样本 {i + 1} 重构对比')
                    plt.xlabel('时间步')
                    plt.ylabel('BPM')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(config.RESULT_DIR, f'{experiment_name}_sample_{i + 1}.png'))
                    plt.close()
    all_indices = np.concatenate(all_indices, axis=0)
    experiment_dir = os.path.join(config.CLUSTER_ANALYSIS_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    np.save(os.path.join(experiment_dir, 'test_indices.npy'), all_indices)
    np.save(os.path.join(experiment_dir, 'test_codebook.npy'), model.codebook.weight.detach().cpu().numpy())
    metadata_df = pd.DataFrame({
        'Song': all_songs,
        'Performer': all_performers,
        'Phrase_Index': all_phrase_indices,
        'Cluster_Index': all_indices.flatten()
    })
    metadata_df.to_csv(os.path.join(experiment_dir, 'test_metadata.csv'), index=False)
    max_seq_len = max(arr.shape[1] for arr in all_recon if len(arr.shape) > 1)

    def pad_to_max_length(arrays, max_len):
        padded = []
        for arr in arrays:
            if len(arr.shape) == 2:
                _, seq_len = arr.shape
                pad_width = ((0, 0), (0, max(0, max_len - seq_len)))
                padded.append(np.pad(arr, pad_width, 'constant', constant_values=0))
        return np.concatenate(padded) if padded else np.array([])

    all_recon = pad_to_max_length(all_recon, max_seq_len).reshape(-1)
    all_true = pad_to_max_length(all_true, max_seq_len).reshape(-1)
    recon_error = np.mean((all_recon - all_true) ** 2)
    test_mse_loss /= len(test_loader.dataset)
    print(f"{experiment_name} - 重构误差: {recon_error:.4f}, 测试 MSE 损失: {test_mse_loss:.4f}")
    return recon_error, test_mse_loss

# 运行对比实验
def run_contrast_experiments(config, train_loader, val_loader, test_loader, device):
    num_embeddings_list = [128, 64, 32, 16, 8]  # 码本大小列表
    embedding_dim_list = [512, 256, 128, 64, 32]  # 嵌入维度列表
    effective_codebook_matrix = np.zeros((len(num_embeddings_list), len(embedding_dim_list)))
    merged_codebook_matrix = np.zeros((len(num_embeddings_list), len(embedding_dim_list)))
    experiment_metadata = []

    for i, num_embeddings in enumerate(num_embeddings_list):
        for j, embedding_dim in enumerate(embedding_dim_list):
            config.NUM_EMBEDDINGS = num_embeddings
            config.EMBEDDING_DIM = embedding_dim
            experiment_name = f"VQVAE_numEmb_{num_embeddings}_embDim_{embedding_dim}"
            experiment_dir = os.path.join(config.CLUSTER_ANALYSIS_DIR, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            print(f"\n运行实验: {experiment_name}")

            seed = 42
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = VQVAE(
                input_dim=config.INPUT_DIM,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_LAYERS,
                dropout_rate=config.DROPOUT_RATE,
                attn_heads=config.ATTN_HEADS,
                num_embeddings=config.NUM_EMBEDDINGS,
                embedding_dim=config.EMBEDDING_DIM
            ).to(device)

            trained_model, effective_codebook_size = train_vqvae(model, train_loader, val_loader, config, device,
                                                                experiment_name)
            recon_error, test_mse_loss = evaluate_vqvae(trained_model, test_loader, config, device, experiment_name)
            effective_codebook_matrix[i, j] = effective_codebook_size
            merged_clusters = merge_codebook(trained_model.codebook.weight, config.CODEBOOK_MERGE_THRESHOLD)
            merged_codebook_matrix[i, j] = merged_clusters
            print(f"{experiment_name} 合并后簇数量: {merged_clusters}")

            generate_phrase_cluster_assignments(trained_model, train_loader, val_loader, test_loader, config, device,
                                               experiment_name, num_embeddings, embedding_dim)

            experiment_data = {
                'model_state': os.path.join(experiment_dir, 'final_model.pth'),
                'codebook': os.path.join(experiment_dir, 'test_codebook.npy'),
                'indices': os.path.join(experiment_dir, 'test_indices.npy'),
                'metadata': os.path.join(experiment_dir, 'test_metadata.csv'),
                'recon_error': recon_error,
                'test_mse_loss': test_mse_loss,
                'effective_codebook_size': effective_codebook_size,
                'merged_clusters': merged_clusters
            }
            experiment_metadata.append({
                'experiment_name': experiment_name,
                'num_embeddings': num_embeddings,
                'embedding_dim': embedding_dim,
                'effective_codebook_size': effective_codebook_size,
                'merged_clusters': merged_clusters,
                'cluster_dir': experiment_dir
            })

    df_effective = pd.DataFrame(effective_codebook_matrix,
                                index=[f"NumEmb_{n}" for n in num_embeddings_list],
                                columns=[f"EmbDim_{d}" for d in embedding_dim_list])
    df_effective.to_csv(os.path.join(config.RESULT_DIR, 'effective_codebook_matrix.csv'))
    print("\n5×5 有效码本大小矩阵:")
    print(df_effective)

    df_merged = pd.DataFrame(merged_codebook_matrix,
                             index=[f"NumEmb_{n}" for n in num_embeddings_list],
                             columns=[f"EmbDim_{d}" for d in embedding_dim_list])
    df_merged.to_csv(os.path.join(config.RESULT_DIR, 'merged_codebook_matrix.csv'))
    print("\n5×5 合并后簇数量矩阵:")
    print(df_merged)

    merged_matrix_np = merged_codebook_matrix.flatten()
    sorted_indices = np.argsort(merged_matrix_np)[::-1]
    sorted_matrix = merged_matrix_np[sorted_indices]
    sorted_matrix_2d = sorted_matrix.reshape(len(num_embeddings_list), len(embedding_dim_list))
    scipy.io.savemat(os.path.join(config.RESULT_DIR, 'sorted_merged_codebook_matrix.mat'),
                     {'sorted_merged_codebook_matrix': sorted_matrix_2d})
    print("\n排序后的合并簇数量矩阵已保存为 sorted_merged_codebook_matrix.mat")

    with open(os.path.join(config.RESULT_DIR, 'experiment_metadata.json'), 'w') as f:
        json.dump(experiment_metadata, f, indent=4)
    print("实验元数据已保存为 experiment_metadata.json")

if __name__ == "__main__":
    config = Config()
    print(f"配置加载完成，设备: {config.DEVICE}，CUDA 可用: {torch.cuda.is_available()}")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    device = config.DEVICE
    run_contrast_experiments(config, train_loader, val_loader, test_loader, device)