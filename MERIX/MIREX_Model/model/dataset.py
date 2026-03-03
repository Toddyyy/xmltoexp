import numpy as np
import torch

def create_onset_deviation_boundaries(num_bins=50, value_range=(-5, 5)):
    """
    起音偏移boundaries：按比例分配 bins
    - 核心区间 [-0.5, 0.5]：70% 的 bins（最重要的区间）
    - 中等区间 [-2, -0.5] + [0.5, 2]：25% 的 bins
    - 极端区间 [-5, -2] + [2, 5]：5% 的 bins
    """
    min_val, max_val = value_range
    
    # 按比例分配 bins
    core_bins = int(num_bins * 0.7)      # 70%
    medium_bins = int(num_bins * 0.25)   # 25%
    extreme_bins = num_bins - core_bins - medium_bins  # 剩余的给极端区间
    
    # 确保至少有1个bin给每个区间
    if extreme_bins < 2:
        extreme_bins = 2
        medium_bins = int(num_bins * 0.25)
        core_bins = num_bins - medium_bins - extreme_bins
    
    boundaries = []
    
    # 极端左区间: [min_val, -2]
    extreme_left_bins = extreme_bins // 2
    if extreme_left_bins > 0:
        left_boundaries = np.linspace(min_val, -2.0, extreme_left_bins + 1)
        boundaries.extend(left_boundaries[:-1])
    
    # 中等左区间: [-2, -0.5]
    medium_left_bins = medium_bins // 2
    if medium_left_bins > 0:
        medium_left_boundaries = np.linspace(-2.0, -0.5, medium_left_bins + 1)
        boundaries.extend(medium_left_boundaries[:-1])
    
    # 核心区间: [-0.5, 0.5]
    core_boundaries = np.linspace(-0.5, 0.5, core_bins + 1)
    boundaries.extend(core_boundaries[:-1])
    
    # 中等右区间: [0.5, 2]
    medium_right_bins = medium_bins - medium_left_bins
    if medium_right_bins > 0:
        medium_right_boundaries = np.linspace(0.5, 2.0, medium_right_bins + 1)
        boundaries.extend(medium_right_boundaries[:-1])
    
    # 极端右区间: [2, max_val]
    extreme_right_bins = extreme_bins - extreme_left_bins
    if extreme_right_bins > 0:
        right_boundaries = np.linspace(2.0, max_val, extreme_right_bins + 1)
        boundaries.extend(right_boundaries)
    else:
        boundaries.append(max_val)
    
    boundaries = np.array(boundaries)
    
    print(f"Onset deviation boundaries: {len(boundaries)} boundaries for {num_bins} bins")
    print(f"  Core bins: {core_bins}, Medium bins: {medium_bins}, Extreme bins: {extreme_bins}")
    
    assert len(boundaries) == num_bins + 1, f"需要 {num_bins + 1} 个boundaries，实际有 {len(boundaries)} 个"
    
    return torch.tensor(boundaries, dtype=torch.float32)


def create_duration_deviation_boundaries(num_bins=40, value_range=(-4, 4)):
    """
    持续时间偏移boundaries：按比例分配
    - 核心区间 [-0.3, 0.3]：60% 的 bins
    - 中等区间 [-1.5, -0.3] + [0.3, 1.5]：30% 的 bins
    - 极端区间 [-4, -1.5] + [1.5, 4]：10% 的 bins
    """
    min_val, max_val = value_range
    
    # 按比例分配 bins
    core_bins = int(num_bins * 0.6)      # 60%
    medium_bins = int(num_bins * 0.3)    # 30%
    extreme_bins = num_bins - core_bins - medium_bins  # 剩余的
    
    # 确保至少有1个bin给每个区间
    if extreme_bins < 2:
        extreme_bins = 2
        medium_bins = max(2, int(num_bins * 0.3))
        core_bins = num_bins - medium_bins - extreme_bins
    
    boundaries = []
    
    # 极端左: [min_val, -1.5]
    extreme_left_bins = extreme_bins // 2
    if extreme_left_bins > 0:
        extreme_left_boundaries = np.linspace(min_val, -1.5, extreme_left_bins + 1)
        boundaries.extend(extreme_left_boundaries[:-1])
    
    # 中等左: [-1.5, -0.3]
    medium_left_bins = medium_bins // 2
    if medium_left_bins > 0:
        medium_left_boundaries = np.linspace(-1.5, -0.3, medium_left_bins + 1)
        boundaries.extend(medium_left_boundaries[:-1])
    
    # 核心: [-0.3, 0.3]
    core_boundaries = np.linspace(-0.3, 0.3, core_bins + 1)
    boundaries.extend(core_boundaries[:-1])
    
    # 中等右: [0.3, 1.5]
    medium_right_bins = medium_bins - medium_left_bins
    if medium_right_bins > 0:
        medium_right_boundaries = np.linspace(0.3, 1.5, medium_right_bins + 1)
        boundaries.extend(medium_right_boundaries[:-1])
    
    # 极端右: [1.5, max_val]
    extreme_right_bins = extreme_bins - extreme_left_bins
    if extreme_right_bins > 0:
        extreme_right_boundaries = np.linspace(1.5, max_val, extreme_right_bins + 1)
        boundaries.extend(extreme_right_boundaries)
    else:
        boundaries.append(max_val)
    
    boundaries = np.array(boundaries)
    
    print(f"Duration deviation boundaries: {len(boundaries)} boundaries for {num_bins} bins")
    print(f"  Core bins: {core_bins}, Medium bins: {medium_bins}, Extreme bins: {extreme_bins}")
    
    assert len(boundaries) == num_bins + 1, f"需要 {num_bins + 1} 个boundaries，实际有 {len(boundaries)} 个"
    
    return torch.tensor(boundaries, dtype=torch.float32)


def create_local_tempo_boundaries(num_bins=50, value_range=(5.0, 500.0)):
    """
    本地节拍boundaries：按音乐常用性分配
    - 常用区间 [60, 180]：70% 的 bins
    - 偶尔区间 [40, 60] + [180, 250]：25% 的 bins  
    - 极端区间 [5, 40] + [250, 500]：5% 的 bins
    """
    min_val, max_val = value_range
    
    # 按比例分配 bins
    common_bins = int(num_bins * 0.7)        # 70%
    occasional_bins = int(num_bins * 0.25)   # 25%
    extreme_bins = num_bins - common_bins - occasional_bins  # 剩余的
    
    # 确保至少有1个bin给每个区间
    if extreme_bins < 2:
        extreme_bins = 2
        occasional_bins = max(2, int(num_bins * 0.25))
        common_bins = num_bins - occasional_bins - extreme_bins
    
    boundaries = []
    
    # 极端左: [5, 40]
    extreme_left_bins = extreme_bins // 2
    if extreme_left_bins > 0:
        extreme_left_boundaries = np.linspace(min_val, 40, extreme_left_bins + 1)
        boundaries.extend(extreme_left_boundaries[:-1])
    
    # 偶尔左: [40, 60]
    occasional_left_bins = occasional_bins // 2
    if occasional_left_bins > 0:
        occasional_left_boundaries = np.linspace(40, 60, occasional_left_bins + 1)
        boundaries.extend(occasional_left_boundaries[:-1])
    
    # 常用: [60, 180]
    common_boundaries = np.linspace(60, 180, common_bins + 1)
    boundaries.extend(common_boundaries[:-1])
    
    # 偶尔右: [180, 250]
    occasional_right_bins = occasional_bins - occasional_left_bins
    if occasional_right_bins > 0:
        occasional_right_boundaries = np.linspace(180, 250, occasional_right_bins + 1)
        boundaries.extend(occasional_right_boundaries[:-1])
    
    # 极端右: [250, 500]
    extreme_right_bins = extreme_bins - extreme_left_bins
    if extreme_right_bins > 0:
        extreme_right_boundaries = np.linspace(250, max_val, extreme_right_bins + 1)
        boundaries.extend(extreme_right_boundaries)
    else:
        boundaries.append(max_val)
    
    boundaries = np.array(boundaries)
    
    print(f"Local tempo boundaries: {len(boundaries)} boundaries for {num_bins} bins")
    print(f"  Common bins: {common_bins}, Occasional bins: {occasional_bins}, Extreme bins: {extreme_bins}")
    
    assert len(boundaries) == num_bins + 1, f"需要 {num_bins + 1} 个boundaries，实际有 {len(boundaries)} 个"
    
    return torch.tensor(boundaries, dtype=torch.float32)


def create_duration_boundaries(num_bins=40, value_range=(0, 8.0)):
    """
    音符持续时间boundaries：按音乐常用性分配
    - 短音符 [0, 1]：40% 的 bins（需要高精度）
    - 中等音符 [1, 4]：50% 的 bins
    - 长音符 [4, 8]：10% 的 bins（精度要求较低）
    """
    min_val, max_val = value_range
    
    # 按比例分配 bins
    short_bins = int(num_bins * 0.4)     # 40%
    medium_bins = int(num_bins * 0.5)    # 50%
    long_bins = num_bins - short_bins - medium_bins  # 剩余的
    
    # 确保至少有1个bin给每个区间
    if long_bins < 1:
        long_bins = 1
        medium_bins = max(1, int(num_bins * 0.5))
        short_bins = num_bins - medium_bins - long_bins
    
    boundaries = []
    
    # 短音符: [0, 1]
    if short_bins > 0:
        short_boundaries = np.linspace(min_val, 1.0, short_bins + 1)
        boundaries.extend(short_boundaries[:-1])
    
    # 中等音符: [1, 4]
    if medium_bins > 0:
        medium_boundaries = np.linspace(1.0, 4.0, medium_bins + 1)
        boundaries.extend(medium_boundaries[:-1])
    
    # 长音符: [4, 8]
    if long_bins > 0:
        long_boundaries = np.linspace(4.0, max_val, long_bins + 1)
        boundaries.extend(long_boundaries)
    else:
        boundaries.append(max_val)
    
    boundaries = np.array(boundaries)
    
    print(f"Duration boundaries: {len(boundaries)} boundaries for {num_bins} bins")
    print(f"  Short bins: {short_bins}, Medium bins: {medium_bins}, Long bins: {long_bins}")
    
    assert len(boundaries) == num_bins + 1, f"需要 {num_bins + 1} 个boundaries，实际有 {len(boundaries)} 个"
    
    return torch.tensor(boundaries, dtype=torch.float32)


def create_velocity_boundaries(num_bins=32, value_range=(1.0, 127.0)):
    """
    Velocity boundaries：按音乐动态层次分配
    - ppp-pp [1, 30]：15% 的 bins
    - p-mp [30, 55]：25% 的 bins
    - mf [55, 75]：30% 的 bins
    - f-ff [75, 100]：20% 的 bins
    - fff+ [100, 127]：10% 的 bins
    """
    min_val, max_val = value_range
    
    # 按比例分配 bins
    ppp_bins = max(1, int(num_bins * 0.15))     # 15%
    p_bins = max(1, int(num_bins * 0.25))       # 25%
    mf_bins = max(1, int(num_bins * 0.30))      # 30%
    f_bins = max(1, int(num_bins * 0.20))       # 20%
    fff_bins = num_bins - ppp_bins - p_bins - mf_bins - f_bins  # 剩余的
    
    boundaries = []
    
    # ppp-pp: [1, 30]
    if ppp_bins > 0:
        ppp_boundaries = np.linspace(min_val, 30, ppp_bins + 1)
        boundaries.extend(ppp_boundaries[:-1])
    
    # p-mp: [30, 55]
    if p_bins > 0:
        p_boundaries = np.linspace(30, 55, p_bins + 1)
        boundaries.extend(p_boundaries[:-1])
    
    # mf: [55, 75]
    if mf_bins > 0:
        mf_boundaries = np.linspace(55, 75, mf_bins + 1)
        boundaries.extend(mf_boundaries[:-1])
    
    # f-ff: [75, 100]
    if f_bins > 0:
        f_boundaries = np.linspace(75, 100, f_bins + 1)
        boundaries.extend(f_boundaries[:-1])
    
    # fff+: [100, 127]
    if fff_bins > 0:
        fff_boundaries = np.linspace(100, max_val, fff_bins + 1)
        boundaries.extend(fff_boundaries)
    else:
        boundaries.append(max_val)
    
    boundaries = np.array(boundaries)
    
    print(f"Velocity boundaries: {len(boundaries)} boundaries for {num_bins} bins")
    print(f"  ppp: {ppp_bins}, p: {p_bins}, mf: {mf_bins}, f: {f_bins}, fff: {fff_bins}")
    
    assert len(boundaries) == num_bins + 1, f"需要 {num_bins + 1} 个boundaries，实际有 {len(boundaries)} 个"
    
    return torch.tensor(boundaries, dtype=torch.float32)


def create_sustain_level_boundaries(num_bins=32, value_range=(0, 127)):
    """
    Sustain level boundaries：按踏板使用习惯分配
    - 不踩踏板 [0, 15]：25% 的 bins
    - 半踏板 [15, 63]：25% 的 bins
    - 踩踏板 [63, 127]：50% 的 bins（最常用）
    """
    min_val, max_val = value_range
    
    # 按比例分配 bins
    no_pedal_bins = max(1, int(num_bins * 0.25))     # 25%
    half_pedal_bins = max(1, int(num_bins * 0.25))   # 25%
    full_pedal_bins = num_bins - no_pedal_bins - half_pedal_bins  # 50%
    
    boundaries = []
    
    # 不踩踏板: [0, 15]
    if no_pedal_bins > 0:
        no_pedal_boundaries = np.linspace(min_val, 15, no_pedal_bins + 1)
        boundaries.extend(no_pedal_boundaries[:-1])
    
    # 半踏板: [15, 63]
    if half_pedal_bins > 0:
        half_pedal_boundaries = np.linspace(15, 63, half_pedal_bins + 1)
        boundaries.extend(half_pedal_boundaries[:-1])
    
    # 踩踏板: [63, 127]
    if full_pedal_bins > 0:
        full_pedal_boundaries = np.linspace(63, max_val, full_pedal_bins + 1)
        boundaries.extend(full_pedal_boundaries)
    else:
        boundaries.append(max_val)
    
    boundaries = np.array(boundaries)
    
    print(f"Sustain level boundaries: {len(boundaries)} boundaries for {num_bins} bins")
    print(f"  No pedal: {no_pedal_bins}, Half pedal: {half_pedal_bins}, Full pedal: {full_pedal_bins}")
    
    assert len(boundaries) == num_bins + 1, f"需要 {num_bins + 1} 个boundaries，实际有 {len(boundaries)} 个"
    
    return torch.tensor(boundaries, dtype=torch.float32)


def generate_binned_config_with_boundaries():
    """
    生成数量正确的boundaries配置
    """
    print("Generate Boundaries Config")
    print("=" * 70)
    
    # 生成所有boundaries（确保数量正确）
    print("\n1. Onset Deviation:")
    onset_boundaries = create_onset_deviation_boundaries(50, (-5, 5))
    
    print("\n2. Duration Deviation:")
    duration_boundaries = create_duration_deviation_boundaries(40, (-4, 4))
    
    print("\n3. Local Tempo:")
    tempo_boundaries = create_local_tempo_boundaries(50, (5.0, 500.0))
    
    print("\n4. Velocity:")
    velocity_boundaries = create_velocity_boundaries(32, (1.0, 127.0))
    
    print("\n5. Sustain Level:")
    sustain_boundaries = create_sustain_level_boundaries(32, (0, 127))
    
    # 验证所有boundaries数量
    print(f"\n✅ results:")
    print(f"  Onset: {len(onset_boundaries)} boundaries for 100 bins")
    print(f"  Duration: {len(duration_boundaries)} boundaries for 80 bins")
    print(f"  Tempo: {len(tempo_boundaries)} boundaries for 100 bins")
    print(f"  Velocity: {len(velocity_boundaries)} boundaries for 64 bins")
    print(f"  Sustain: {len(sustain_boundaries)} boundaries for 64 bins")
    
    # 生成最终配置
    binned_config_corrected = {
        # 'onset_deviation_in_seconds': {
        #     'num_bins': 50,
        #     'value_range': (-2, 2),
        #     'custom_boundaries': onset_boundaries.tolist()
        # },
        # 'duration_deviation_in_seconds': {
        #     'num_bins': 40,
        #     'value_range': (-2, 2),
        #     'custom_boundaries': duration_boundaries.tolist()
        # },
        'local_tempo': {
            'num_bins': 50,
            'value_range': (5.0, 500.0),
            'custom_boundaries': tempo_boundaries.tolist()
        },
        'velocity': {
            'num_bins': 32,
            'value_range': (1.0, 127.0),  # 修正为标准MIDI范围
            'custom_boundaries': velocity_boundaries.tolist()
        },
        # 'sustain_level': {
        #     'num_bins': 32,
        #     'value_range': (0, 127),
        #     'custom_boundaries': sustain_boundaries.tolist()
        # },
        'duration':{
            'num_bins': 40,
            'value_range': (0, 8.0),
            'custom_boundaries': None
        
    }
    }
    
    return binned_config_corrected


def test_boundaries_with_binned_head():
    """
    测试boundaries是否能正确work with BinnedTokenHead
    """
    print(f"\n🧪 测试Boundaries与BinnedTokenHead的兼容性")
    print("=" * 50)
    
    config = generate_correct_binned_config()
    
    # 模拟BinnedTokenHead的验证逻辑
    for feature, settings in config.items():
        num_bins = settings['num_bins']
        boundaries = torch.tensor(settings['custom_boundaries'])
        
        # 这是你代码中的断言检查
        expected_boundaries = num_bins + 1
        actual_boundaries = len(boundaries)
        
        print(f"{feature}:")
        print(f"  Expected boundaries: {expected_boundaries}")
        print(f"  Actual boundaries: {actual_boundaries}")
        
        if actual_boundaries == expected_boundaries:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL - 数量不匹配!")
            
        # 验证boundaries是否单调递增
        if torch.all(boundaries[1:] > boundaries[:-1]):
            print(f"  ✅ Boundaries单调递增")
        else:
            print(f"  ❌ Boundaries不是单调递增!")
        
        print()
    
    return config


if __name__ == "__main__":
    # 生成并测试修正后的配置
    corrected_config = test_boundaries_with_binned_head()
    
    print("🎯 修正后的配置已生成，应该可以解决AssertionError问题！")
    print("\n使用方法:")
    print("将返回的 corrected_config 直接用于你的模型初始化")