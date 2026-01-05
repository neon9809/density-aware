# 稠密感知快放算法：原理深度解析

**版本: 2.0.0**

本文档旨在深入剖析“稠密感知快放算法” (Density-Aware Speed-Up Algorithm) 的核心原理、数学模型和实现细节，帮助开发者和研究者更好地理解其工作机制。

---

## 目录

1.  [核心思想：为什么传统倍速不够好？](#核心思想)
2.  [处理流程管道图](#处理流程管道图)
    *   [普通模式 (Normal Mode)](#普通模式)
    *   [严格相对位置模式 (Strict Position Mode)](#严格相对位置模式)
3.  [第一层：语音活动检测 (VAD)](#第一层语音活动检测)
4.  [第二层：密度分级与片段化](#第二层密度分级与片段化)
5.  [第三层：自适应变速与数学模型](#第三层自适应变速与数学模型)
    *   [普通模式的数学原理](#普通模式的数学原理)
    *   [严格相对位置模式的数学原理](#严格相对位置模式的数学原理)
6.  [第四层：音频处理与合成](#第四层音频处理与合成)
7.  [音频感知推荐策略](#音频感知推荐策略)
8.  [关键代码实现解析](#关键代码实现解析)

---

<a name="核心思想"></a>
## 1. 核心思想：为什么传统倍速不够好？

人类说话的节奏包含两个维度：

1.  **语音内容的速度** (音节密度)
2.  **停顿和间隙** (呼吸、思考、强调)

传统的倍速播放对这两个维度一视同仁，导致：

*   重要内容被压缩得过快，理解困难
*   停顿被过度压缩，失去自然节奏
*   整体听感“机械”、“赶”

**稠密感知算法的创新** 在于它能够区分“哪里该快”和“哪里该慢”，通过差异化处理，在保证总时长符合目标倍速的前提下，最大化保留语音的清晰度和自然感。

---

<a name="处理流程管道图"></a>
## 2. 处理流程管道图

算法的核心处理流程根据是否启用“严格相对位置模式”而有所不同。

<a name="普通模式"></a>
### 普通模式 (Normal Mode)

此模式的目标是在满足总时长约束的前提下，最大化语音部分的清晰度。它会大幅压缩静音部分，为语音"腾出"更多时间。

![Normal Mode Pipeline](docs/pipeline_normal.png)

<a name="严格相对位置模式"></a>
### 严格相对位置模式 (Strict Position Mode)

此模式专为视频音画同步场景设计。其核心目标是保证音频中任何一个事件点（如语音片段的中点）在变速后的相对位置严格遵循 `新位置 = 原位置 / 倍速` 的线性关系。

![Strict Position Mode Pipeline](docs/pipeline_strict.png)

---

<a name="第一层语音活动检测"></a>
## 3. 第一层：语音活动检测 (VAD)

这是整个算法的基础。我们使用 [Silero-VAD](https://github.com/snakers4/silero-vad) 模型，这是一个轻量级、高精度的语音活动检测模型。

1.  **输入**: 原始音频文件
2.  **处理**: 模型以 32ms 为一个窗口，滑动分析整个音频，并为每个窗口输出一个语音概率（0 到 1 之间的浮点数）。
3.  **输出**: 一个时间序列数组，每个元素包含 `start` (ms), `end` (ms), 和 `speech_prob` (语音概率)。

> **代码参考**: `analyzer.py` 中的 `get_speech_probabilities` 函数。

---

<a name="第二层密度分级与片段化"></a>
## 4. 第二层：密度分级与片段化

获得VAD概率序列后，我们根据预设的阈值将其合并和分类，形成连续的、具有单一密度的音频“片段” (Segment)。

*   **高密度语音 (High-Density Speech)**: `prob > 0.7`，通常是连续、快速的语音内容。
*   **低密度语音 (Low-Density Speech)**: `0.3 < prob ≤ 0.7`，通常是停顿、语气词、过渡音。
*   **静音 (Silence)**: `prob ≤ 0.3`，通常是间隙、呼吸、长停顿。

这个过程会将一个由成千上万个小窗口组成的VAD概率序列，简化为几十到几百个更长的、类型统一的片段列表。

> **代码参考**: `algorithm.py` 中的 `create_multi_level_segments` 函数。

---

<a name="第三层自适应变速与数学模型"></a>
## 5. 第三层：自适应变速与数学模型

这是算法的核心，根据不同的模式计算每个片段应该使用的变速倍率。

### 定义基础变量

| 变量 | 含义 |
| :--- | :--- |
| $T_{orig}$ | 原始音频总时长 (ms) |
| $R$ | 用户设定的基准倍速 (如 1.8) |
| $T_{target}$ | 目标输出总时长，计算公式为 $T_{target} = T_{orig} / R$ |
| $T_h, T_l, T_s$ | 高密度语音、低密度语音、静音各自的总时长 |
| $\alpha$ | 高密度语音速度调节因子 (0.5 ~ 1.0) |
| $\beta$ | 低密度语音速度调节因子 (1.0 ~ 2.0) |
| $v_h, v_l, v_s$ | 高密度语音、低密度语音、静音各自的实际变速倍率 |

<a name="普通模式的数学原理"></a>
### 普通模式的数学原理

**核心思想**: 优先确定语音部分的播放速度，然后通过大幅压缩静音来满足总时长约束。

1.  **计算语音部分的速度**:
    *   高密度语音速度: $v_h = R \cdot \alpha$
    *   低密度语音速度: $v_l = R \cdot \beta$

2.  **计算语音部分处理后的时长**:
    *   $T'_{h} = T_h / v_h$
    *   $T'_{l} = T_l / v_l$

3.  **计算静音部分需要被压缩到的时长**:
    *   $T'_{s} = T_{target} - T'_{h} - T'_{l}$

4.  **反推出静音部分的速度**:
    *   $v_s = T_s / T'_{s}$

通过这个模型，我们保证了最终的总时长严格等于 $T_{target}$，即 $\frac{T_h}{v_h} + \frac{T_l}{v_l} + \frac{T_s}{v_s} = T_{target}$。

> **代码参考**: `algorithm.py` 中 `intelligent_speed_up_v3` 函数的 `else` 分支 (非 `strict_position` 模式)。

<a name="严格相对位置模式的数学原理"></a>
### 严格相对位置模式的数学原理

**核心思想**: 所有类型的片段（语音、静音）都必须以基准倍速 $R$ 为中心进行变速，但允许语音片段从相邻的静音片段中“借用”一小部分时间来放慢语速，以提升清晰度，前提是不能改变语音片段的中点位置。

1.  **初始速度设定**:
    *   所有片段的初始速度都设定为基准倍速 $R$。即 $v_h = v_l = v_s = R$。

2.  **时间借用机制**:
    *   对于一个高密度语音片段，其理想速度是 $R \cdot \alpha$ (比 $R$ 慢)。它需要额外的时间为 $T_h \cdot (1/v_h - 1/R)$。
    *   这个时间只能从其前后的静音片段中“借”。算法会检查相邻片段是否为静音以及是否有足够的时间可借。
    *   借用时间的总量是有限的，以确保不会过度压缩静音导致听感突兀。

3.  **最终速度计算**:
    *   该模式下，静音的速度 $v_s$ 会非常接近 $R$。
    *   语音速度 $v_h$ 和 $v_l$ 会在 $R$ 的附近小范围浮动，但整体平均速度仍然是 $R$。

这个模型确保了音频的整体时间轴是线性缩放的，从而保证了音画同步。

> **代码参考**: `algorithm.py` 中的 `calculate_strict_position_speeds` 函数。

---

<a name="第四层音频处理与合成"></a>
## 6. 第四层：音频处理与合成

计算出每个片段的速度后，就进入实际的音频处理阶段。

1.  **分段处理**: 遍历每一个片段 (Segment)，从原始音频中切割出对应的块 (Chunk)。
2.  **变速处理**: 调用 `pyrubberband` 库对每个 Chunk 进行变速处理。这里有一个关键细节：
    *   `pyrubberband` 需要输入归一化的 `float` 类型数据 (-1.0 ~ 1.0)。
    *   而 `pydub` 提供的是 `int` 类型数据 (如 int16: -32768 ~ 32767)。
    *   因此，在处理前需要进行 `int -> float` 的归一化，处理后需要进行 `float -> int` 的反归一化。**这是 v2.0 修复静音问题的关键**。
3.  **无缝拼接**: 将处理完的 Chunk 按顺序拼接成一个完整的音频。
4.  **质量配置与导出**: 最后，根据用户的质量要求（采样率、位深、格式、码率等）导出最终文件。

> **代码参考**: `algorithm.py` 中的 `_process_audio_chunk` 函数。

---

<a name="音频感知推荐策略"></a>
## 7. 音频感知推荐策略

为了让非专业用户也能获得良好效果，算法提供了基于音频特征的智能参数推荐功能。

**分析阶段**: `analyzer.py` 中的 `analyze_audio_characteristics` 函数会计算以下特征：

*   高/低密度语音占比
*   静音占比
*   平均语音概率
*   语音密度方差

**推荐阶段**: `config.py` 中的 `get_recommended_factors` 函数会根据这些特征，匹配一个最合适的预设模式。

| 音频特征 | 参数调整策略 | 理由 |
| :--- | :--- | :--- |
| 高密度语音占比 > 60% | 降低高密度因子 (更接近0.5) | 语音密集，需保留更多细节 |
| 高密度语音占比 < 40% | 提高高密度因子 (更接近1.0) | 语音稀疏，可更激进加速 |
| 静音占比 > 40% | 提高低密度因子 (更接近2.0) | 静音多，可加强压缩 |
| 静音占比 < 15% | 降低低密度因子 (更接近1.0) | 语音连续，需保持流畅 |

---

<a name="关键代码实现解析"></a>
## 8. 关键代码实现解析

### `algorithm.py: _process_audio_chunk()` - 变速核心

```python
def _process_audio_chunk(chunk: AudioSegment, speed: float) -> AudioSegment:
    # ... (参数获取)

    # 确定数据类型和最大值 (例如 int16 的最大值是 32767)
    if sample_width == 2:
        dtype = np.int16
        max_val = 32767.0
    # ... (其他位深)
    
    # 1. 归一化到 -1.0 ~ 1.0 范围
    samples_normalized = samples.astype(np.float64) / max_val
    
    # 2. 调用 pyrubberband 进行变速
    if channels == 2:
        # ... (立体声分离处理)
        left_stretched = pyrb.time_stretch(left_channel, frame_rate, speed)
        right_stretched = pyrb.time_stretch(right_channel, frame_rate, speed)
        # ... (合并)
    else:
        processed_normalized = pyrb.time_stretch(samples_normalized, frame_rate, speed)
    
    # 3. 从归一化范围转换回整数范围
    processed_samples = processed_normalized * max_val
    processed_samples = np.clip(processed_samples, -max_val, max_val).astype(dtype)
    
    # 4. 创建新的音频片段
    return AudioSegment(processed_samples.tobytes(), ...)
```

### `algorithm.py: calculate_strict_position_speeds()` - 严格模式速度计算

```python
def calculate_strict_position_speeds(...):
    # ...
    # 遍历所有片段，计算每个语音片段理论上可以从前后静音片段借用的时间
    for i, segment in enumerate(segments):
        if segment['type'] == 'high_density_speech':
            # 计算需要借用的时间
            time_needed = segment_duration * (1 / (base_rate * high_density_factor) - 1 / base_rate)
            # ... 尝试从前后借用 ...
    # ...
    # 根据借用结果，重新计算每个片段的最终速度
    return speeds
```
