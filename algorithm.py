"""
稠密感知快放算法 (Density-Aware Speed-Up Algorithm)
核心算法模块

本项目代码由Manus AI完成。
"""

import torch
import numpy as np
from pydub import AudioSegment
import pyrubberband as pyrb
from typing import List, Dict
import os
import time

# ==============================================================================
# 1. 精细化VAD：获取语音概率序列
# ==============================================================================

def get_speech_probabilities(
    audio_path: str,
    vad_model,
    sampling_rate: int = 16000
) -> List[Dict[str, float]]:
    """
    使用Silero-VAD模型，获取每个音频块的语音概率。

    Args:
        audio_path (str): 输入音频文件路径。
        vad_model: 加载的Silero-VAD模型。
        sampling_rate (int): VAD模型期望的采样率。

    Returns:
        List[Dict[str, float]]: 一个列表，每个元素包含块的'start'/'end'（毫秒）和'prob'（语音概率）。
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(sampling_rate).set_channels(1)
    except Exception as e:
        print(f"加载音频时出错: {e}")
        return []

    audio_samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
    
    # VAD模型期望的块大小（以样本为单位）
    # Silero-VAD 在16kHz采样率下期望 512 samples
    chunk_size_samples = 512
    
    probabilities = []
    
    for i in range(0, len(audio_samples), chunk_size_samples):
        chunk = audio_samples[i: i + chunk_size_samples]
        if len(chunk) < chunk_size_samples:
            padding = np.zeros(chunk_size_samples - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, padding])
            
        speech_prob = vad_model(torch.from_numpy(chunk), sampling_rate).item()
        
        start_ms = (i / sampling_rate) * 1000
        end_ms = ((i + len(chunk)) / sampling_rate) * 1000
        
        probabilities.append({'start': start_ms, 'end': end_ms, 'prob': speech_prob})

    print(f"VAD概率分析完成，生成了 {len(probabilities)} 个概率块。")
    return probabilities

# ==============================================================================
# 2. 生成多级处理片段列表
# ==============================================================================

def create_multi_level_segments(
    total_duration_ms: int,
    speech_probs: List[Dict[str, float]],
    silence_thresh: float,
    low_density_thresh: float
) -> List[Dict]:
    """
    根据语音概率，将音频划分为"高密度"、"低密度"和"静音"片段。

    Args:
        total_duration_ms (int): 音频总时长（毫秒）。
        speech_probs (List[Dict[str, float]]): VAD概率块列表。
        silence_thresh (float): 低于此概率的为静音。
        low_density_thresh (float): 低于此概率的为低密度语音。

    Returns:
        List[Dict]: 包含所有片段信息（类型、开始、结束时间）的列表。
    """
    segments = []
    if not speech_probs:
        return []

    current_segment_type = None
    current_segment_start = 0

    for prob_chunk in speech_probs:
        prob = prob_chunk['prob']
        
        if prob < silence_thresh:
            segment_type = 'silence'
        elif prob < low_density_thresh:
            segment_type = 'low_density_speech'
        else:
            segment_type = 'high_density_speech'
            
        if current_segment_type is None:
            current_segment_type = segment_type

        if segment_type != current_segment_type:
            segments.append({
                'type': current_segment_type,
                'start': current_segment_start,
                'end': prob_chunk['start']
            })
            current_segment_start = prob_chunk['start']
            current_segment_type = segment_type
    
    # 添加最后一个片段
    if current_segment_type is not None:
        segments.append({
            'type': current_segment_type,
            'start': current_segment_start,
            'end': total_duration_ms
        })
    
    print(f"音频被划分为 {len(segments)} 个多级片段。")
    return segments

# ==============================================================================
# 3. 核心算法 v2：带语音密度感知的非线性变速
# ==============================================================================

def intelligent_speed_up_v2(
    audio_path: str,
    output_path: str,
    base_rate: float,
    high_density_factor: float,
    low_density_factor: float,
    silence_threshold: float = 0.2,
    low_density_threshold: float = 0.7
):
    """
    对音频文件进行带语音密度感知的智能变速处理。

    Args:
        audio_path (str): 输入音频文件路径。
        output_path (str): 输出文件路径。
        base_rate (float): 基准倍速。
        high_density_factor (float): 高密度语音的速度调节因子。
        low_density_factor (float): 低密度语音的速度调节因子。
        silence_threshold (float): VAD概率低于此值为静音。
        low_density_threshold (float): VAD概率低于此值为低密度语音。
    """
    start_time = time.time()
    print("开始加载模型和音频 (v2)...")

    try:
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        raise
        
    audio = audio.set_frame_rate(16000).set_channels(1)
    total_duration_ms = len(audio)
    print(f"音频加载完成，总时长: {total_duration_ms / 1000:.2f} 秒。")

    print("\n--- 步骤 1: VAD概率分析 ---")
    speech_probabilities = get_speech_probabilities(audio_path, model)
    if not speech_probabilities:
        print("无法分析音频概率，已退出。")
        raise ValueError("VAD分析失败")

    print("\n--- 步骤 2: 计算多级变速速率 ---")
    segments = create_multi_level_segments(total_duration_ms, speech_probabilities, silence_threshold, low_density_threshold)
    
    durations = {'silence': 0, 'low_density_speech': 0, 'high_density_speech': 0}
    for s in segments:
        durations[s['type']] += (s['end'] - s['start'])

    # 计算各类型语音的速度
    speed_high = base_rate * high_density_factor
    speed_low = base_rate * low_density_factor
    
    # 根据总时长约束，反推静音速度
    target_total_duration = total_duration_ms / base_rate
    duration_after_speech_processing = (durations['high_density_speech'] / speed_high) + \
                                       (durations['low_density_speech'] / speed_low)
    
    remaining_duration_for_silence = target_total_duration - duration_after_speech_processing
    
    if remaining_duration_for_silence <= 0 or durations['silence'] == 0:
        speed_silence = 100.0
        print("警告: 人声部分变速后已超过目标总时长，静音将被极度压缩。")
    else:
        speed_silence = durations['silence'] / remaining_duration_for_silence

    speeds = {
        'high_density_speech': speed_high,
        'low_density_speech': speed_low,
        'silence': speed_silence
    }
    
    print(f"基准倍速: {base_rate:.2f}x")
    print(f"高密度语音速度: {speed_high:.2f}x (因子: {high_density_factor})")
    print(f"低密度语音速度: {speed_low:.2f}x (因子: {low_density_factor})")
    print(f"计算得出的静音速度: {speed_silence:.2f}x")

    print("\n--- 步骤 3: 分段变速与合并 ---")
    processed_segments = []
    for i, segment_info in enumerate(segments):
        start_ms, end_ms = segment_info['start'], segment_info['end']
        seg_type = segment_info['type']
        
        if start_ms >= end_ms: 
            continue

        original_chunk = audio[start_ms:end_ms]
        
        dtype = getattr(np, f"int{original_chunk.sample_width * 8}")
        samples = np.array(original_chunk.get_array_of_samples(), dtype=dtype)

        speed = speeds[seg_type]
        
        if abs(speed - 1.0) < 0.01:
            processed_chunk_samples = samples
        else:
            processed_chunk_samples = pyrb.time_stretch(samples, original_chunk.frame_rate, speed)

        processed_chunk_samples = (processed_chunk_samples * (2**15)).astype(dtype)
        
        processed_chunk = AudioSegment(
            processed_chunk_samples.tobytes(),
            frame_rate=original_chunk.frame_rate,
            sample_width=original_chunk.sample_width,
            channels=original_chunk.channels
        )
        processed_segments.append(processed_chunk)
        print(f"处理片段 {i+1}/{len(segments)}: 类型={seg_type}, 速度={speed:.2f}x, "
              f"原始时长={len(original_chunk)/1000:.2f}s, "
              f"处理后时长={len(processed_chunk)/1000:.2f}s")

    final_audio = sum(processed_segments, AudioSegment.empty())
    
    print("\n--- 步骤 4: 导出最终音频 ---")
    final_audio.export(output_path, format=os.path.splitext(output_path)[1][1:])
    
    end_time = time.time()
    original_duration = total_duration_ms / 1000
    final_duration = len(final_audio) / 1000
    
    print("\n🎉 处理完成 (v2)！")
    print(f"文件已保存至: {output_path}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"原始音频时长: {original_duration:.2f} 秒")
    print(f"输出音频时长: {final_duration:.2f} 秒")
    print(f"实现的平均倍速: {original_duration / final_duration:.2f}x (目标: {base_rate:.2f}x)")
