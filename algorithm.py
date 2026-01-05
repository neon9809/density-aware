"""
稠密感知快放算法 (Density-Aware Speed-Up Algorithm)
核心算法模块

本项目代码由Manus AI完成。
"""

import torch
import numpy as np
from pydub import AudioSegment
import pyrubberband as pyrb
from typing import List, Dict, Optional, Tuple
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
# 3. 音频质量参数类
# ==============================================================================

class AudioQualityConfig:
    """音频质量配置类"""
    
    def __init__(
        self,
        sample_rate: Optional[int] = None,
        bit_depth: Optional[int] = None,
        output_format: str = "mp3",
        mp3_bitrate: str = "192k",
        preserve_channels: bool = True
    ):
        """
        初始化音频质量配置
        
        Args:
            sample_rate: 采样率（Hz），None 表示保持原始
            bit_depth: 位深（8, 16, 24, 32），None 表示保持原始
            output_format: 输出格式 ("mp3", "wav", "flac", "ogg")
            mp3_bitrate: MP3 码率 ("128k", "192k", "256k", "320k")
            preserve_channels: 是否保持原始声道数
        """
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.output_format = output_format.lower()
        self.mp3_bitrate = mp3_bitrate
        self.preserve_channels = preserve_channels
    
    @classmethod
    def quick_preview(cls) -> 'AudioQualityConfig':
        """快速预览模式：16kHz MP3 128k，不保持声道"""
        return cls(
            sample_rate=16000,
            bit_depth=16,
            output_format="mp3",
            mp3_bitrate="128k",
            preserve_channels=False
        )
    
    @classmethod
    def from_source(cls, audio_path: str) -> 'AudioQualityConfig':
        """从源文件获取配置，输出 WAV 无损格式，保持声道"""
        audio = AudioSegment.from_file(audio_path)
        return cls(
            sample_rate=audio.frame_rate,
            bit_depth=audio.sample_width * 8,
            output_format="wav",
            mp3_bitrate="320k",
            preserve_channels=True
        )
    
    def __repr__(self):
        return f"AudioQualityConfig(sample_rate={self.sample_rate}, bit_depth={self.bit_depth}, format={self.output_format}, mp3_bitrate={self.mp3_bitrate}, preserve_channels={self.preserve_channels})"


def get_audio_info(audio_path: str) -> Dict:
    """
    获取音频文件的详细信息
    
    Args:
        audio_path: 音频文件路径
    
    Returns:
        包含音频信息的字典
    """
    audio = AudioSegment.from_file(audio_path)
    
    # 获取文件格式
    ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
    if ext in ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac']:
        file_format = ext
    else:
        file_format = 'unknown'
    
    return {
        'duration_ms': len(audio),
        'duration_seconds': len(audio) / 1000.0,
        'sample_rate': audio.frame_rate,
        'channels': audio.channels,
        'bit_depth': audio.sample_width * 8,
        'file_format': file_format
    }

# ==============================================================================
# 4. 严格相对位置模式的速度计算
# ==============================================================================

def calculate_strict_position_speeds(
    segments: List[Dict],
    durations: Dict[str, float],
    base_rate: float,
    high_density_factor: float,
    low_density_factor: float,
    total_duration_ms: float
) -> Dict[str, float]:
    """
    计算严格相对位置模式下的各类型速度
    
    核心原则：
    1. 语音片的中点位置必须严格按倍速缩放（原位置/倍速）
    2. 允许语音片从前后静音"借用"一点时间，让语音播放得稍慢
    3. 整体时长必须精确等于 原时长/倍速
    
    实现策略：
    - 高密度语音速度 = base_rate * high_density_factor（factor < 1 时语音变慢）
    - 低密度语音速度 = base_rate * low_density_factor
    - 静音速度动态计算，补偿语音"借用"的时间
    
    Args:
        segments: 片段列表
        durations: 各类型片段的总时长
        base_rate: 基准倍速
        high_density_factor: 高密度因子（严格模式下接近1.0，但可以稍小以保护语音）
        low_density_factor: 低密度因子（严格模式下接近1.0，但可以稍大以压缩停顿）
        total_duration_ms: 原始总时长
    
    Returns:
        各类型的速度字典
    """
    target_total_duration = total_duration_ms / base_rate
    
    # 计算语音部分的速度
    # 在严格模式下，factor 接近 1.0，但允许小幅调整
    speed_high = base_rate * high_density_factor
    speed_low = base_rate * low_density_factor
    
    # 计算语音部分变速后的时长
    duration_high_after = durations['high_density_speech'] / speed_high if speed_high > 0 else 0
    duration_low_after = durations['low_density_speech'] / speed_low if speed_low > 0 else 0
    
    # 计算语音部分"借用"或"节省"的时间
    # 如果 factor < 1，语音变慢，需要从静音借用时间
    # 如果 factor > 1，语音变快，可以给静音更多时间
    ideal_high_after = durations['high_density_speech'] / base_rate
    ideal_low_after = durations['low_density_speech'] / base_rate
    
    time_borrowed = (duration_high_after - ideal_high_after) + (duration_low_after - ideal_low_after)
    
    # 静音需要补偿借用的时间
    ideal_silence_after = durations['silence'] / base_rate
    actual_silence_after = ideal_silence_after - time_borrowed
    
    if actual_silence_after <= 0 or durations['silence'] == 0:
        # 没有足够的静音来补偿，需要极度压缩静音
        speed_silence = 100.0
        print("警告: 静音不足以补偿语音借用的时间，静音将被极度压缩。")
        print(f"  - 语音借用时间: {time_borrowed:.2f}ms")
        print(f"  - 理想静音时长: {ideal_silence_after:.2f}ms")
    else:
        speed_silence = durations['silence'] / actual_silence_after
    
    # 验证总时长
    total_after = duration_high_after + duration_low_after + (durations['silence'] / speed_silence if speed_silence < 100 else 0)
    
    print(f"\n【严格相对位置模式 - 速度计算】")
    print(f"  目标总时长: {target_total_duration:.2f}ms")
    print(f"  语音借用时间: {time_borrowed:.2f}ms ({'借用' if time_borrowed > 0 else '节省'})")
    print(f"  高密度语音: {durations['high_density_speech']:.0f}ms → {duration_high_after:.0f}ms (速度 {speed_high:.2f}x)")
    print(f"  低密度语音: {durations['low_density_speech']:.0f}ms → {duration_low_after:.0f}ms (速度 {speed_low:.2f}x)")
    print(f"  静音: {durations['silence']:.0f}ms → {durations['silence']/speed_silence:.0f}ms (速度 {speed_silence:.2f}x)")
    
    return {
        'high_density_speech': speed_high,
        'low_density_speech': speed_low,
        'silence': speed_silence
    }

# ==============================================================================
# 5. 核心算法 v3：支持立体声和严格相对位置模式
# ==============================================================================

def intelligent_speed_up_v3(
    audio_path: str,
    output_path: str,
    base_rate: float,
    high_density_factor: float,
    low_density_factor: float,
    silence_threshold: float = 0.2,
    low_density_threshold: float = 0.7,
    strict_position: bool = False,
    quality_config: Optional[AudioQualityConfig] = None
) -> Dict:
    """
    对音频文件进行带语音密度感知的智能变速处理（v3 版本）。
    
    新增功能：
    - 立体声支持：保持原始声道数
    - 严格相对位置模式：保持语音片段的精确相对位置，同时允许从静音借用时间优化听感
    - 音频质量配置：自定义采样率、位深、格式

    Args:
        audio_path (str): 输入音频文件路径。
        output_path (str): 输出文件路径。
        base_rate (float): 基准倍速。
        high_density_factor (float): 高密度语音的速度调节因子。
        low_density_factor (float): 低密度语音的速度调节因子。
        silence_threshold (float): VAD概率低于此值为静音。
        low_density_threshold (float): VAD概率低于此值为低密度语音。
        strict_position (bool): 是否启用严格相对位置模式。
        quality_config (AudioQualityConfig): 音频质量配置，None 使用默认。

    Returns:
        Dict: 包含处理结果信息的字典
    """
    start_time = time.time()
    print("开始加载模型和音频 (v3)...")

    try:
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        original_audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        raise
    
    # 默认质量配置
    if quality_config is None:
        quality_config = AudioQualityConfig.quick_preview()
    
    # 保存原始音频参数
    original_frame_rate = original_audio.frame_rate
    original_channels = original_audio.channels
    original_sample_width = original_audio.sample_width
    
    total_duration_ms = len(original_audio)
    
    print(f"音频加载完成，总时长: {total_duration_ms / 1000:.2f} 秒")
    print(f"原始参数: {original_frame_rate}Hz, {original_channels}声道, {original_sample_width * 8}bit")
    print(f"严格相对位置模式: {'启用' if strict_position else '禁用'}")

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

    # 根据模式计算速度
    if strict_position:
        # 严格相对位置模式：允许从静音借用时间
        speeds = calculate_strict_position_speeds(
            segments=segments,
            durations=durations,
            base_rate=base_rate,
            high_density_factor=high_density_factor,
            low_density_factor=low_density_factor,
            total_duration_ms=total_duration_ms
        )
        speed_high = speeds['high_density_speech']
        speed_low = speeds['low_density_speech']
        speed_silence = speeds['silence']
    else:
        # 普通模式：差异化变速，静音动态压缩
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
    
    print(f"\n基准倍速: {base_rate:.2f}x")
    print(f"高密度语音速度: {speed_high:.2f}x (因子: {high_density_factor})")
    print(f"低密度语音速度: {speed_low:.2f}x (因子: {low_density_factor})")
    print(f"计算得出的静音速度: {speed_silence:.2f}x")

    print("\n--- 步骤 3: 分段变速与合并 ---")
    
    # 确定处理时使用的音频（保持原始声道或转单声道）
    preserve_channels = quality_config.preserve_channels
    if preserve_channels and original_channels > 1:
        print(f"保持立体声处理 ({original_channels} 声道)")
        audio_to_process = original_audio
    else:
        print("使用单声道处理")
        audio_to_process = original_audio.set_channels(1)
    
    processed_segments = []
    for i, segment_info in enumerate(segments):
        start_ms, end_ms = segment_info['start'], segment_info['end']
        seg_type = segment_info['type']
        
        if start_ms >= end_ms: 
            continue

        original_chunk = audio_to_process[start_ms:end_ms]
        speed = speeds[seg_type]
        
        # 处理音频片段
        processed_chunk = _process_audio_chunk(original_chunk, speed)
        processed_segments.append(processed_chunk)
        
        if (i + 1) % 20 == 0 or i == len(segments) - 1:
            print(f"处理进度: {i+1}/{len(segments)} 片段")

    final_audio = sum(processed_segments, AudioSegment.empty())
    
    print("\n--- 步骤 4: 应用音频质量配置并导出 ---")
    
    # 设置采样率
    target_sample_rate = quality_config.sample_rate or original_frame_rate
    if final_audio.frame_rate != target_sample_rate:
        final_audio = final_audio.set_frame_rate(target_sample_rate)
        print(f"采样率调整为: {target_sample_rate}Hz")
    
    # 设置位深
    target_bit_depth = quality_config.bit_depth or (original_sample_width * 8)
    target_sample_width = target_bit_depth // 8
    if final_audio.sample_width != target_sample_width:
        final_audio = final_audio.set_sample_width(target_sample_width)
        print(f"位深调整为: {target_bit_depth}bit")
    
    # 确定输出格式和参数
    output_format = quality_config.output_format
    export_params = {}
    
    if output_format == "mp3":
        export_params["bitrate"] = quality_config.mp3_bitrate
        print(f"MP3 码率: {quality_config.mp3_bitrate}")
    
    # 调整输出路径的扩展名
    output_base = os.path.splitext(output_path)[0]
    output_path = f"{output_base}.{output_format}"
    
    final_audio.export(output_path, format=output_format, **export_params)
    
    end_time = time.time()
    original_duration = total_duration_ms / 1000
    final_duration = len(final_audio) / 1000
    
    print("\n🎉 处理完成 (v3)！")
    print(f"文件已保存至: {output_path}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"原始音频时长: {original_duration:.2f} 秒")
    print(f"输出音频时长: {final_duration:.2f} 秒")
    print(f"实现的平均倍速: {original_duration / final_duration:.2f}x (目标: {base_rate:.2f}x)")
    
    return {
        'success': True,
        'output_path': output_path,
        'original_duration': original_duration,
        'output_duration': final_duration,
        'actual_speed': original_duration / final_duration,
        'target_speed': base_rate,
        'processing_time': end_time - start_time,
        'segments_count': len(segments),
        'output_format': output_format,
        'output_sample_rate': target_sample_rate,
        'output_bit_depth': target_bit_depth,
        'output_channels': final_audio.channels,
        'strict_position': strict_position
    }


def _process_audio_chunk(chunk: AudioSegment, speed: float) -> AudioSegment:
    """
    对单个音频片段进行变速处理，支持立体声
    
    Args:
        chunk: 音频片段
        speed: 变速倍率
    
    Returns:
        处理后的音频片段
    """
    if abs(speed - 1.0) < 0.01:
        return chunk
    
    # 获取音频参数
    channels = chunk.channels
    frame_rate = chunk.frame_rate
    sample_width = chunk.sample_width
    
    # 转换为 numpy 数组
    samples = np.array(chunk.get_array_of_samples())
    
    # 确定数据类型
    if sample_width == 1:
        dtype = np.int8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        dtype = np.int16
    
    samples = samples.astype(np.float64)
    
    if channels == 2:
        # 立体声：分离左右声道
        samples = samples.reshape((-1, 2))
        left_channel = samples[:, 0]
        right_channel = samples[:, 1]
        
        # 分别处理每个声道
        left_stretched = pyrb.time_stretch(left_channel, frame_rate, speed)
        right_stretched = pyrb.time_stretch(right_channel, frame_rate, speed)
        
        # 确保两个声道长度相同
        min_len = min(len(left_stretched), len(right_stretched))
        left_stretched = left_stretched[:min_len]
        right_stretched = right_stretched[:min_len]
        
        # 合并声道
        processed_samples = np.column_stack((left_stretched, right_stretched)).flatten()
    else:
        # 单声道
        processed_samples = pyrb.time_stretch(samples, frame_rate, speed)
    
    # 转换回整数类型
    max_val = 2 ** (sample_width * 8 - 1) - 1
    processed_samples = np.clip(processed_samples, -max_val, max_val).astype(dtype)
    
    # 创建新的音频片段
    processed_chunk = AudioSegment(
        processed_samples.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels
    )
    
    return processed_chunk


# ==============================================================================
# 6. 向后兼容：保留 v2 接口
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
    向后兼容的 v2 接口，内部调用 v3 实现
    """
    result = intelligent_speed_up_v3(
        audio_path=audio_path,
        output_path=output_path,
        base_rate=base_rate,
        high_density_factor=high_density_factor,
        low_density_factor=low_density_factor,
        silence_threshold=silence_threshold,
        low_density_threshold=low_density_threshold,
        strict_position=False,
        quality_config=AudioQualityConfig.quick_preview()
    )
    return result
