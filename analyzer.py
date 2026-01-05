"""
音频预分析模块
分析音频的语音密度特征，并根据特征推荐最优参数

本项目代码由Manus AI完成。
"""

import torch
import torchaudio
from pydub import AudioSegment
import numpy as np
import os


def analyze_audio_characteristics(audio_path: str, base_rate: float = 1.8, strict_position: bool = False) -> dict:
    """
    分析音频的语音密度特征，并推荐最优参数
    
    Args:
        audio_path: 音频文件路径
        base_rate: 用户设定的基准倍速
        strict_position: 是否为严格相对位置模式推荐参数
    
    Returns:
        dict: 包含分析结果和推荐参数
    """
    print(f"开始分析音频特征: {audio_path}")
    print(f"模式: {'严格相对位置' if strict_position else '普通'}")
    
    try:
        # 加载VAD模型
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        (get_speech_timestamps, _, read_audio, _, _) = utils
        
        # 加载原始音频获取音频信息
        original_audio = AudioSegment.from_file(audio_path)
        
        # 获取原始音频参数
        original_sample_rate = original_audio.frame_rate
        original_channels = original_audio.channels
        original_bit_depth = original_audio.sample_width * 8
        
        # 获取文件格式
        ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
        if ext in ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac']:
            file_format = ext
        else:
            file_format = 'unknown'
        
        # 转换为16kHz单声道用于VAD分析
        audio = original_audio.set_frame_rate(16000).set_channels(1)
        total_duration = len(original_audio) / 1000.0  # 秒
        
        # 转换为numpy数组
        audio_samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
        
        # VAD分析
        chunk_size_samples = 512
        probabilities = []
        
        for i in range(0, len(audio_samples), chunk_size_samples):
            chunk = audio_samples[i:i + chunk_size_samples]
            
            if len(chunk) < chunk_size_samples:
                chunk = np.pad(chunk, (0, chunk_size_samples - len(chunk)), mode='constant')
            
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0)
            
            with torch.no_grad():
                speech_prob = model(chunk_tensor, 16000).item()
            
            probabilities.append(speech_prob)
        
        # 分析语音密度分布
        probabilities = np.array(probabilities)
        
        # 计算各类片段的时长占比
        high_density_mask = probabilities > 0.7
        low_density_mask = (probabilities > 0.3) & (probabilities <= 0.7)
        silence_mask = probabilities <= 0.3
        
        high_density_ratio = np.sum(high_density_mask) / len(probabilities)
        low_density_ratio = np.sum(low_density_mask) / len(probabilities)
        silence_ratio = np.sum(silence_mask) / len(probabilities)
        
        # 计算平均语音概率
        avg_speech_prob = np.mean(probabilities)
        
        # 计算语音密度的方差（反映语音分布的均匀程度）
        speech_variance = np.var(probabilities)
        
        print(f"分析完成:")
        print(f"  - 总时长: {total_duration:.2f}秒")
        print(f"  - 采样率: {original_sample_rate}Hz")
        print(f"  - 声道数: {original_channels}")
        print(f"  - 位深: {original_bit_depth}bit")
        print(f"  - 格式: {file_format}")
        print(f"  - 高密度语音占比: {high_density_ratio*100:.1f}%")
        print(f"  - 低密度语音占比: {low_density_ratio*100:.1f}%")
        print(f"  - 静音占比: {silence_ratio*100:.1f}%")
        print(f"  - 平均语音概率: {avg_speech_prob:.3f}")
        print(f"  - 语音密度方差: {speech_variance:.3f}")
        
        # 根据分析结果推荐参数（普通模式）
        normal_recommendation = recommend_parameters_from_analysis(
            base_rate=base_rate,
            high_density_ratio=high_density_ratio,
            silence_ratio=silence_ratio,
            avg_speech_prob=avg_speech_prob,
            speech_variance=speech_variance,
            strict_position=False
        )
        
        # 根据分析结果推荐参数（严格相对位置模式）
        strict_recommendation = recommend_parameters_from_analysis(
            base_rate=base_rate,
            high_density_ratio=high_density_ratio,
            silence_ratio=silence_ratio,
            avg_speech_prob=avg_speech_prob,
            speech_variance=speech_variance,
            strict_position=True
        )
        
        # 根据请求的模式返回对应的推荐
        primary_recommendation = strict_recommendation if strict_position else normal_recommendation
        
        return {
            'success': True,
            'analysis': {
                'total_duration': round(total_duration, 2),
                'high_density_ratio': round(high_density_ratio, 3),
                'low_density_ratio': round(low_density_ratio, 3),
                'silence_ratio': round(silence_ratio, 3),
                'avg_speech_prob': round(avg_speech_prob, 3),
                'speech_variance': round(speech_variance, 3)
            },
            'audio_info': {
                'sample_rate': original_sample_rate,
                'channels': original_channels,
                'bit_depth': original_bit_depth,
                'file_format': file_format,
                'duration_seconds': round(total_duration, 2)
            },
            'recommendation': primary_recommendation,
            'recommendations': {
                'normal': normal_recommendation,
                'strict_position': strict_recommendation
            }
        }
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 分析失败时，返回基于倍速的默认推荐
        from config import get_recommended_factors
        fallback_normal = get_recommended_factors(base_rate, strict_position=False)
        fallback_strict = get_recommended_factors(base_rate, strict_position=True)
        
        return {
            'success': False,
            'error': str(e),
            'analysis': None,
            'audio_info': None,
            'recommendation': fallback_strict if strict_position else fallback_normal,
            'recommendations': {
                'normal': fallback_normal,
                'strict_position': fallback_strict
            }
        }


def recommend_parameters_from_analysis(
    base_rate: float,
    high_density_ratio: float,
    silence_ratio: float,
    avg_speech_prob: float,
    speech_variance: float,
    strict_position: bool = False
) -> dict:
    """
    根据音频特征分析结果推荐参数
    
    【普通模式策略】
    1. 高密度语音占比高 → 高密度因子应该更小（保留更多细节）
    2. 静音占比高 → 可以更激进地压缩静音
    3. 平均语音概率高 → 整体语音密集，需要更保守的加速
    4. 语音密度方差大 → 语音分布不均，需要更灵活的分级处理
    
    【严格相对位置模式策略】
    1. 所有片段必须保持相对位置，因子接近 1.0
    2. 高密度语音占比高 → 稍微降低高密度因子，但幅度很小
    3. 静音占比高 → 可以稍微提高低密度因子，但不能破坏位置关系
    4. 整体调整幅度受限，确保时间轴线性缩放
    """
    
    # 基础推荐（基于倍速和模式）
    from config import get_recommended_factors
    base_recommendation = get_recommended_factors(base_rate, strict_position=strict_position)
    
    high_density_factor = base_recommendation['high_density_factor']
    low_density_factor = base_recommendation['low_density_factor']
    
    if strict_position:
        # 严格相对位置模式：调整幅度受限
        description_suffix = ""
        
        # 根据高密度语音占比微调（幅度很小）
        if high_density_ratio > 0.6:
            # 语音非常密集，稍微保护语音
            high_density_factor = max(0.92, high_density_factor - 0.02)
            description_suffix = "，语音密集，已微调保护"
        elif high_density_ratio < 0.3:
            # 语音稀疏，可以稍微加速语音
            high_density_factor = min(1.0, high_density_factor + 0.02)
            description_suffix = "，语音稀疏，已优化效率"
        else:
            description_suffix = "，语音分布均衡"
        
        # 根据静音占比微调（幅度很小）
        if silence_ratio > 0.4:
            # 静音很多，稍微加速静音
            low_density_factor = min(1.10, low_density_factor + 0.02)
            description_suffix += "；静音较多"
        elif silence_ratio < 0.15:
            # 静音很少
            low_density_factor = max(1.0, low_density_factor - 0.02)
            description_suffix += "；语音连续"
        
        # 确保因子在合理范围内，不破坏相对位置
        # 严格模式下，高密度因子和低密度因子的乘积应接近 1.0
        # 这样可以确保整体时间轴接近线性缩放
        product = high_density_factor * low_density_factor
        if abs(product - 1.0) > 0.1:
            # 如果偏差太大，进行修正
            correction = 1.0 / product
            high_density_factor *= correction ** 0.5
            low_density_factor *= correction ** 0.5
        
    else:
        # 普通模式：允许更大的调整幅度
        description_suffix = ""
        
        # 根据高密度语音占比调整
        if high_density_ratio > 0.6:
            # 语音非常密集，需要更保守地处理高密度部分
            high_density_factor = max(0.75, high_density_factor - 0.05)
            description_suffix = "，检测到语音密度极高，已优化为保留更多细节"
        elif high_density_ratio > 0.4:
            # 语音较密集
            description_suffix = "，语音密度适中"
        else:
            # 语音稀疏，可以更激进
            high_density_factor = min(0.95, high_density_factor + 0.05)
            description_suffix = "，语音密度较低，已优化为更高效模式"
        
        # 根据静音占比调整
        if silence_ratio > 0.4:
            # 静音很多，可以更激进地压缩
            low_density_factor = min(2.0, low_density_factor + 0.2)
            description_suffix += "；检测到较多静音，已加强静音压缩"
        elif silence_ratio < 0.15:
            # 静音很少，说明语音连续
            low_density_factor = max(1.0, low_density_factor - 0.1)
            description_suffix += "；语音连续性强"
        
        # 根据语音密度方差调整
        if speech_variance > 0.08:
            # 方差大，说明语音分布不均，分级处理更重要
            description_suffix += "；语音分布不均，已启用精细分级"
    
    return {
        'high_density_factor': round(high_density_factor, 2),
        'low_density_factor': round(low_density_factor, 2),
        'description': base_recommendation['description'] + description_suffix,
        'analysis_based': True,
        'mode': 'strict_position' if strict_position else 'normal'
    }
