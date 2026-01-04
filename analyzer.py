"""
音频预分析模块
分析音频的语音密度特征，并根据特征推荐最优参数

本项目代码由Manus AI完成。
"""

import torch
import torchaudio
from pydub import AudioSegment
import numpy as np


def analyze_audio_characteristics(audio_path: str, base_rate: float = 1.8) -> dict:
    """
    分析音频的语音密度特征，并推荐最优参数
    
    Args:
        audio_path: 音频文件路径
        base_rate: 用户设定的基准倍速
    
    Returns:
        dict: 包含分析结果和推荐参数
    """
    print(f"开始分析音频特征: {audio_path}")
    
    try:
        # 加载VAD模型
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        (get_speech_timestamps, _, read_audio, _, _) = utils
        
        # 加载音频
        audio = AudioSegment.from_file(audio_path)
        
        # 转换为16kHz单声道
        audio = audio.set_frame_rate(16000).set_channels(1)
        total_duration = len(audio) / 1000.0  # 秒
        
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
        print(f"  - 高密度语音占比: {high_density_ratio*100:.1f}%")
        print(f"  - 低密度语音占比: {low_density_ratio*100:.1f}%")
        print(f"  - 静音占比: {silence_ratio*100:.1f}%")
        print(f"  - 平均语音概率: {avg_speech_prob:.3f}")
        print(f"  - 语音密度方差: {speech_variance:.3f}")
        
        # 根据分析结果推荐参数
        recommended = recommend_parameters_from_analysis(
            base_rate=base_rate,
            high_density_ratio=high_density_ratio,
            silence_ratio=silence_ratio,
            avg_speech_prob=avg_speech_prob,
            speech_variance=speech_variance
        )
        
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
            'recommendation': recommended
        }
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 分析失败时，返回基于倍速的默认推荐
        from config import get_recommended_factors
        fallback = get_recommended_factors(base_rate)
        
        return {
            'success': False,
            'error': str(e),
            'analysis': None,
            'recommendation': fallback
        }


def recommend_parameters_from_analysis(
    base_rate: float,
    high_density_ratio: float,
    silence_ratio: float,
    avg_speech_prob: float,
    speech_variance: float
) -> dict:
    """
    根据音频特征分析结果推荐参数
    
    策略：
    1. 高密度语音占比高 → 高密度因子应该更小（保留更多细节）
    2. 静音占比高 → 可以更激进地压缩静音
    3. 平均语音概率高 → 整体语音密集，需要更保守的加速
    4. 语音密度方差大 → 语音分布不均，需要更灵活的分级处理
    """
    
    # 基础推荐（基于倍速）
    from config import get_recommended_factors
    base_recommendation = get_recommended_factors(base_rate)
    
    high_density_factor = base_recommendation['high_density_factor']
    low_density_factor = base_recommendation['low_density_factor']
    
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
        'analysis_based': True
    }
