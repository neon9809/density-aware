"""
参数推荐配置模块
根据基准倍速自动推荐最优的高密度和低密度因子

本项目代码由Manus AI完成。
"""

def get_recommended_factors(base_rate: float, strict_position: bool = False) -> dict:
    """
    根据基准倍速推荐最优的高密度和低密度因子。
    
    设计理念：
    
    【普通模式】
    - 基准倍速越高，高密度因子应该越小（保留更多重要内容的清晰度）
    - 基准倍速越高，低密度因子应该越大（更激进地压缩非核心内容）
    - 静音会被动态压缩以满足总时长约束
    - 在常见倍速范围（1.5x-2.0x）内提供最佳听感体验
    
    【严格相对位置模式】
    - 所有片段的相对位置必须严格按倍速缩放
    - 语音片中点位置 = 原位置 / 倍速
    - 高密度因子应更接近 1.0（语音部分需要更多压缩）
    - 低密度因子也应接近 1.0（保持停顿的比例关系）
    - 适用于需要与视频画面同步的场景
    
    Args:
        base_rate (float): 用户设定的基准倍速 (1.0 - 3.0)
        strict_position (bool): 是否启用严格相对位置模式
    
    Returns:
        dict: 包含推荐的 high_density_factor 和 low_density_factor
    """
    
    # 参数验证
    if base_rate < 1.0:
        base_rate = 1.0
    elif base_rate > 3.0:
        base_rate = 3.0
    
    if strict_position:
        # 严格相对位置模式：所有片段接近统一倍速
        return _get_strict_position_factors(base_rate)
    else:
        # 普通模式：差异化变速
        return _get_normal_factors(base_rate)


def _get_normal_factors(base_rate: float) -> dict:
    """
    普通模式的参数推荐
    通过差异化变速实现更好的听感
    """
    # 根据基准倍速区间推荐参数
    if base_rate <= 1.3:
        # 低速区：接近原速，保持自然
        high_density_factor = 0.95
        low_density_factor = 1.1
        
    elif base_rate <= 1.6:
        # 中低速区：常见的轻度加速
        high_density_factor = 0.90
        low_density_factor = 1.2
        
    elif base_rate <= 2.0:
        # 中速区：最常用的倍速范围，平衡效率和清晰度
        high_density_factor = 0.85
        low_density_factor = 1.3
        
    elif base_rate <= 2.5:
        # 中高速区：追求效率，但仍保留可理解性
        high_density_factor = 0.80
        low_density_factor = 1.5
        
    else:
        # 高速区：极限加速，最大化效率
        high_density_factor = 0.75
        low_density_factor = 1.8
    
    return {
        'high_density_factor': high_density_factor,
        'low_density_factor': low_density_factor,
        'description': _get_speed_description(base_rate, strict_position=False),
        'mode': 'normal'
    }


def _get_strict_position_factors(base_rate: float) -> dict:
    """
    严格相对位置模式的参数推荐
    
    核心原则：
    - 保持语音片段的精确相对位置
    - 语音片中点位置严格按倍速缩放
    - 首尾静音也按比例保持
    
    策略：
    - 高密度因子接近 1.0，让语音部分按接近基准倍速压缩
    - 低密度因子也接近 1.0，保持停顿的时间比例
    - 允许小幅调整以优化听感，但不能破坏相对位置
    """
    
    if base_rate <= 1.5:
        # 低倍速：允许稍微保护语音
        high_density_factor = 0.98
        low_density_factor = 1.02
        
    elif base_rate <= 2.0:
        # 中倍速：轻微差异化
        high_density_factor = 0.96
        low_density_factor = 1.04
        
    elif base_rate <= 2.5:
        # 中高倍速：稍微加大差异
        high_density_factor = 0.94
        low_density_factor = 1.06
        
    else:
        # 高倍速：最大允许差异
        high_density_factor = 0.92
        low_density_factor = 1.08
    
    return {
        'high_density_factor': high_density_factor,
        'low_density_factor': low_density_factor,
        'description': _get_speed_description(base_rate, strict_position=True),
        'mode': 'strict_position'
    }


def _get_speed_description(base_rate: float, strict_position: bool = False) -> str:
    """
    根据倍速和模式返回人类可读的描述
    """
    if strict_position:
        mode_desc = "【严格相对位置模式】"
        if base_rate <= 1.5:
            return f"{mode_desc}低倍速同步模式，适合精细剪辑"
        elif base_rate <= 2.0:
            return f"{mode_desc}标准同步模式，适合视频快放"
        elif base_rate <= 2.5:
            return f"{mode_desc}高速同步模式，保持画面对齐"
        else:
            return f"{mode_desc}极速同步模式，最大化压缩"
    else:
        if base_rate <= 1.3:
            return "轻度加速，保持自然听感"
        elif base_rate <= 1.6:
            return "适度加速，平衡效率与舒适度"
        elif base_rate <= 2.0:
            return "标准加速，推荐用于大多数场景"
        elif base_rate <= 2.5:
            return "高速模式，追求效率"
        else:
            return "极速模式，最大化时间节省"


# 预设配置方案
PRESETS = {
    'natural': {
        'name': '自然模式',
        'base_rate': 1.5,
        'high_density_factor': 0.90,
        'low_density_factor': 1.2,
        'strict_position': False,
        'description': '保持最自然的听感，适合初次使用'
    },
    'balanced': {
        'name': '平衡模式',
        'base_rate': 1.8,
        'high_density_factor': 0.85,
        'low_density_factor': 1.3,
        'strict_position': False,
        'description': '平衡效率和清晰度，推荐日常使用'
    },
    'efficient': {
        'name': '高效模式',
        'base_rate': 2.2,
        'high_density_factor': 0.80,
        'low_density_factor': 1.5,
        'strict_position': False,
        'description': '追求效率，适合熟悉的内容'
    },
    'extreme': {
        'name': '极速模式',
        'base_rate': 2.8,
        'high_density_factor': 0.75,
        'low_density_factor': 1.8,
        'strict_position': False,
        'description': '最大化时间节省，适合快速浏览'
    },
    'video_sync': {
        'name': '视频同步模式',
        'base_rate': 2.0,
        'high_density_factor': 0.96,
        'low_density_factor': 1.04,
        'strict_position': True,
        'description': '严格保持相对位置，适合视频快放和平台集成'
    }
}
