"""
参数推荐配置模块
根据基准倍速自动推荐最优的高密度和低密度因子

本项目代码由Manus AI完成。
"""

def get_recommended_factors(base_rate: float) -> dict:
    """
    根据基准倍速推荐最优的高密度和低密度因子。
    
    设计理念：
    - 基准倍速越高，高密度因子应该越小（保留更多重要内容的清晰度）
    - 基准倍速越高，低密度因子应该越大（更激进地压缩非核心内容）
    - 在常见倍速范围（1.5x-2.0x）内提供最佳听感体验
    
    Args:
        base_rate (float): 用户设定的基准倍速 (1.0 - 3.0)
    
    Returns:
        dict: 包含推荐的 high_density_factor 和 low_density_factor
    """
    
    # 参数验证
    if base_rate < 1.0:
        base_rate = 1.0
    elif base_rate > 3.0:
        base_rate = 3.0
    
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
        'description': _get_speed_description(base_rate)
    }


def _get_speed_description(base_rate: float) -> str:
    """
    根据倍速返回人类可读的描述
    """
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
        'description': '保持最自然的听感，适合初次使用'
    },
    'balanced': {
        'name': '平衡模式',
        'base_rate': 1.8,
        'high_density_factor': 0.85,
        'low_density_factor': 1.3,
        'description': '平衡效率和清晰度，推荐日常使用'
    },
    'efficient': {
        'name': '高效模式',
        'base_rate': 2.2,
        'high_density_factor': 0.80,
        'low_density_factor': 1.5,
        'description': '追求效率，适合熟悉的内容'
    },
    'extreme': {
        'name': '极速模式',
        'base_rate': 2.8,
        'high_density_factor': 0.75,
        'low_density_factor': 1.8,
        'description': '最大化时间节省，适合快速浏览'
    }
}
