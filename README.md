# 稠密感知快放算法 (Density-Aware Speed-Up Algorithm)

一个革命性的音频变速工具，通过智能识别语音密度，实现比传统整体倍速更高的识别度和更自然的听感。

**本项目代码由 Manus AI 完成。**

[![GitHub](https://img.shields.io/badge/GitHub-neon9809%2Fdensity--aware-blue?logo=github)](https://github.com/neon9809/density-aware)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ghcr.io-2496ED?logo=docker)](https://github.com/neon9809/density-aware/pkgs/container/density-aware)

---

## 🌟 为什么选择稠密感知快放？

传统的音频倍速播放存在一个根本问题：**人说话快并不是整体快，必要的停顿仍然相似**。但整体快放后，停顿也被压缩了，导致听感不自然、理解困难。

**稠密感知快放算法**通过以下创新解决了这个问题：

1. **智能语音密度识别**：使用 Silero-VAD 模型分析音频，识别高密度语音、低密度语音和静音
2. **分级变速处理**：对不同密度的音频片段应用不同的变速系数
3. **音频特征感知**：根据每个音频的独特特征（语音密度分布、静音比例等）动态推荐最优参数
4. **精确位置保持**：通过数学约束确保处理后的音频保持原有的时间结构

**实测效果**：相比传统整体倍速，识别度显著提高，听感更加自然流畅！

---

## ✨ 核心特点

### 🎯 音频感知的智能推荐
- 自动分析上传音频的语音密度特征
- 根据分析结果动态推荐最优参数
- 每个音频都获得量身定制的处理方案

### 📊 三级分类变速处理
- **高密度语音** (重要内容)：应用较小的变速系数，保留清晰度
- **低密度语音** (停顿、语气词)：应用中等的变速系数，压缩非核心内容
- **静音** (间隙)：应用最大的变速系数，快速跳过

### 🔗 相对位置精确保持
通过数学约束确保：
- 语音块之间的相对位置不变
- 整体播放时长精确符合用户指定的倍速
- 听感自然流畅，不会出现"时间漂移"

### 🎵 高质量变速处理
采用 **WSOLA (Waveform Similarity-based Overlap-Add)** 算法实现"变速不变调"效果，保证音频质量。

### 🖥️ 友好的用户界面
- 简洁直观的 Web 界面
- 实时预分析和进度显示
- 可视化的音频特征报告
- 可选的高级参数微调

---

## 📋 工作原理

```
用户上传音频
   ↓
音频特征预分析 (VAD)
   ├─ 语音密度分布
   ├─ 静音比例
   └─ 平均语音概率
   ↓
智能参数推荐
   ├─ 高密度因子
   ├─ 低密度因子
   └─ 推荐理由
   ↓
三级片段分类
   ├─ 高密度语音
   ├─ 低密度语音
   └─ 静音
   ↓
多速度变速处理 (WSOLA)
   ↓
无缝拼接
   ↓
输出音频
```

### 数学模型

设原始音频总时长为 $T_{orig}$，用户指定的基准倍速为 $R$，则目标输出时长为 $T_{target} = T_{orig} / R$。

设：
- $T_h$：高密度语音总时长
- $T_l$：低密度语音总时长  
- $T_s$：静音总时长
- $\alpha$：高密度语音速度调节因子 (0 < α ≤ 1)
- $\beta$：低密度语音速度调节因子 (β ≥ 1)

则：
- 高密度语音速度：$v_h = R \cdot \alpha$
- 低密度语音速度：$v_l = R \cdot \beta$
- 静音速度：$v_s = \frac{T_s}{T_{target} - \frac{T_h}{v_h} - \frac{T_l}{v_l}}$

这个公式确保了 $\frac{T_h}{v_h} + \frac{T_l}{v_l} + \frac{T_s}{v_s} = T_{target}$，从而精确保持相对位置。

### 音频感知推荐策略

算法根据预分析结果动态调整参数：

| 音频特征 | 参数调整策略 | 理由 |
|---------|------------|------|
| 高密度语音占比 > 60% | 降低高密度因子 | 语音密集，需保留更多细节 |
| 高密度语音占比 < 40% | 提高高密度因子 | 语音稀疏，可更激进加速 |
| 静音占比 > 40% | 提高低密度因子 | 静音多，可加强压缩 |
| 静音占比 < 15% | 降低低密度因子 | 语音连续，需保持流畅 |
| 语音密度方差大 | 启用精细分级 | 分布不均，分级处理更重要 |

---

## 🚀 快速开始

### 方式 1: Docker (推荐)

最简单的方式是使用 Docker 镜像：

```bash
docker run -d \
  -p 8000:8000 \
  --name density-aware \
  ghcr.io/neon9809/density-aware:latest
```

然后在浏览器中打开 `http://localhost:8000`

### 方式 2: 本地部署

#### 前置要求
- Python 3.8+
- FFmpeg
- Rubberband CLI

#### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/neon9809/density-aware.git
cd density-aware
```

2. **创建虚拟环境**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **安装系统工具**

**Ubuntu/Debian:**
```bash
sudo apt-get install rubberband-cli ffmpeg
```

**macOS:**
```bash
brew install rubberband ffmpeg
```

**Windows:**
下载 Rubberband 和 FFmpeg 预编译二进制文件，添加到 PATH

5. **启动服务**
```bash
python main.py
```

访问 `http://localhost:8000`

---

## 📖 使用指南

### Web 界面使用流程

1. **上传音频文件**
   - 支持 MP3, WAV, FLAC, OGG 等常见格式
   - 文件大小无特殊限制（取决于服务器内存）

2. **设定播放倍速**
   - 拖动滑块选择 1.0x ~ 3.0x 的倍速
   - 算法会根据音频特征自动优化参数

3. **（可选）查看高级选项**
   - 点击"显示高级选项"
   - 系统会自动分析音频特征，显示：
     - 语音密度分布统计
     - 推荐参数及理由
   - 可以手动微调参数（系统会标记为"自定义模式"）

4. **开始处理**
   - 点击"开始处理"按钮
   - 如果未预分析，系统会自动先分析再处理
   - 等待处理完成

5. **试听和下载**
   - 在线试听处理结果
   - 下载处理后的音频文件

### 参数说明

| 参数 | 范围 | 默认值 | 说明 |
|-----|------|--------|------|
| **基准倍速** | 1.0x ~ 3.0x | 1.8x | 整体播放速度目标 |
| **高密度因子** | 0.5 ~ 1.0 | 自动推荐 | 越小越清晰，保留更多重要内容细节 |
| **低密度因子** | 1.0 ~ 2.0 | 自动推荐 | 越大压缩越多，跳过更多停顿和语气词 |

**推荐做法**：只设置基准倍速，让算法自动分析并推荐其他参数。

### API 接口

#### 1. 分析音频特征

```bash
curl -X POST "http://localhost:8000/analyze-audio/" \
  -F "file=@your_audio.mp3" \
  -F "base_rate=1.8"
```

**响应示例：**
```json
{
  "success": true,
  "analysis": {
    "total_duration": 78.38,
    "high_density_ratio": 0.523,
    "low_density_ratio": 0.287,
    "silence_ratio": 0.190,
    "avg_speech_prob": 0.612,
    "speech_variance": 0.085
  },
  "recommendation": {
    "high_density_factor": 0.85,
    "low_density_factor": 1.3,
    "description": "标准加速，推荐用于大多数场景；语音密度适中；检测到较多静音，已加强静音压缩",
    "analysis_based": true
  }
}
```

#### 2. 处理音频

```bash
curl -X POST "http://localhost:8000/process-audio/" \
  -F "file=@your_audio.mp3" \
  -F "base_rate=1.8" \
  -F "use_recommended=true" \
  -o output.mp3
```

**参数说明：**
- `file`: 音频文件（必需）
- `base_rate`: 基准倍速，默认 1.8
- `use_recommended`: 是否使用推荐参数，默认 true
- `high_density_factor`: 高密度因子（仅当 use_recommended=false 时需要）
- `low_density_factor`: 低密度因子（仅当 use_recommended=false 时需要）

**响应：** 处理后的音频文件（MP3格式）

---

## 🎯 使用场景

### 📚 有声书和播客
- 快速消费长篇内容，节省时间
- 保留重要信息的清晰度
- 自然的听感体验，不会感到"机械"

### 🎓 在线课程和讲座
- 加快学习进度，提高效率
- 压缩讲师的停顿和重复
- 保持核心知识的可理解性

### 📞 会议录音和采访
- 快速回放会议内容
- 压缩不必要的间隙和停顿
- 保留关键发言的清晰度

### 🎙️ 语言学习
- 调整播放速度以适应学习进度
- 保留发音的自然性
- 灵活控制学习节奏

### 📺 视频教程和演讲
- 提高观看效率
- 跳过冗长的停顿
- 保持内容的连贯性

---

## 🔧 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **后端框架** | FastAPI | 高性能异步Web框架 |
| **服务器** | Uvicorn | ASGI服务器 |
| **语音检测** | Silero-VAD | 轻量级、高精度的语音活动检测 |
| **变速处理** | Pyrubberband | WSOLA算法的Python接口 |
| **音频处理** | PyDub | 简单易用的音频操作库 |
| **数值计算** | NumPy | 高效的数组运算 |
| **深度学习** | PyTorch | VAD模型推理 |
| **前端** | HTML5 + CSS3 + Vanilla JS | 无依赖的轻量级前端 |
| **容器化** | Docker | 一键部署 |

---

## 📊 性能指标

基于测试数据（78秒音频，基准倍速1.8x）：

| 指标 | 值 |
|------|-----|
| 预分析时间 | ~2.5 秒 |
| 处理时间 | ~4.6 秒 |
| 总耗时 | ~7.1 秒 |
| 输出时长 | 43.55 秒 |
| 实现倍速 | 1.80x (精确) |
| 片段数 | 125 个 |
| 内存占用 | ~500MB |

---

## 🐳 Docker 部署

### 使用预构建镜像（推荐）

```bash
# 拉取镜像
docker pull ghcr.io/neon9809/density-aware:latest

# 运行容器
docker run -d \
  -p 8000:8000 \
  --name density-aware \
  ghcr.io/neon9809/density-aware:latest
```

### 本地构建镜像

```bash
# 构建镜像
docker build -t density-aware:latest .

# 运行容器
docker run -d \
  -p 8000:8000 \
  --name density-aware \
  density-aware:latest
```

### Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  density-aware:
    image: ghcr.io/neon9809/density-aware:latest
    ports:
      - "8000:8000"
    restart: unless-stopped
    environment:
      - TZ=Asia/Shanghai
```

启动：
```bash
docker-compose up -d
```

---

## 📁 项目结构

```
density-aware/
├── algorithm.py          # 核心算法实现
├── analyzer.py           # 音频预分析模块
├── config.py             # 参数推荐配置
├── main.py               # FastAPI 应用主程序
├── requirements.txt      # Python 依赖
├── Dockerfile            # Docker 镜像构建文件
├── static/
│   └── index.html        # 前端网页界面
├── temp_files/           # 临时文件目录（运行时创建）
└── README.md             # 项目文档
```

---

## 🐛 故障排除

### 问题：处理失败，显示 "No module named 'packaging'"
**解决方案：** 安装缺失的依赖
```bash
pip install packaging
```

### 问题：Silero-VAD 模型下载缓慢
**解决方案：** 模型会被缓存在 `~/.cache/torch/hub/` 中。第一次运行会较慢，后续运行会使用缓存。

### 问题：处理时间过长
**解决方案：** 
- 检查系统资源（CPU、内存）
- 减少音频文件大小
- 使用更快的硬件

### 问题：输出音频质量下降
**解决方案：**
- 调整 `high_density_factor` 为更小的值（如 0.7）
- 减少 `base_rate` 值
- 确保输入音频质量良好

### 问题：Docker 容器无法启动
**解决方案：**
- 检查端口 8000 是否被占用：`lsof -i :8000`
- 查看容器日志：`docker logs density-aware`
- 确保 Docker 有足够的内存分配（至少 2GB）

---

## 🔬 算法原理深度解析

### 为什么传统倍速不够好？

人类说话的节奏包含两个维度：
1. **语音内容的速度**（音节密度）
2. **停顿和间隙**（呼吸、思考、强调）

传统倍速播放对这两个维度一视同仁，导致：
- 重要内容被压缩得过快，理解困难
- 停顿被过度压缩，失去自然节奏
- 整体听感"机械"、"赶"

### 稠密感知算法的创新

本算法通过以下三个层次解决问题：

#### 第一层：语音活动检测 (VAD)
使用 Silero-VAD 模型对每 32ms 的音频片段计算语音概率（0~1），精确识别哪些部分是语音、哪些是静音。

#### 第二层：密度分级
根据语音概率将音频分为三级：
- **高密度语音** (prob > 0.7)：连续、快速的语音内容
- **低密度语音** (0.3 < prob ≤ 0.7)：停顿、语气词、过渡音
- **静音** (prob ≤ 0.3)：间隙、呼吸、长停顿

#### 第三层：自适应变速
对不同密度的片段应用不同的变速系数，并通过数学约束确保：
1. 整体时长精确符合用户指定的倍速
2. 各片段的相对位置保持不变
3. 听感自然流畅

### 音频感知推荐的价值

不同的音频有不同的特征：
- **演讲类**：语音密集，停顿少
- **对话类**：语音分布不均，停顿多
- **教学类**：语音密度适中，强调停顿

通过预分析音频特征，算法可以为每个音频量身定制最优参数，实现最佳效果。

---

## 📚 参考文献

1. **Silero-VAD**: https://github.com/snakers4/silero-vad
   - 轻量级、高精度的语音活动检测模型

2. **Pyrubberband**: https://github.com/bmcfee/pyrubberband
   - Rubberband库的Python接口

3. **WSOLA 算法**: Verhelst, W., & Rombouts, G. (2002). *An overlap-add technique based on waveform similarity (WSOLA) for high quality time-scale modification of speech.*

4. **FastAPI**: https://fastapi.tiangolo.com/
   - 现代、快速的 Python Web 框架

---

## 📄 许可证

MIT License

Copyright (c) 2025 neon9809

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

如果您有任何改进建议或发现了 bug，请随时：
1. 提交 Issue 描述问题
2. Fork 本仓库并创建您的分支
3. 提交 Pull Request

---

## 💬 反馈和支持

如有问题或建议，请在 GitHub 上提交 Issue：
https://github.com/neon9809/density-aware/issues

---

## 🙏 致谢

- **Manus AI**: 项目核心算法、音频感知系统和 Web 应用开发
- **Silero Team**: 提供高质量的 VAD 模型
- **Rubberband**: 提供优秀的音频变速库
- **FastAPI**: 提供现代化的 Web 框架

---

## 🌐 相关链接

- **项目主页**: https://github.com/neon9809/density-aware
- **Docker 镜像**: https://github.com/neon9809/density-aware/pkgs/container/density-aware
- **问题反馈**: https://github.com/neon9809/density-aware/issues

---

**本项目代码由 Manus AI 完成。**

如果这个项目对您有帮助，请给我们一个 ⭐️ Star！
