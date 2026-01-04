# 稠密感知快放算法 (Density-Aware Speed-Up Algorithm)

一个智能的、语音密度感知的音频变速服务。通过识别音频中的语音密度，对不同密度的语音和静音分别应用不同的变速系数，在提高播放效率的同时保持听感自然。

**本项目代码由 Manus AI 完成。** | [GitHub](https://github.com/neon9809/density-aware)

---

## ✨ 核心特点

### 🎯 智能语音密度识别
- 使用 **Silero-VAD** 模型精确检测音频中的语音概率
- 将音频自动分为三个等级：**高密度语音**、**低密度语音**、**静音**
- 每个等级对应不同的信息密度和重要程度

### 📊 三级变速处理
- **高密度语音** (重要内容)：应用较小的变速系数，保留清晰度
- **低密度语音** (停顿、语气词)：应用较大的变速系数，压缩非核心内容
- **静音** (间隙)：应用最大的变速系数，快速跳过

### 🔗 相对位置精确保持
通过数学约束确保处理后的音频保持原有的时间结构，使得：
- 语音块之间的相对位置不变
- 整体播放时长精确符合用户指定的倍速
- 听感自然流畅，不会出现"时间漂移"

### 🎵 高质量变速处理
采用 **WSOLA (Waveform Similarity-based Overlap-Add)** 算法实现"变速不变调"效果，保证音频质量。

---

## 📋 工作原理

```
输入音频 
   ↓
VAD语音概率分析 (Silero-VAD)
   ↓
三级片段分类 (高密度/低密度/静音)
   ↓
多速度变速处理 (Pyrubberband WSOLA)
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
下载 Rubberband 预编译二进制文件，添加到 PATH

5. **启动服务**
```bash
python main.py
```

访问 `http://localhost:8000`

---

## 📖 使用指南

### Web 界面

打开应用后，您会看到一个直观的网页界面：

1. **上传音频文件**
   - 支持 MP3、WAV、FLAC、OGG 等常见格式
   - 文件大小无特殊限制（取决于服务器内存）

2. **调整参数**
   - **基准倍速** (1.0x ~ 3.0x)：整体播放速度目标
     - 1.0x = 原速
     - 1.5x = 1.5倍速（节省33%时间）
     - 2.0x = 2倍速（节省50%时间）
   
   - **高密度语音调节** (0.5 ~ 1.0)：保留重要内容的清晰度
     - 越小越清晰（保留更多细节）
     - 推荐值：0.8 ~ 0.9
   
   - **低密度语音调节** (1.0 ~ 2.0)：压缩非核心语音
     - 越大压缩越多（跳过更多停顿）
     - 推荐值：1.2 ~ 1.5

3. **开始处理**
   - 点击"开始处理"按钮
   - 等待处理完成（时间取决于音频长度）

4. **试听和下载**
   - 处理完成后，页面会显示音频播放器
   - 可以直接试听处理结果
   - 点击"下载结果"保存处理后的音频

### API 接口

如果您想以编程方式调用服务，可以使用 REST API：

```bash
curl -X POST "http://localhost:8000/process-audio/" \
  -F "file=@your_audio.mp3" \
  -F "base_rate=1.8" \
  -F "high_density_factor=0.9" \
  -F "low_density_factor=1.2" \
  -o output.mp3
```

**参数说明：**
- `file`: 音频文件（必需）
- `base_rate`: 基准倍速，默认 1.8
- `high_density_factor`: 高密度因子，默认 0.9
- `low_density_factor`: 低密度因子，默认 1.2

**响应：** 处理后的音频文件（MP3格式）

---

## 🎯 使用场景

### 📚 有声书和播客
- 快速消费长篇内容
- 保留重要信息的清晰度
- 自然的听感体验

### 🎓 在线课程和讲座
- 加快学习进度
- 压缩讲师的停顿和重复
- 保持核心知识的可理解性

### 📞 会议录音和采访
- 快速回放会议内容
- 压缩不必要的间隙
- 保留关键发言的清晰度

### 🎙️ 语言学习
- 调整播放速度以适应学习进度
- 保留发音的自然性
- 灵活控制学习节奏

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
| 处理时间 | ~4.6 秒 |
| 输出时长 | 43.55 秒 |
| 实现倍速 | 1.80x (精确) |
| 片段数 | 125 个 |
| 内存占用 | ~500MB |

---

## 🐳 Docker 部署

### 构建镜像

```bash
docker build -t density-aware:latest .
```

### 运行容器

```bash
docker run -d \
  -p 8000:8000 \
  --name density-aware \
  -v /path/to/uploads:/app/uploads \
  density-aware:latest
```

### 使用 ghcr.io 镜像

```bash
docker pull ghcr.io/neon9809/density-aware:latest
docker run -d -p 8000:8000 ghcr.io/neon9809/density-aware:latest
```

---

## 📝 配置文件

### `requirements.txt`
列出所有Python依赖。可以通过修改版本号来调整依赖版本。

### `algorithm.py`
核心算法实现。包含VAD分析、片段分类和变速处理的所有逻辑。

### `main.py`
FastAPI应用主程序。定义API端点和Web服务配置。

### `static/index.html`
前端网页界面。包含参数调整、文件上传和结果播放的UI。

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

---

## 📚 参考文献

1. **Silero-VAD**: https://github.com/snakers4/silero-vad
   - 轻量级、高精度的语音活动检测模型

2. **Pyrubberband**: https://github.com/bmcfee/pyrubberband
   - Rubberband库的Python接口

3. **WSOLA 算法**: Verhelst, W., & Rombouts, G. (2002). An overlap-add technique based on waveform similarity (WSOLA) for high quality time-scale modification of speech.

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 💬 反馈和支持

如有问题或建议，请在 GitHub 上提交 Issue。

---

## 🙏 致谢

- **Manus AI**: 项目核心算法和Web应用开发
- **Silero**: 提供高质量的VAD模型
- **Rubberband**: 提供优秀的音频变速库

---

**项目链接**: https://github.com/neon9809/density-aware

**本项目代码由 Manus AI 完成。**
