"""
稠密感知快放算法 Web服务
FastAPI应用主程序

本项目代码由Manus AI完成。
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import shutil
import os
import uuid
import asyncio
from algorithm import intelligent_speed_up_v2
from config import get_recommended_factors, PRESETS
from analyzer import analyze_audio_characteristics

# 创建FastAPI应用
app = FastAPI(
    title="稠密感知快放算法 API",
    description="一个智能的、语音密度感知的音频变速服务",
    version="1.0.0"
)

# 创建用于存放临时文件的目录
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# 挂载 static 目录
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ==============================================================================
# 路由端点
# ==============================================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    """
    返回主页面 (index.html)
    """
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>稠密感知快放算法 API</h1><p>欢迎使用！</p><p><a href='/docs'>查看API文档</a></p>")

@app.post("/process-audio/")
async def process_audio_endpoint(
    file: UploadFile = File(...),
    base_rate: float = Form(1.8),
    high_density_factor: float = Form(None),
    low_density_factor: float = Form(None),
    use_recommended: bool = Form(True)
):
    """
    上传音频文件，应用"稠密感知快放算法"，并返回处理后的文件。
    
    参数说明：
    - base_rate: 基准倍速（1.0-3.0），例如1.8表示1.8倍速
    - high_density_factor: 高密度语音调节因子（0.5-1.0），越小越慢，保留更多细节
    - low_density_factor: 低密度语音调节因子（1.0-2.0），越大越快，压缩非核心内容
    """
    
    # 参数验证
    if not (1.0 <= base_rate <= 3.0):
        raise HTTPException(status_code=400, detail="base_rate 必须在 1.0 到 3.0 之间")
    
    # 如果用户选择使用推荐值或未提供因子，则自动计算
    if use_recommended or high_density_factor is None or low_density_factor is None:
        recommended = get_recommended_factors(base_rate)
        high_density_factor = recommended['high_density_factor']
        low_density_factor = recommended['low_density_factor']
        print(f"使用推荐参数: {recommended['description']}")
    else:
        # 验证用户提供的自定义参数
        if not (0.5 <= high_density_factor <= 1.0):
            raise HTTPException(status_code=400, detail="high_density_factor 必须在 0.5 到 1.0 之间")
        if not (1.0 <= low_density_factor <= 2.0):
            raise HTTPException(status_code=400, detail="low_density_factor 必须在 1.0 到 2.0 之间")
        print(f"使用自定义参数")
    
    # 为本次请求生成唯一ID
    request_id = str(uuid.uuid4())
    input_filename = f"{request_id}_{file.filename}"
    output_filename = f"processed_{input_filename}"
    
    input_path = os.path.join(TEMP_DIR, input_filename)
    output_path = os.path.join(TEMP_DIR, output_filename)

    try:
        # 保存上传的音频文件到临时位置
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"\n{'='*60}")
        print(f"开始处理文件: {input_path}")
        print(f"参数: base_rate={base_rate}, high_density_factor={high_density_factor}, low_density_factor={low_density_factor}")
        print(f"{'='*60}\n")
        
        # 调用核心算法进行处理 (在线程池中运行以避免阻塞)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            intelligent_speed_up_v2,
            input_path,
            output_path,
            base_rate,
            high_density_factor,
            low_density_factor
        )

        # 检查输出文件是否存在
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="算法处理失败，未能生成输出文件。")

        print(f"处理完成，输出文件: {output_path}\n")

        # 返回处理后的文件
        return FileResponse(
            path=output_path,
            media_type='audio/mpeg',
            filename=f"processed_{file.filename}"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"处理出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理音频时发生错误: {str(e)}")
    
    finally:
        # 清理输入文件
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass

@app.get("/presets", tags=["Configuration"])
async def get_presets():
    """
    获取预设配置方案
    """
    return PRESETS

@app.get("/recommend", tags=["Configuration"])
async def get_recommendation(base_rate: float = 1.8):
    """
    根据基准倍速获取推荐的参数配置（不考虑音频特征）
    """
    if not (1.0 <= base_rate <= 3.0):
        raise HTTPException(status_code=400, detail="base_rate 必须在 1.0 到 3.0 之间")
    return get_recommended_factors(base_rate)

@app.post("/analyze-audio/", tags=["Configuration"])
async def analyze_audio_endpoint(
    file: UploadFile = File(...),
    base_rate: float = Form(1.8)
):
    """
    分析上传的音频文件，返回语音密度特征和推荐参数
    
    这个端点用于在用户展开高级选项或开始处理前进行预分析
    """
    if not (1.0 <= base_rate <= 3.0):
        raise HTTPException(status_code=400, detail="base_rate 必须在 1.0 到 3.0 之间")
    
    # 生成唯一ID
    request_id = str(uuid.uuid4())
    input_filename = f"analyze_{request_id}_{file.filename}"
    input_path = os.path.join(TEMP_DIR, input_filename)
    
    try:
        # 保存上传的文件
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 分析音频特征
        result = await asyncio.to_thread(
            analyze_audio_characteristics,
            input_path,
            base_rate
        )
        
        return result
        
    except Exception as e:
        print(f"分析音频时出错: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分析音频时发生错误: {str(e)}")
    
    finally:
        # 清理临时文件
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass

@app.get("/health", tags=["Health"])
async def health_check():
    """
    健康检查端点
    """
    return {"status": "healthy", "service": "稠密感知快放算法 API"}

# ==============================================================================
# 启动应用
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
