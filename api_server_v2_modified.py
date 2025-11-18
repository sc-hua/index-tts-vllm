import os
import asyncio
import io
import traceback
import base64
import hashlib
import tempfile
from pathlib import Path
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

tts = None

# 临时文件目录
TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def decode_base64_audio(base64_str: str, prefix: str = "audio") -> str:
    """
    将 base64 音频数据解码并保存到临时文件
    
    Args:
        base64_str: base64 编码的音频数据（可能包含 data:audio/wav;base64, 前缀）
        prefix: 文件名前缀
    
    Returns:
        临时文件的路径
    """
    # 移除 data URL 前缀（如果存在）
    if base64_str.startswith('data:'):
        # data:audio/wav;base64,xxxxx
        base64_str = base64_str.split(',', 1)[1] if ',' in base64_str else base64_str
    
    try:
        # 解码 base64
        audio_bytes = base64.b64decode(base64_str)
        
        # 使用内容的 hash 作为文件名（去重）
        content_hash = hashlib.md5(audio_bytes).hexdigest()
        temp_file_path = TEMP_AUDIO_DIR / f"{prefix}_{content_hash}.wav"
        
        # 如果文件已存在，直接返回路径（缓存）
        if temp_file_path.exists():
            logger.debug(f"使用缓存的音频文件: {temp_file_path}")
            return str(temp_file_path)
        
        # 保存到临时文件
        with open(temp_file_path, 'wb') as f:
            f.write(audio_bytes)
        
        logger.info(f"保存 base64 音频到临时文件: {temp_file_path} ({len(audio_bytes)} bytes)")
        return str(temp_file_path)
        
    except Exception as e:
        logger.error(f"解码 base64 音频失败: {e}")
        raise ValueError(f"无效的 base64 音频数据: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    """
    TTS 合成接口（支持文件路径和 base64）
    
    支持的音频输入方式：
    1. spk_audio_path: 文件系统路径（向后兼容）
    2. spk_audio_base64: base64 编码的音频数据（新增）
    
    优先级：spk_audio_base64 > spk_audio_path
    """
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        
        # ============ 说话人音频处理 ============
        spk_audio_path = None
        spk_audio_base64 = data.get("spk_audio_base64")
        
        if spk_audio_base64:
            # 如果提供了 base64，解码并保存到临时文件
            logger.info("检测到 spk_audio_base64，正在解码...")
            spk_audio_path = decode_base64_audio(spk_audio_base64, prefix="speaker")
        else:
            # 使用文件路径（原有逻辑）
            spk_audio_path = data.get("spk_audio_path")
            if not spk_audio_path:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "error": "必须提供 spk_audio_path 或 spk_audio_base64"
                    }
                )
        
        # ============ 情感参考音频处理 ============
        emo_ref_path = None
        emo_ref_base64 = data.get("emo_ref_base64")
        
        if emo_ref_base64:
            # 如果提供了情感参考音频的 base64
            logger.info("检测到 emo_ref_base64，正在解码...")
            emo_ref_path = decode_base64_audio(emo_ref_base64, prefix="emotion")
        else:
            emo_ref_path = data.get("emo_ref_path", None)
        
        # ============ 其他参数 ============
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        logger.info(f"TTS 推理 - 说话人音频: {spk_audio_path}, 情感模式: {emo_control_method}")
        
        # 执行 TTS 推理
        sr, wav = await tts.infer(
            spk_audio_prompt=spk_audio_path, 
            text=text,
            output_path=None,
            emo_audio_prompt=emo_ref_path, 
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method==3), 
            emo_text=emo_text,
            use_random=emo_random,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence)
        )
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        logger.error(f"TTS 推理失败: {tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts_multipart", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_multipart(
    text: str = Form(...),
    speaker_audio: UploadFile = File(...),
    emo_control_method: int = Form(0),
    emo_ref_audio: Optional[UploadFile] = File(None),
    emo_weight: float = Form(1.0),
    emo_vec: str = Form("[0,0,0,0,0,0,0,0]"),
    emo_text: Optional[str] = Form(None),
    emo_random: bool = Form(False),
    max_text_tokens_per_sentence: int = Form(120)
):
    """
    TTS 合成接口（multipart/form-data 方式）
    
    这个接口完全模仿 SoulX-Podcast 的使用方式，支持直接上传音频文件
    """
    try:
        # 保存上传的说话人音频到临时文件
        speaker_audio_content = await speaker_audio.read()
        speaker_hash = hashlib.md5(speaker_audio_content).hexdigest()
        speaker_temp_path = TEMP_AUDIO_DIR / f"speaker_{speaker_hash}.wav"
        
        with open(speaker_temp_path, 'wb') as f:
            f.write(speaker_audio_content)
        
        logger.info(f"上传的说话人音频: {speaker_audio.filename} -> {speaker_temp_path}")
        
        # 处理情感参考音频
        emo_ref_path = None
        if emo_ref_audio:
            emo_ref_content = await emo_ref_audio.read()
            emo_ref_hash = hashlib.md5(emo_ref_content).hexdigest()
            emo_ref_temp_path = TEMP_AUDIO_DIR / f"emotion_{emo_ref_hash}.wav"
            
            with open(emo_ref_temp_path, 'wb') as f:
                f.write(emo_ref_content)
            
            emo_ref_path = str(emo_ref_temp_path)
            logger.info(f"上传的情感音频: {emo_ref_audio.filename} -> {emo_ref_path}")
        
        # 解析情感向量
        try:
            emo_vec_list = json.loads(emo_vec) if isinstance(emo_vec, str) else emo_vec
        except:
            emo_vec_list = [0] * 8
        
        # 执行推理
        global tts
        
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        
        vec = emo_vec_list if emo_control_method == 2 else None
        if vec:
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        
        sr, wav = await tts.infer(
            spk_audio_prompt=str(speaker_temp_path),
            text=text,
            output_path=None,
            emo_audio_prompt=emo_ref_path,
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method==3),
            emo_text=emo_text,
            use_random=emo_random,
            max_text_tokens_per_sentence=max_text_tokens_per_sentence
        )
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()
        
        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        logger.error(f"TTS 推理失败 (multipart): {tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7879)
    parser.add_argument("--model_dir", type=str, default="/data/proj/hsc/ckpts/indextts-2-vllm", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=True, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.2)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose mode")
    args = parser.parse_args()
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)
