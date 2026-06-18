from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio
from PIL import Image
import io
import base64
import numpy as np
import sys
import os

# 添加 lang-segment-anything 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
lang_sam_path = os.path.join(current_dir, "../../../lang-segment-anything")
sys.path.insert(0, lang_sam_path)

from lang_sam import LangSAM
import torch, gc
import time
from collections import deque
import argparse
import logging
from contextlib import asynccontextmanager
import os
import datetime

class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    text: str   # prompt text

class PredictionResponse(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    masks: List[List[List[int]]]  # add mask field

class BatchProcessor:
    def __init__(self, max_batch_size: int = 8, max_queue_size: int = 100):
        self.model = LangSAM(gdino_model_ckpt_path="/mnt/wj/Model/grounding-dino-base", gdino_processor_ckpt_path="/mnt/wj/Model/grounding-dino-base")
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False
        self.start_time = time.time()
        self.port = args.port

    async def _print_queue_stats(self):
        while True:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            queue_size = self.request_queue.qsize()
            app.state.logger.info(f"Port: {self.port}, Uptime: {elapsed_time:.2f}s, Current queue size: {queue_size}")
            await asyncio.sleep(60)  # print every 60 seconds

    async def add_request(self, image: Image.Image, text: str):
        if self.request_queue.qsize() >= self.max_queue_size:
            raise HTTPException(
                status_code=503,
                detail="Server queue is full, please try again later"
            )
        
        future = asyncio.Future()
        await self.request_queue.put((image, text, future))
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        try:
            return await asyncio.wait_for(future, timeout=3000.0)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Processing timeout"
            )

    async def _process_batch(self):
        self.processing = True
        
        while not self.request_queue.empty():
            batch_images = []
            batch_texts = []
            batch_futures = []
            
            # Get all requests in current queue, up to max_batch_size
            try:
                while len(batch_images) < self.max_batch_size and not self.request_queue.empty():
                    # Modified here to use get_nowait() instead of await get_nowait()
                    image, text, future = self.request_queue.get_nowait()
                    if not future.cancelled():
                        batch_images.append(image)
                        batch_texts.append(text)
                        batch_futures.append(future)
                    self.request_queue.task_done()
            except asyncio.QueueEmpty:
                pass

            if batch_images:  # Only predict if there are requests
                try:
                    with torch.no_grad():
                        results = self.model.predict(
                            images_pil=batch_images,
                            texts_prompt=batch_texts,
                            box_threshold=0.3,
                            # box_threshold=0.1,
                            text_threshold=0.25
                        )
                    
                    for future, result in zip(batch_futures, results):
                        future.set_result({
                            "boxes": result["boxes"].tolist() if len(result["boxes"]) else [],
                            "labels": result["labels"] if len(result["labels"]) else [],
                            "masks": result["masks"].tolist() if len(result["masks"]) else []  # add mask results
                        })
                    
                    del results
                    gc.collect()
                    
                    # Log current batch processing stats
                    remaining_samples = self.request_queue.qsize()
                    app.state.logger.info(f"Batch processed, samples: {len(batch_images)}, remaining: {remaining_samples}")
                    
                except Exception as e:
                    # Create error log directory
                    error_dir = "error_logs"
                    os.makedirs(error_dir, exist_ok=True)
                    
                    # Get current timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save error info
                    error_log_path = os.path.join(error_dir, f"error_{timestamp}.txt")
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        f.write(f"Error: {str(e)}\n\n")
                        f.write("Texts:\n")
                        for i, text in enumerate(batch_texts):
                            f.write(f"{i}: {text}\n")
                    
                    # Save images
                    for i, img in enumerate(batch_images):
                        img_path = os.path.join(error_dir, f"error_{timestamp}_img_{i}.jpg")
                        img.save(img_path)
                    
                    app.state.logger.error(f"Error processing batch, logs saved to {error_dir}")
                    
                    for future in batch_futures:
                        future.set_exception(e)

        self.processing = False

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000, help="Service port")
parser.add_argument("--max_batch_size", type=int, default=8, help="Max batch size")
args = parser.parse_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.logger = logging.getLogger("uvicorn")
    stats_task = asyncio.create_task(processor._print_queue_stats())
    yield
    # Shutdown
    stats_task.cancel()

app = FastAPI(lifespan=lifespan)
processor = BatchProcessor(
    max_batch_size=args.max_batch_size,
    max_queue_size=10000  # set max queue length
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Submit request to batch processor
        result = await processor.add_request(image, request.text)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # args are executed in the above
    uvicorn.run("model_service:app", host="0.0.0.0", port=args.port, limit_concurrency=10000, backlog=10000, log_level="debug")
