from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class PredictInput(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=100)

@app.middleware("http")
async def log_req(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    cost_ms = round((time.time() - start) * 1000, 2)
    logger.info(f"请求路径={request.url.path}, 请求方式={request.method}, 耗时={cost_ms}ms")
    return resp

@app.get("/health")
def health():
    return {"status": "正常", "消息": "服务正在运行"}

@app.post("/predict")
def predict(input_data: PredictInput):
    try:
        text = input_data.input_text
        pred_result = f"模型输出结果：{text}"
        return {"ok": True, "result": pred_result}
    except Exception as e:
        logger.error(f"推理出错: {str(e)}")
        raise HTTPException(status_code=500, detail="模型推理失败")

@app.exception_handler(Exception)
async def global_err_handler(req, exc):
    logger.error(f"未知错误: {exc}")
    raise HTTPException(status_code=500, detail="服务器内部错误")
