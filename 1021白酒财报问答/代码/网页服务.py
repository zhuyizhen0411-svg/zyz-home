# -*- coding: utf-8 -*-
"""财报问答服务（FastAPI）。

启动：
    python 代码/serve.py --port 7860
    # 然后浏览器打开 http://127.0.0.1:7860

接口：
    GET  /            问答页面
    GET  /api/stats   语料统计
    POST /api/search  只召回，不生成回答
    POST /api/ask     召回 + 本地大模型生成带出处的回答
"""

import argparse
import os
import sys
import time
from contextlib import asynccontextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "代码"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from 生成回答 import ANSWER_MODEL, answer as answer_with_citations  # noqa: E402
from 引用追溯 import get_chunk  # noqa: E402
from 检索引擎 import Retriever  # noqa: E402

R = None


@asynccontextmanager
async def lifespan(app_):
    global R
    t = time.time()
    R = Retriever(with_model=True)
    print(f"[boot] 索引加载完成 {time.time()-t:.1f}s", flush=True)
    yield


app = FastAPI(title="白酒财报问答库")
app.router.lifespan_context = lifespan

# 允许从静态预览页（其它端口）直接调用本服务，避免相对路径请求打到静态服务器
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _err(request: Request, exc: Exception, status: int = 500):
    """任何异常都以 JSON 返回，绝不返回空响应——否则前端 response.json() 会直接崩。"""
    import traceback

    traceback.print_exc()
    return JSONResponse(
        status_code=status,
        content={"error": f"{type(exc).__name__}: {exc}", "path": str(request.url.path)},
    )


@app.exception_handler(Exception)
async def on_exception(request: Request, exc: Exception):
    return _err(request, exc, 500)


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "请求参数不合法", "detail": str(exc).replace("\n", " ")[:400]},
    )


@app.exception_handler(StarletteHTTPException)
async def on_http_exception(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


def _retrieve(req):
    """统一检索入口：默认走跨公司路由，route=False 时退回全局 TopK。"""
    if req.route:
        hits, info = R.route(
            req.question, top_k=req.top_k, per_company_k=req.per_company_k,
            mode=req.mode, company=req.company, year=req.year, rtype=req.rtype,
            rerank=req.rerank,
        )
        return hits, info
    hits = R.search(
        req.question, mode=req.mode, top_k=req.top_k, company=req.company,
        year=req.year, rtype=req.rtype, rerank=req.rerank,
    )
    return hits, {"route": "off", "companies": [], "year": req.year, "rtype": req.rtype}


class AskReq(BaseModel):
    question: str
    mode: str = "hybrid"
    top_k: int = 6
    company: str | None = None
    year: int | None = None
    rtype: str | None = None
    rerank: bool = True
    no_llm: bool = False
    route: bool = True
    per_company_k: int = 3


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "网页", "检索调试页.html"))


@app.get("/app")
def app_page():
    """问答 + 出处追溯页面（只读展示，不改检索与引用逻辑）。"""
    return FileResponse(os.path.join(ROOT, "网页", "问答追溯页.html"))


@app.get("/api/stats")
def stats():
    return R.stats()


@app.get("/api/chunk/{chunk_id}")
def chunk(chunk_id: int):
    """按文本块编号取回原文与完整出处，用于人工复核引用是否站得住。"""
    c = get_chunk(chunk_id)
    if not c:
        return {"error": f"未找到块 #{chunk_id}"}
    return c


@app.post("/api/search")
def search(req: AskReq):
    t = time.time()
    hits, info = _retrieve(req)
    return {"hits": hits, "elapsed": round(time.time() - t, 3), "mode": req.mode, "route": info}


@app.post("/api/ask")
def ask(req: AskReq):
    t0 = time.time()
    hits, info = _retrieve(req)
    ret_t = time.time() - t0
    answer, gen_t, citations = ("", 0.0, None)
    if not req.no_llm:
        answer, gen_t, citations = answer_with_citations(req.question, hits)
    return {
        "question": req.question,
        "mode": req.mode,
        "answer": answer,
        "hits": hits,
        "retrieve_sec": round(ret_t, 3),
        "generate_sec": round(gen_t, 3),
        "hit_companies": sorted({h["company"] for h in hits}),
        "route": info,
        "citations": citations,
        "model": ANSWER_MODEL if not req.no_llm else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
