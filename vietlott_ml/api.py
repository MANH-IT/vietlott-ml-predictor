"""Web demo FastAPI: tạo bộ số gợi ý từ các mô hình đã huấn luyện."""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from .config import (
    DISCLAIMER,
    LOTTERIES,
    TEMPLATES_DIR,
    lottery_config,
    report_path,
)
from .data_collection import load_canonical
from .models import DISPLAY_NAMES, MODEL_NAMES, model_path
from .predict import generate_numbers

app = FastAPI(
    title="Vietlott ML Predictor",
    description=DISCLAIMER,
    version="1.0.0",
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class PredictRequest(BaseModel):
    lottery: str = Field(..., description="mega645 hoặc power655")
    model: str = Field(default="lstm")
    seed: int | None = None


def _latest_draw(lottery: str) -> dict:
    cfg = lottery_config(lottery)
    draws = load_canonical(lottery)
    if not draws:
        return {
            "lottery": lottery,
            "name": cfg["name"],
            "draw": None,
        }
    latest = draws[-1]
    return {
        "lottery": lottery,
        "name": cfg["name"],
        "draw": {
            "draw_id": latest.draw_id,
            "date": latest.date,
            "numbers": list(latest.numbers),
            "special": latest.special,
            "source": latest.source,
        },
        "total_draws": len(draws),
    }


def _available_models(lottery: str) -> list[dict]:
    out = []
    for name in MODEL_NAMES:
        out.append(
            {
                "key": name,
                "name": DISPLAY_NAMES.get(name, name),
                "available": model_path(lottery, name).exists(),
            }
        )
    out.append(
        {
            "key": "random",
            "name": DISPLAY_NAMES["random"],
            "available": True,
        }
    )
    return out


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "lotteries": [
                {
                    "key": cfg["key"],
                    "name": cfg["name"],
                    "schedule": cfg["schedule"],
                }
                for cfg in LOTTERIES.values()
            ],
            "disclaimer": DISCLAIMER,
        },
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/api/lotteries")
def lotteries() -> list[dict]:
    result = []
    for key, cfg in LOTTERIES.items():
        item = {
            "key": key,
            "name": cfg["name"],
            "n_features": cfg["n_features"],
            "pick": cfg["pick"],
            "has_special": cfg["has_special"],
            "schedule": cfg["schedule"],
            "models": _available_models(key),
        }
        item.update(_latest_draw(key))
        result.append(item)
    return result


@app.get("/api/latest/{lottery}")
def latest(lottery: str) -> dict:
    try:
        lottery_config(lottery)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _latest_draw(lottery)


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict:
    try:
        lottery_config(payload.lottery)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.model not in MODEL_NAMES + ["random"]:
        raise HTTPException(
            status_code=400,
            detail=f"Mô hình không hợp lệ. Chọn: {', '.join(MODEL_NAMES + ['random'])}",
        )
    try:
        return generate_numbers(payload.lottery, payload.model, seed=payload.seed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/metrics/{lottery}")
def metrics(lottery: str) -> dict:
    try:
        lottery_config(lottery)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = report_path(lottery)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Chưa có báo cáo cho {lottery}. Hãy chạy train trước.",
        )
    return json.loads(path.read_text(encoding="utf-8"))
