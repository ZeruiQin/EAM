"""Visual feature extraction service for EAM."""

from __future__ import annotations

import os
from io import BytesIO
from threading import Lock
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models


DEFAULT_MODEL = os.environ.get("FEATURE_MODEL", "resnet50")
DEFAULT_DEVICE = os.environ.get("FEATURE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="EAM Feature Service", version="1.0.0")
_model_lock = Lock()
_model_name = ""
_device = torch.device(DEFAULT_DEVICE)
_model: torch.nn.Module | None = None
_preprocess = None
_dimension = 2048


def _build_model(model_name: str) -> tuple[torch.nn.Module, object, int]:
    normalized = model_name.lower()
    if normalized != "resnet50":
        raise ValueError(f"Unsupported model_name={model_name!r}; only 'resnet50' is supported.")

    weights = models.ResNet50_Weights.DEFAULT
    backbone = models.resnet50(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(_device)
    return feature_extractor, weights.transforms(), 2048


def _ensure_model(model_name: str | None = None) -> None:
    global _model, _model_name, _preprocess, _dimension
    requested = (model_name or DEFAULT_MODEL).lower()
    if _model is not None and _model_name == requested:
        return
    with _model_lock:
        if _model is not None and _model_name == requested:
            return
        try:
            _model, _preprocess, _dimension = _build_model(requested)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _model_name = requested


def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc


def _extract(images: list[Image.Image], model_name: str) -> list[list[float]]:
    _ensure_model(model_name)
    assert _model is not None
    assert _preprocess is not None

    tensors = [_preprocess(image) for image in images]
    batch = torch.stack(tensors, dim=0).to(_device)
    with torch.inference_mode():
        features = _model(batch).flatten(1)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.detach().cpu().numpy().astype(np.float32).tolist()


@app.on_event("startup")
def _startup() -> None:
    _ensure_model(DEFAULT_MODEL)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_name": _model_name or DEFAULT_MODEL,
        "dimension": _dimension,
        "device": str(_device),
    }


@app.post("/set_model")
def set_model(payload: dict) -> dict:
    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing required field: model_name")
    _ensure_model(str(model_name))
    return {
        "status": "ok",
        "model_name": _model_name,
        "dimension": _dimension,
        "device": str(_device),
    }


@app.post("/extract_single")
async def extract_single(
    file: Annotated[UploadFile, File()],
    model_name: str = DEFAULT_MODEL,
) -> dict:
    image = _read_image(await file.read())
    features = _extract([image], model_name)
    return {"model_name": _model_name, "dimension": _dimension, "features": features}


@app.post("/extract_batch")
async def extract_batch(
    files: Annotated[list[UploadFile], File()],
    model_name: str = DEFAULT_MODEL,
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    images = [_read_image(await file.read()) for file in files]
    features = _extract(images, model_name)
    return {"model_name": _model_name, "dimension": _dimension, "features": features}
