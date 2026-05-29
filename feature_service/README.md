# EAM Feature Service

This service provides visual embeddings for EAM knowledge-graph construction
and online fallback retrieval. It exposes the API expected by
`task_explorer.utils.img_tool.extract_features()` and
`android_world.utils.graph_utils.extract_features()`.

## API

```text
GET  /health
POST /set_model
POST /extract_single?model_name=resnet50
POST /extract_batch?model_name=resnet50
```

`/extract_single` expects multipart field `file`.
`/extract_batch` expects repeated multipart field `files`.

Responses use this shape:

```json
{"model_name": "resnet50", "dimension": 2048, "features": [[0.1, 0.2]]}
```

Single-image extraction still returns a list containing one vector so existing
KG code can use `features["features"][0]`.

## Docker

Build and run:

```bash
docker build -t eam-feature-service feature_service
docker run --rm -p 8001:8001 eam-feature-service
```

Then configure:

```bash
FEATURE_URI=http://127.0.0.1:8001
```

The default Dockerfile installs CPU PyTorch wheels for portability. If you want
GPU acceleration, replace the base image with a CUDA PyTorch image and install
matching `torchvision`.

