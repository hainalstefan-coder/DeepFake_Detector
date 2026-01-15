# Deepfake Detector

A multi-agent deepfake detection system for static face images, inspired by the **Mandrake** fault-tolerant decentralized design.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange)

## Features

- **Multi-Agent Architecture**: 5 independent detection agents running concurrently
- **Real-Time Updates**: WebSocket streaming of agent progress and results
- **Fault Tolerance**: Inspired by Mandrake patterns (Remind, Checkpoint, Continue)
- **Modern UI**: Vercel-inspired dark theme with live agent status cards
- **GPU Accelerated**: Optimized for RTX 3090 Ti with CPU fallback

## Mandrake Concepts Applied

| Concept | Implementation |
|---------|---------------|
| **Agents** | Independent detection tasks (CNN, ViT, Frequency, Embedding, FaceXray) |
| **Asynchronous Messaging** | Message bus with TaskMessage/ResultMessage/ErrorMessage |
| **Remind** | Timeout (12s) + configurable retry mechanism |
| **Checkpoint** | Immediate persistence of agent outputs |
| **Continue** | Quorum-based finalization (3 of 5 agents) |

## Detection Agents

1. **CNNClassifierAgent** - EfficientNet-based deep learning classifier
2. **ViTClassifierAgent** - Vision Transformer with attention analysis
3. **FrequencyAgent** - FFT/DCT frequency spectrum analysis for GAN artifacts
4. **EmbeddingAnomalyAgent** - Face embedding anomaly detection (ArcFace)
5. **FaceXrayLikeAgent** - Blend boundary and edge inconsistency detection

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, CPU fallback available)
- 8GB+ RAM

### Installation

```bash
# Clone or navigate to project
cd deepfake-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
# Start the server
uvicorn app.main:app --reload

# Or with specific host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the UI

Open your browser to: **http://localhost:8000**

## API Endpoints

### Upload Image for Analysis

```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/api/analyze
```

Response:
```json
{
  "job_id": "abc12345",
  "status": "processing",
  "message": "Analysis started with 5 agents"
}
```

### Get Job Result

```bash
curl http://localhost:8000/api/result/abc12345
```

### WebSocket (Real-Time Updates)

Connect to `/ws/{job_id}` to receive live updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/abc12345');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.event_type, data.data);
};
```

Event types:
- `job_started` - Analysis began
- `agent_started` - Agent processing started
- `agent_completed` - Agent finished with score
- `agent_error` - Agent failed
- `agent_retry` - Agent retrying (Remind pattern)
- `job_completed` - Final verdict ready

### Health Check

```bash
curl http://localhost:8000/api/health
```

## Project Structure

```
deepfake-detector/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration loader
│   ├── models/
│   │   └── schemas.py       # Pydantic models (TaskMessage, ResultMessage, etc.)
│   ├── core/
│   │   ├── message_bus.py   # Async pub/sub message bus
│   │   ├── agent_base.py    # Base agent class
│   │   └── orchestrator.py  # Mandrake-inspired orchestrator
│   ├── agents/
│   │   ├── registry.py      # Agent auto-discovery
│   │   ├── cnn_classifier.py
│   │   ├── vit_classifier.py
│   │   ├── frequency.py
│   │   ├── embedding_anomaly.py
│   │   └── face_xray.py
│   ├── preprocessing/
│   │   └── face_pipeline.py # Face detection and cropping
│   ├── storage/
│   │   └── job_store.py     # SQLite/in-memory storage
│   ├── api/
│   │   ├── routes.py        # REST endpoints
│   │   └── websocket.py     # WebSocket handler
│   └── static/
│       ├── index.html       # Frontend UI
│       ├── styles.css       # Vercel-inspired styling
│       └── app.js           # Frontend logic
├── config/
│   └── config.yaml          # Configuration
├── data/
│   └── jobs/                # Per-job storage
├── tests/
│   ├── test_message_bus.py
│   └── test_aggregation.py
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/config.yaml`:

```yaml
agents:
  timeout_seconds: 12    # Agent timeout
  max_retries: 1         # Retry attempts (Remind)
  quorum: 3              # Min agents for decision (Continue)
  
  weights:               # Aggregation weights
    CNNClassifierAgent: 1.0
    ViTClassifierAgent: 1.0
    FrequencyAgent: 0.8
    EmbeddingAnomalyAgent: 0.9
    FaceXrayLikeAgent: 0.85

detection:
  fake_threshold: 0.6    # Score >= threshold = FAKE
```

## Adding New Agents

1. Create a new file in `app/agents/`:

```python
from app.core.agent_base import BaseAgent
from app.models.schemas import ResultMessage, TaskMessage

class MyNewAgent(BaseAgent):
    name = "MyNewAgent"
    description = "My custom detection method"
    
    async def initialize(self) -> None:
        # Load models here
        self._initialized = True
    
    async def process(self, task: TaskMessage) -> ResultMessage:
        # Your detection logic
        score = 0.5  # 0=real, 1=fake
        
        return self.create_result(
            task,
            score=score,
            explanation="My detection result"
        )
```

2. Register in `app/agents/registry.py`:

```python
from app.agents.my_new_agent import MyNewAgent
_agent_registry[MyNewAgent.name] = MyNewAgent
```

3. Add weight in `config/config.yaml`:

```yaml
agents:
  weights:
    MyNewAgent: 1.0
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

## Sample Output

```json
{
  "job_id": "abc12345",
  "verdict": "FAKE",
  "confidence": 0.72,
  "final_score": 0.68,
  "quorum_reached": true,
  "explanation": "Analysis indicates FAKE (68% fake probability). Supporting evidence: ViTClassifierAgent (82%): Attention patterns indicate synthetic generation; CNNClassifierAgent (75%): CNN detected manipulation artifacts. Counter-evidence: EmbeddingAnomalyAgent (45%): Embedding within normal range.",
  "agent_results": [
    {"agent_name": "CNNClassifierAgent", "score": 0.75, "latency_ms": 150},
    {"agent_name": "ViTClassifierAgent", "score": 0.82, "latency_ms": 200},
    {"agent_name": "FrequencyAgent", "score": 0.65, "latency_ms": 50},
    {"agent_name": "EmbeddingAnomalyAgent", "score": 0.45, "latency_ms": 100},
    {"agent_name": "FaceXrayLikeAgent", "score": 0.70, "latency_ms": 80}
  ]
}
```

## Future Extensions

The architecture is designed for easy extension to other modalities:

- **Video**: Add frame extraction, temporal analysis agents
- **Audio**: Add voice clone detection agents
- **Multi-modal**: Combine analysis across modalities

See `MediaTask.modality` field in schemas for preparation.

## License

MIT License

## Acknowledgments

- Inspired by the [Mandrake paper](https://doi.org/10.1007/s10458-021-09540-8) on fault-tolerant multiagent systems
- Face detection powered by [MTCNN](https://github.com/ipazc/mtcnn)
- Models from [timm](https://github.com/huggingface/pytorch-image-models)
