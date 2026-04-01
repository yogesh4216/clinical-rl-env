---
title: Clinical Workflow Optimization Environment
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# Clinical Workflow Optimization Environment (OpenEnv RL)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)
![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2%2B-purple.svg)

## 🔗 Submission Links

- **Hugging Face Space URL:** [https://huggingface.co/spaces/yogeshm2005/clinical-rl-env](https://huggingface.co/spaces/yogeshm2005/clinical-rl-env)
- **GitHub Repository URL:** [ADD_YOUR_GITHUB_LINK_HERE]

## 📌 Overview

The **Clinical Workflow Optimization Environment** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant Reinforcement Learning environment that simulates complex clinical workflows. An AI agent must navigate a surgical procedure simulation, balancing rapid procedural progression with critical patient safety under stochastic complications.

The environment implements the full OpenEnv specification — typed Pydantic models, `step()`/`reset()`/`state()` API, and an `openenv.yaml` manifest — and is validated via `openenv validate`.

## 🎯 Problem Statement

In modern clinical settings, continuous decision-making is necessary to ensure patient safety while successfully completing medical procedures. The RL agent must:
- **Progress Procedures:** Advance through surgical phases efficiently.
- **Handle Complications:** Swiftly respond to bleeding, infection, and equipment failure.
- **Maintain Patient Stability:** Keep vital signs within safe physiological thresholds.
- **Optimize Outcomes:** Maximize procedural success while minimizing time and penalties.

## 🏗️ Environment Architecture

### 📊 Observation Space (Typed: `ClinicalObservation`)

| Field          | Type              | Description                                |
|----------------|-------------------|--------------------------------------------|
| `phase`        | `str`             | Current procedural phase                   |
| `progress`     | `float [0.0–1.0]` | Procedure completion percentage           |
| `vitals.HR`    | `int [40–160]`    | Heart Rate in bpm                          |
| `vitals.BP`    | `int [60–200]`    | Blood Pressure (systolic) in mmHg          |
| `vitals.O2`    | `int [0–100]`     | Oxygen Saturation in %                     |
| `complication` | `str \| null`      | Active complication or null               |
| `time`         | `int`             | Elapsed procedural steps                   |
| `done`         | `bool`            | Whether the episode has terminated         |
| `reward`       | `float`           | Reward signal from the last action         |

### 🎮 Action Space (Typed: `ClinicalAction`)

| Action                 | Description                                                    |
|------------------------|----------------------------------------------------------------|
| `perform_step`         | Advance the procedure (risk of triggering complications)       |
| `handle_complication`  | Mitigate an active complication and stabilize vitals            |

### ⚠️ Complications & Dynamics

| Complication   | Effect                              |
|----------------|-------------------------------------|
| `bleeding`     | Rapid decrease in O2 Saturation     |
| `infection`    | Significant spike in Heart Rate     |
| `tool_failure` | Immediate score penalty             |

### 🚥 Tasks / Difficulty Levels

| Task     | Complication Probability | Description                                  |
|----------|--------------------------|----------------------------------------------|
| `easy`   | 10%                      | Low risk; vitals remain stable               |
| `medium` | 40%                      | Moderate risk; requires vigilant monitoring  |
| `hard`   | 70%                      | High unpredictability; frequent hurdles      |

### 🏅 Reward Function

| Signal       | Value     | Condition                                          |
|--------------|-----------|----------------------------------------------------|
| `+progress`  | +1.0      | Successful `perform_step` with no complication     |
| `+handling`  | +1.0–2.0  | Successfully handling an active complication        |
| `-time`      | −0.1      | Applied every step                                 |
| `-ignore`    | −4.0      | Ignoring an active complication                    |
| `-failure`   | −10.0     | Vitals exceed safety bounds                        |
| `+success`   | +5.0      | Procedure completed (progress ≥ 1.0)              |

### 🛑 Failure Conditions

- **O2 Saturation:** `< 50%` → immediate episode termination
- **Heart Rate:** `> 140 bpm` → immediate episode termination

## 🔌 API Reference

### OpenEnv Standard Endpoints

| Endpoint   | Method | Description                                                     |
|------------|--------|-----------------------------------------------------------------|
| `/reset`   | `POST` | Reset the environment. Body: `{"difficulty": "easy"}`. Returns `ResetResponse`. |
| `/step`    | `POST` | Execute an action. Body: `{"action": {"action": "perform_step"}}`. Returns `StepResponse`. |
| `/state`   | `GET`  | Returns current episode metadata (`State`).                     |
| `/health`  | `GET`  | Health check.                                                   |
| `/schema`  | `GET`  | Returns JSON schemas for Action, Observation, and State.       |

### Custom Endpoints

| Endpoint    | Method | Description                                                    |
|-------------|--------|----------------------------------------------------------------|
| `/tasks`    | `GET`  | List available tasks with difficulty descriptions.             |
| `/grader`   | `GET`  | Run a graded episode. Query: `?task=easy`. Returns `{score}`.  |
| `/baseline` | `GET`  | Run rule-based & random baselines across all tasks.            |

## 📈 Baseline Results

| Task   | Rule-Based | Random |
|--------|------------|--------|
| Easy   | ~0.88      | ~0.52  |
| Medium | ~0.70      | ~0.35  |
| Hard   | ~0.56      | ~0.20  |

## 💻 Tech Stack

- **OpenEnv Core** — Environment base class, `create_app()` server factory, typed models
- **FastAPI** — High-performance API layer
- **Pydantic** — Typed Action, Observation, and State models
- **OpenAI SDK** — LLM inference via OpenAI-compatible API
- **Docker** — Containerized deployment to Hugging Face Spaces

## 📂 Project Structure

```
clinical-rl-env/
├── models.py                          # Typed Pydantic models (Action, Observation, State)
├── inference.py                       # OpenEnv competition inference script
├── openenv.yaml                       # OpenEnv environment manifest
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container for HF Spaces deployment
├── .dockerignore
├── README.md
├── server/
│   ├── __init__.py
│   ├── app.py                         # create_app() — OpenEnv FastAPI server
│   └── clinical_environment.py        # Environment subclass (core RL logic)
└── env/                               # (legacy — kept for reference)
    ├── __init__.py
    ├── environment.py
    ├── grader.py
    └── logger.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### Installation

```bash
git clone https://github.com/yourusername/clinical-rl-env.git
cd clinical-rl-env
pip install -r requirements.txt
```

### Running the API Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

The interactive Swagger docs will be available at `http://localhost:7860/docs`.

### Running the Inference Script

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export HF_TOKEN=your_hf_token_here
export ENV_URL=http://localhost:7860

python inference.py
```

### Docker Build & Run

```bash
docker build -t clinical-rl-env .
docker run -p 7860:7860 clinical-rl-env
```

## 🧪 How to Test (For Judges & Reviewers)

🔗 Live Demo: https://yogeshm2005-clinical-rl-env.hf.space/docs

### Step 1: Reset Environment
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "medium"}'
```

### Step 2: Take Actions
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": "perform_step"}}'
```

### Step 3: Check State
```bash
curl http://localhost:7860/state
```

### Step 4: Grade a Task
```bash
curl "http://localhost:7860/grader?task=easy"
curl "http://localhost:7860/grader?task=medium"
curl "http://localhost:7860/grader?task=hard"
```

### Step 5: Run Baselines
```bash
curl http://localhost:7860/baseline
```

---

## 🎯 Expected Behavior

- Rule-based agent consistently scores higher than random
- Performance decreases as difficulty increases (easy → hard)
- Proper handling of complications significantly improves score
- All grader scores fall within the 0.0–1.0 range
- `openenv validate` passes on this project

## 👨‍💻 Author

**Yogesh M**
*AI Researcher & Engineer*

---
*If you find this project useful for your research, please consider giving it a star!*
