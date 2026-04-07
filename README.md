---
title: Oncall Engineer Env
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
---

# OnCall Engineer Environment

An OpenEnv RL environment where an AI agent acts as an on-call engineer,
diagnosing and resolving production software incidents.

## Tasks

| Task              | Difficulty | Scenario                                     | Max Steps |
| ----------------- | ---------- | -------------------------------------------- | --------- |
| `easy_crash`      | Easy       | Payment service crashed after bad deployment | 15        |
| `medium_cascade`  | Medium     | Database connection exhaustion cascade       | 20        |
| `hard_corruption` | Hard       | Silent data corruption, no alerts            | 25        |

## Baseline Scores

| Task              | Random Agent | Baseline LLM | Our Agent's best score |
| ----------------- | ------------ | ------------ | ---------------------- |
| `easy_crash`      | 0.05         | 0.72         | **0.93**               |
| `medium_cascade`  | 0.04         | 0.51         | **0.92**               |
| `hard_corruption` | 0.02         | 0.38         | **0.83**               |

## Setup

```bash
git clone https://github.com/GauravSethi22/oncall-engineer-env.git
cd oncall-engineer-env
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

| Variable       | Description                 |
| -------------- | --------------------------- |
| `HF_TOKEN`     | Your Hugging Face / API key |
| `API_BASE_URL` | LLM API endpoint            |
| `MODEL_NAME`   | Model identifier            |
| `ENV_URL`      | Environment URL             |

## Running Inference

```bash
export HF_TOKEN=your_api_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export ENV_URL=https://Gaurav206-oncall-engineer-env.hf.space
python inference.py
```

## Action Space

| Action          | Required Fields                           |
| --------------- | ----------------------------------------- |
| `query_logs`    | `target_service`, `log_filter` (optional) |
| `check_metrics` | `target_service`, `metric_name`           |
| `check_deps`    | `target_service`                          |
| `apply_fix`     | `target_service`, `fix_type`              |
| `escalate`      | `team`                                    |
| `write_summary` | `summary_text`                            |

## Observation Space

| Field                | Description                     |
| -------------------- | ------------------------------- |
| `alerts`             | Active firing alerts            |
| `services`           | Health status of all 5 services |
| `last_action_result` | What the last action returned   |
| `elapsed_minutes`    | Time since incident started     |
| `unlocked_hints`     | Hints revealed progressively    |

## API Endpoints

| Endpoint  | Method    | Description           |
| --------- | --------- | --------------------- |
| `/health` | GET       | Health check          |
| `/reset`  | POST      | Start new episode     |
| `/step`   | POST      | Execute action        |
| `/state`  | GET       | Current episode state |
| `/ws`     | WebSocket | Persistent connection |

## Project Structure

```
oncall-engineer-env/
├── models.py          ← typed Action, Observation, State
├── client.py          ← what the agent imports
├── inference.py       ← baseline agent script
├── openenv.yaml       ← hackathon manifest
├── pyproject.toml     ← dependencies
├── README.md
├── oncall_env/
│   └── __init__.py
└── server/
    ├── app.py         ← FastAPI server
    ├── environment.py ← main game logic
    ├── simulator.py   ← fake production system
    ├── tasks.py       ← 3 incident scenarios
    ├── graders.py     ← scoring functions
    └── Dockerfile
```
