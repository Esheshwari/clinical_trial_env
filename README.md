# Clinical Trial Protocol Deviation Detector

An OpenEnv environment where AI agents simulate the role of Clinical Research Associates (CRAs) in monitoring clinical trials for protocol deviations, safety signals, and compliance issues.

## Environment Description

This environment simulates the real-world task of pharmaceutical and biotech companies' CRAs who review patient data, lab results, and protocol documents to ensure trial integrity. Agents must identify deviations (e.g., dosing errors, missed visits, unreported adverse events), flag safety concerns, and recommend corrective actions. This fills a critical gap in AI evaluation for regulated, document-heavy domains.

### Motivation
Clinical trials are the backbone of drug development, costing billions annually. Undetected deviations lead to trial failures, regulatory penalties, and patient safety risks. This environment trains AI to assist CRAs, potentially reducing errors and accelerating trials.

## Action and Observation Spaces

### Observation Space
- `trial_status`: Dict with enrollment numbers, completion status.
- `patient_records`: List of dicts, each containing patient ID, visits (dosing, dates), labs (CBC, LFTs), adverse events.
- `protocol`: Dict with dosing schedule, visit requirements, lab standards, AE reporting rules.
- `deviations_found`: List of flagged deviations by the agent.

### Action Space
- `flag_deviations`: List of patient indices (ints) suspected of deviations.
- `corrective_actions`: List of strings (e.g., "correct", "escalate", "halt") corresponding to flagged deviations.

### Reward Function
Rewards partial progress: F1 score on deviation detection (precision/recall), timeliness bonuses, penalties for false positives/negatives or excessive actions. Range: 0.0 (poor) to 1.0 (perfect).

## Tasks

1. **Easy**: 5 patients, 2 clear deviations (e.g., obvious dosing errors). Focus on basic detection.
2. **Medium**: 20 patients, 5 subtle deviations (e.g., timing variances, unreported AEs). Requires pattern recognition.
3. **Hard**: 100 patients, 15 complex deviations (e.g., data fabrication, multi-site issues). Involves ethical and regulatory judgment.

Each task has a grader scoring 0.0–1.0 based on accuracy, impact, and compliance.

## Setup Instructions

1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `uvicorn app:app --reload`
4. For Docker: `docker build -t clinical-trial-env . && docker run -p 8000:8000 clinical-trial-env`

### API Endpoints
- `POST /reset?task=easy`: Initialize environment, returns initial observation.
- `POST /step`: Send action dict, returns observation, reward, done, info.
- `GET /state`: Get current state.

## Baseline Scores
Run `python inference.py` with OpenAI API key set. Expected scores (approximate):
- Easy: 0.7–0.9
- Medium: 0.5–0.7
- Hard: 0.3–0.5

## Deployment
Deployed on Hugging Face Spaces at [your-space-url]. Containerized with Dockerfile, responds to reset() and step() calls.