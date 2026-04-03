from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from environment import ClinicalTrialEnv, Action, Observation
import uvicorn

app = FastAPI(title="Clinical Trial Deviation Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(task: str = "easy"):
    global env
    env = ClinicalTrialEnv(task=task)
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: dict):
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    try:
        act = Action(**action)
        obs, reward, done, info = env.step(act)
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/state")
def get_state():
    if env is None:
        return {"error": "Environment not initialized."}
    return env.state()

@app.get("/")
def root():
    return {"message": "Clinical Trial OpenEnv API"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, log_level="info")