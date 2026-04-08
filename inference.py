import os
import json
from openai import OpenAI
from environment import ClinicalTrialEnv, Action

# Use environment variables injected by the validator
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

# Small epsilon to keep scores strictly between (0,1)
EPSILON = 1e-6


def clamp_score(score: float) -> float:
    """Ensure score is strictly within (0,1)"""
    return max(EPSILON, min(1 - EPSILON, score))


def run_inference(task: str):
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return EPSILON  # ✅ never return 0.0

    env = ClinicalTrialEnv(task=task)
    obs = env.reset()

    total_reward = 0.0
    step_count = 0

    print(f"[START] Task: {task}")

    while True:
        step_count += 1

        prompt = f"""
You are a Clinical Research Associate monitoring a clinical trial.

Observation:
{json.dumps(obs.dict(), indent=2)}

Identify:
1. flag_deviations: list of patient indices with issues
2. corrective_actions: list like ["correct"] or ["escalate"]

Respond ONLY in JSON:
{{"flag_deviations": [1,2], "corrective_actions": ["correct"]}}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            raw_output = response.choices[0].message.content.strip()

            try:
                action_data = json.loads(raw_output)
                action = Action(**action_data)
            except Exception as parse_error:
                print(f"JSON parse failed: {parse_error}")
                action = Action(flag_deviations=[], corrective_actions=[])

        except Exception as e:
            print(f"LLM call failed: {e}")
            action = Action(flag_deviations=[], corrective_actions=[])

        obs, reward, done, info = env.step(action)

        # ✅ Clamp reward BEFORE using it
        safe_reward = clamp_score(reward.score)

        total_reward += safe_reward

        print(f"[STEP] {step_count} | Raw: {reward.score:.4f} | Safe: {safe_reward:.4f}")

        if done:
            break

    # Avoid division by zero
    if step_count == 0:
        return EPSILON

    avg_reward = total_reward / step_count

    # ✅ FINAL CLAMP (MOST IMPORTANT)
    avg_reward = clamp_score(avg_reward)

    print(f"[END] Task: {task}, Avg Reward: {avg_reward:.4f}")

    return avg_reward


if __name__ == "__main__":
    tasks = ['easy', 'medium', 'hard']
    scores = {}

    for task in tasks:
        try:
            score = run_inference(task)
        except Exception as e:
            print(f"Task {task} crashed: {e}")
            score = EPSILON  # ✅ safe fallback

        # ✅ Ensure final score is valid
        scores[task] = clamp_score(score)

    print(f"Final Scores: {scores}")
