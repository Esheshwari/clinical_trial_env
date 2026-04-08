import os
import json
from openai import OpenAI
from environment import ClinicalTrialEnv, Action, grade_easy, grade_medium, grade_hard

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

EPSILON = 1e-6


def clamp(score):
    """Force score strictly into (0,1)"""
    if score <= 0:
        return EPSILON
    if score >= 1:
        return 1 - EPSILON
    return score


def run_inference(task: str):
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        print(f"Client init failed: {e}")
        return EPSILON

    env = ClinicalTrialEnv(task=task)
    obs = env.reset()

    actions = []  # Collect actions for grading
    step_count = 0

    print(f"[START] {task}")

    while True:
        step_count += 1

        prompt = f"""
You are a Clinical Research Associate.

Observation:
{json.dumps(obs.dict(), indent=2)}

Return ONLY JSON:
{{"flag_deviations": [], "corrective_actions": []}}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            raw = response.choices[0].message.content.strip()

            try:
                data = json.loads(raw)
                action = Action(**data)
            except:
                action = Action(flag_deviations=[], corrective_actions=[])

        except Exception as e:
            print(f"LLM failed: {e}")
            action = Action(flag_deviations=[], corrective_actions=[])

        obs, reward, done, info = env.step(action)

        # Collect the action for grading
        actions.append(action)

        print(f"[STEP] {step_count} | Action: {action.dict()}")

        if done:
            break

    # Use the appropriate grading function based on task
    if task == "easy":
        score = grade_easy(env, actions)
    elif task == "medium":
        score = grade_medium(env, actions)
    elif task == "hard":
        score = grade_hard(env, actions)
    else:
        score = EPSILON

    # Ensure score is strictly between 0 and 1
    score = clamp(score)

    print(f"[END] {task} → {score}")

    return score


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    scores = {}

    for task in tasks:
        try:
            score = run_inference(task)
        except Exception as e:
            print(f"{task} crashed: {e}")
            score = EPSILON

        scores[task] = score

    print("Final Scores:", scores)
