import os
import json
from openai import OpenAI
from environment import ClinicalTrialEnv, Action, grade_easy, grade_medium, grade_hard

EPSILON = 1e-6


def clamp(score):
    if score <= 0:
        return EPSILON
    if score >= 1:
        return 1 - EPSILON
    return score


API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")


def run_inference(task: str):
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except:
        return EPSILON

    env = ClinicalTrialEnv(task=task)
    obs = env.reset()

    rewards = []
    step_count = 0

    while True:
        step_count += 1

        prompt = f"""
Observation:
{json.dumps(obs.dict())}

Return JSON:
{{"flag_deviations": [], "corrective_actions": []}}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            data = json.loads(response.choices[0].message.content.strip())
            action = Action(**data)

        except:
            action = Action(flag_deviations=[], corrective_actions=[])

        obs, reward, done, _ = env.step(action)

        # 🔥 USE REWARD SCORES (NOT ACTIONS)
        rewards.append(clamp(reward.score))

        if done:
            break

    if task == "easy":
        score = grade_easy(env, rewards)
    elif task == "medium":
        score = grade_medium(env, rewards)
    else:
        score = grade_hard(env, rewards)

    return clamp(score)


if __name__ == "__main__":
    scores = {}

    for task in ["easy", "medium", "hard"]:
        try:
            scores[task] = run_inference(task)
        except:
            scores[task] = EPSILON

    print("Final Scores:", scores)
