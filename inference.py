import os
import json
from openai import OpenAI
from environment import ClinicalTrialEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=API_BASE_URL)

def run_inference(task: str):
    env = ClinicalTrialEnv(task=task)
    obs = env.reset()
    actions = []
    total_reward = 0
    step_count = 0

    print(f"[START] Task: {task}")

    while True:
        step_count += 1
        # Prepare prompt for LLM
        prompt = f"""
You are a Clinical Research Associate monitoring a clinical trial. Review the following observation and decide on actions.

Observation:
{json.dumps(obs.dict(), indent=2)}

Decide which patient indices have deviations (flag_deviations: list of ints) and corrective actions (corrective_actions: list of strings like 'correct', 'escalate').

Respond with JSON: {{"flag_deviations": [1,2], "corrective_actions": ["correct"]}}
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        action_json = response.choices[0].message.content.strip()
        try:
            action_data = json.loads(action_json)
            action = Action(**action_data)
        except:
            action = Action(flag_deviations=[], corrective_actions=[])

        obs, reward, done, info = env.step(action)
        actions.append(reward.score)
        total_reward += reward.score

        print(f"[STEP] Step: {step_count}, Reward: {reward.score:.2f}, Done: {done}")

        if done:
            break

    avg_reward = total_reward / step_count if step_count > 0 else 0
    print(f"[END] Task: {task}, Average Reward: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    tasks = ['easy', 'medium', 'hard']
    scores = {}
    for task in tasks:
        score = run_inference(task)
        scores[task] = score
    print(f"Final Scores: {scores}")