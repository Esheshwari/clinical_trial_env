from pydantic import BaseModel
from typing import List, Dict, Any
import random

EPSILON = 1e-6


def clamp_score(score: float) -> float:
    if score <= 0:
        return EPSILON
    if score >= 1:
        return 1 - EPSILON
    return score


class Observation(BaseModel):
    trial_status: Dict[str, Any]
    patient_records: List[Dict[str, Any]]
    protocol: Dict[str, Any]
    deviations_found: List[Dict[str, Any]]


class Action(BaseModel):
    flag_deviations: List[int]
    corrective_actions: List[str]


class Reward(BaseModel):
    score: float
    details: str


class ClinicalTrialEnv:
    def __init__(self, task: str = 'easy'):
        self.task = task
        self.max_steps = 10
        self.current_step = 0
        self.state = {}
        self.ground_truth_deviations = []
        self.reset()

    def reset(self) -> Observation:
        self.current_step = 0
        self.state = self._generate_trial_data()
        return self._get_observation()

    def step(self, action: Action):
        self.current_step += 1
        reward = self._calculate_reward(action)

        # 🔥 CLAMP HERE (CRITICAL)
        reward.score = clamp_score(reward.score)

        done = self.current_step >= self.max_steps or self._is_done(action)
        info = {'step': self.current_step}

        return self._get_observation(), reward, done, info

    def _generate_trial_data(self):
        if self.task == 'easy':
            num_patients, num_deviations = 5, 2
        elif self.task == 'medium':
            num_patients, num_deviations = 20, 5
        else:
            num_patients, num_deviations = 100, 15

        patients = []
        deviations = []

        for i in range(num_patients):
            patient = {
                'id': i,
                'visits': self._generate_visits(),
                'labs': self._generate_labs(),
                'adverse_events': self._generate_aes()
            }

            if random.random() < num_deviations / num_patients:
                deviations.append({'patient_id': i})

            patients.append(patient)

        self.ground_truth_deviations = deviations

        return {
            'patients': patients,
            'protocol': {},
            'deviations': [],
            'trial_status': {'enrolled': num_patients}
        }

    def _generate_visits(self):
        return [{'visit_num': i} for i in range(4)]

    def _generate_labs(self):
        return {'CBC': random.uniform(0, 100)}

    def _generate_aes(self):
        return []

    def _get_observation(self):
        return Observation(
            trial_status=self.state['trial_status'],
            patient_records=self.state['patients'],
            protocol=self.state['protocol'],
            deviations_found=self.state['deviations']
        )

    def _calculate_reward(self, action: Action) -> Reward:
        flagged = set(action.flag_deviations)
        ground_truth = set(d['patient_id'] for d in self.ground_truth_deviations)

        true_pos = len(flagged & ground_truth)
        false_pos = len(flagged - ground_truth)
        false_neg = len(ground_truth - flagged)

        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        progress = f1 * 0.5 + (1 - self.current_step / self.max_steps) * 0.1
        penalty = -0.1 * len(action.corrective_actions)

        score = progress + penalty

        # 🔥 CLAMP AT SOURCE
        score = clamp_score(score)

        return Reward(score=score, details="ok")

    def _is_done(self, action: Action):
        return self.current_step >= self.max_steps


# ✅ FIXED GRADERS (NOW USE REWARD LIST)

def safe_avg(scores):
    if not scores:
        return EPSILON
    avg = sum(scores) / len(scores)
    return clamp_score(avg)


def grade_easy(env, rewards):
    return safe_avg(rewards)


def grade_medium(env, rewards):
    avg = safe_avg(rewards)
    return clamp_score(min(0.8, avg * 1.2))


def grade_hard(env, rewards):
    avg = safe_avg(rewards)
    return clamp_score(avg * 0.8)
