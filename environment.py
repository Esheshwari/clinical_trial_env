from pydantic import BaseModel
from typing import List, Dict, Any
import random
import json

class Observation(BaseModel):
    trial_status: Dict[str, Any]
    patient_records: List[Dict[str, Any]]
    protocol: Dict[str, Any]
    deviations_found: List[Dict[str, Any]]

class Action(BaseModel):
    flag_deviations: List[int]  # indices of patient records with deviations
    corrective_actions: List[str]  # actions like 'correct', 'escalate', 'halt'

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

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = self._calculate_reward(action)
        done = self.current_step >= self.max_steps or self._is_done(action)
        info = {'step': self.current_step}
        return self._get_observation(), reward, done, info

    def state(self) -> Dict[str, Any]:
        return self.state

    def _generate_trial_data(self) -> Dict[str, Any]:
        if self.task == 'easy':
            num_patients = 5
            num_deviations = 2
            complexity = 'low'
        elif self.task == 'medium':
            num_patients = 20
            num_deviations = 5
            complexity = 'medium'
        elif self.task == 'hard':
            num_patients = 100
            num_deviations = 15
            complexity = 'high'
        else:
            raise ValueError("Invalid task")

        patients = []
        deviations = []
        for i in range(num_patients):
            patient = {
                'id': i,
                'visits': self._generate_visits(complexity),
                'labs': self._generate_labs(complexity),
                'adverse_events': self._generate_aes(complexity)
            }
            patients.append(patient)
            # randomly add deviations
            if random.random() < num_deviations / num_patients:
                dev = self._generate_deviation(complexity)
                deviations.append({'patient_id': i, 'type': dev['type'], 'description': dev['desc']})
                # modify patient data to reflect deviation
                self._apply_deviation_to_patient(patient, dev)

        self.ground_truth_deviations = deviations
        protocol = {
            'dosing_schedule': 'Daily 10mg',
            'visit_schedule': 'Weekly',
            'lab_requirements': 'CBC, LFTs',
            'ae_reporting': 'Within 24h'
        }
        return {
            'patients': patients,
            'protocol': protocol,
            'deviations': [],  # found by agent
            'trial_status': {'enrolled': num_patients, 'completed_visits': 0}
        }

    def _generate_visits(self, complexity):
        num_visits = 4 if complexity == 'low' else 8 if complexity == 'medium' else 12
        visits = []
        for v in range(num_visits):
            visit = {'visit_num': v+1, 'date': f'Day {7*(v+1)}', 'dosing': '10mg' if random.random() > 0.1 else '5mg'}
            visits.append(visit)
        return visits

    def _generate_labs(self, complexity):
        labs = ['CBC', 'LFTs', 'Creatinine']
        return {lab: random.uniform(0, 100) for lab in labs}

    def _generate_aes(self, complexity):
        aes = []
        if random.random() < 0.3:
            aes.append({'type': 'Nausea', 'severity': 'Mild', 'reported': True})
        return aes

    def _generate_deviation(self, complexity):
        types = ['dosing_error', 'missed_visit', 'late_lab', 'unreported_ae']
        if complexity == 'high':
            types.append('data_fabrication')
        dev_type = random.choice(types)
        desc = {
            'dosing_error': 'Incorrect dose administered',
            'missed_visit': 'Patient missed scheduled visit',
            'late_lab': 'Lab results submitted late',
            'unreported_ae': 'Adverse event not reported',
            'data_fabrication': 'Lab values altered'
        }[dev_type]
        return {'type': dev_type, 'desc': desc}

    def _apply_deviation_to_patient(self, patient, dev):
        if dev['type'] == 'dosing_error':
            patient['visits'][0]['dosing'] = '15mg'
        elif dev['type'] == 'missed_visit':
            patient['visits'].pop(1)
        elif dev['type'] == 'late_lab':
            patient['labs']['CBC'] = 999  # invalid
        elif dev['type'] == 'unreported_ae':
            patient['adverse_events'][0]['reported'] = False
        elif dev['type'] == 'data_fabrication':
            patient['labs']['LFTs'] = 0

    def _get_observation(self) -> Observation:
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
        # partial reward for progress
        progress_reward = f1 * 0.5 + (1 - self.current_step / self.max_steps) * 0.1
        # penalty for wrong actions
        penalty = -0.1 * len(action.corrective_actions) if len(action.corrective_actions) > len(flagged) else 0
        total_score = progress_reward + penalty
        details = f'F1: {f1:.2f}, Progress: {progress_reward:.2f}, Penalty: {penalty:.2f}'
        return Reward(score=total_score, details=details)

    def _is_done(self, action: Action) -> bool:
        return len(action.flag_deviations) > 0 and random.random() < 0.5  # simulate completion

# Graders for tasks
def grade_easy(env: ClinicalTrialEnv, actions: List[Action]) -> float:
    # Simple: score based on final reward
    final_reward = sum(a.score for a in actions) / len(actions) if actions else 0
    return min(1.0, max(0.0, final_reward))

def grade_medium(env: ClinicalTrialEnv, actions: List[Action]) -> float:
    # Medium: require higher accuracy
    final_reward = sum(a.score for a in actions) / len(actions) if actions else 0
    return min(1.0, max(0.0, final_reward * 1.2))

def grade_hard(env: ClinicalTrialEnv, actions: List[Action]) -> float:
    # Hard: complex, lower score
    final_reward = sum(a.score for a in actions) / len(actions) if actions else 0
    return min(1.0, max(0.0, final_reward * 0.8))