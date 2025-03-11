import numpy as np

def reward_len_penalty_coef(difficulty, beta):
    return beta * np.exp(-0.5*difficulty)

def compute_score(solution_str, ground_truth, extra_info):
    pass
