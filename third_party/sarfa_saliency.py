"""
Vendorized from `sarfa_saliency.py`.

Source: https://github.com/nikaashpuri/sarfa-saliency (commit `master` on 2026-07-27)
Paper: Puri et al., "Explain Your Move: Understanding Agent Actions Using
       Specific and Relevant Feature Attribution" (SARFA), ICLR 2020.

Key notes & modifications for our usage:
  1. `computeSaliencyUsingSarfa` returns a 6-tuple, not a scalar: 
     (answer, dP, K, QmaxAnswer, action_gap_before, action_gap_after).
  2. It applies softmax internally. Pass raw logits, NOT probabilities or scaled log-probs.
  3. PATCH: Commented out a `print()` inside `cross_entropy()` to prevent massive 
     stdout spam. This is the ONLY modification from the original code.
  4. Saliency is 0 by design when the perturbation *increases* the probability 
     of the chosen action P(a-hat). This asymmetry is intentional. Final range: [0, 1].
"""
import math
import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon


def your_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def cross_entropy(dictP, dictQ, original_action):
    """
    This function calculates normalized cross entropy (KL divergence) of Q-values of state Q wrt state P.
    Input:
        dictP: Q-value dictionary of perturbed state
        dictQ: Q-value dictionary of original state
    Output:
        K: normalized cross entropy
    """
    Dpq = 0.
    Q_p = []  # values of moves in dictP^dictQ wrt P
    Q_q = []  # values of moves in dictP^dictQ wrt Q
    for move in dictP:
        if move == original_action:
            # print('skipping original action for KL-Divergence')  # PATCHED: Silenced to avoid stdout spam.
            continue
        if move in dictQ:
            Q_p.append(dictP[move])
            Q_q.append(dictQ[move])
    # converting Q-values into probability distribution
    Q_p = your_softmax(np.asarray(Q_p))
    Q_q = your_softmax(np.asarray(Q_q))
    KL = entropy(Q_q, Q_p)
    # KL = wasserstein_distance(Q_q, Q_p)
    # return (KL)/(KL + 1.)
    return 1. / (KL + 1.)


def computeSaliencyUsingSarfa(original_action, dict_q_vals_before_perturbation, dict_q_vals_after_perturbation):

    answer = 0

    # probability of original move in perturbed state
    q_value_action_perturbed_state = dict_q_vals_after_perturbation[original_action]
    q_value_action_original_state = dict_q_vals_before_perturbation[original_action]

    q_values_after_perturbation = np.asarray(list(dict_q_vals_after_perturbation.values()))
    q_values_before_perturbation = np.asarray(list(dict_q_vals_before_perturbation.values()))

    probability_action_perturbed_state = np.exp(q_value_action_perturbed_state) / np.sum(np.exp(q_values_after_perturbation))
    probability_action_original_state = np.exp(q_value_action_original_state) / np.sum(np.exp(q_values_before_perturbation))

    K = cross_entropy(dict_q_vals_after_perturbation, dict_q_vals_before_perturbation, original_action)

    dP = probability_action_original_state - probability_action_perturbed_state

    if probability_action_perturbed_state < probability_action_original_state:  # harmonic mean
        answer = 2 * dP * K / (dP + K)

    QmaxAnswer = computeSaliencyUsingQMaxChange(original_action, dict_q_vals_before_perturbation, dict_q_vals_after_perturbation)
    action_gap_before_perturbation, action_gap_after_perturbation = computeSaliencyUsingActionGap(dict_q_vals_before_perturbation, dict_q_vals_after_perturbation)

    return answer, dP, K, QmaxAnswer, action_gap_before_perturbation, action_gap_after_perturbation


def computeSaliencyUsingQMaxChange(original_action, dict_q_vals_before_perturbation, dict_q_vals_after_perturbation):
    answer = 0

    best_action = None
    best_q_value = 0

    for move, q_value in dict_q_vals_after_perturbation.items():
        if best_action is None:
            best_action = move
            best_q_value = q_value
        elif q_value > best_q_value:
            best_q_value = q_value
            best_action = move

    if best_action != original_action:
        answer = 1

    return answer


def computeSaliencyUsingActionGap(dict_q_vals_before_perturbation, dict_q_vals_after_perturbation):
    q_vals_before_perturbation = sorted(dict_q_vals_before_perturbation.values())
    q_vals_after_perturbation = sorted(dict_q_vals_after_perturbation.values())
    action_gap_before_perturbation = q_vals_before_perturbation[-1] - q_vals_before_perturbation[-2]
    action_gap_after_perturbation = q_vals_after_perturbation[-1] - q_vals_after_perturbation[-2]

    return action_gap_before_perturbation, action_gap_after_perturbation
