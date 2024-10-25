import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from adversarial_models import AdversarialExperiment, AdversarialExperimentParams
from constants import CLAP_SR, ADVERSARIAL_EXPERIMENTS_DIR_PATH


ADVERSARIAL_EXPERIMENTS_DIR_PATH = 'adversarial_experiments_copy/mlp/adv_val'


# for a specific min_snr, max_iter and required_target_pred_confidence: compare the attack success rate for different learning rates (plot)
# for a specific min_snr, max_iter and required_target_pred_confidence: compare the mean num_iter (when successful) of different learning rates


WINDOW_SIZE = int(10 * CLAP_SR)
HOP_SIZE = int(10 * CLAP_SR)


def get_covering_adversarial_experiment(adversarial_experiment_params: AdversarialExperimentParams) -> AdversarialExperiment:
    adversarial_experiments_files_names = os.listdir(ADVERSARIAL_EXPERIMENTS_DIR_PATH)
    for other_adversarial_experiment_file_name in adversarial_experiments_files_names:
        other_adversarial_experiment_file_path = os.path.join(ADVERSARIAL_EXPERIMENTS_DIR_PATH, other_adversarial_experiment_file_name)
        with open(other_adversarial_experiment_file_path) as f:
            other_adversarial_experiment = AdversarialExperiment.model_validate_json(f.read())
        other_adversarial_experiment_params = other_adversarial_experiment.params
        if (other_adversarial_experiment_params.window_size == adversarial_experiment_params.window_size and
                other_adversarial_experiment_params.hop_size == adversarial_experiment_params.hop_size and
                other_adversarial_experiment_params.min_snr == adversarial_experiment_params.min_snr and
                other_adversarial_experiment_params.max_iter >= adversarial_experiment_params.max_iter and
                other_adversarial_experiment_params.required_target_pred_confidence >= adversarial_experiment_params.required_target_pred_confidence and
                other_adversarial_experiment_params.learning_rate == adversarial_experiment_params.learning_rate):
            return other_adversarial_experiment
    raise ValueError(f'Adversarial experiment not found with params: {adversarial_experiment_params.model_dump_json(indent=4)}')


def get_attack_success_rate(
        adversarial_experiment: AdversarialExperiment,
        max_iter: int,
        required_target_pred_confidence: float,
        ) -> float:
    success_cnt = 0
    for adversarial_result in adversarial_experiment.results:
        for adversarial_iteration_result in adversarial_result.iterations_results[:max_iter + 1]:
            if adversarial_iteration_result.target_pred_confidence >= required_target_pred_confidence:
                success_cnt += 1
                break
    return success_cnt / len(adversarial_experiment.results)


def get_mean_num_iter_to_success(
        adversarial_experiment: AdversarialExperiment,
        max_iter: int,
        required_target_pred_confidence: float,
        ) -> float:
    num_iter_to_success_list: List[int] = []
    for adversarial_result in adversarial_experiment.results:
        for num_iter, adversarial_iteration_result in enumerate(adversarial_result.iterations_results[:max_iter + 1]):
            if adversarial_iteration_result.target_pred_confidence >= required_target_pred_confidence:
                num_iter_to_success_list.append(num_iter)
                break
    return float(np.mean(num_iter_to_success_list)) if num_iter_to_success_list else float('inf')


def plot_attack_success_rate(
        min_snr: Optional[float],
        max_iter: int,
        required_target_pred_confidence: float,
        learning_rate_values: List[float],
        ) -> None:
    success_rates_list: List[float] = []
    for learning_rate in learning_rate_values:
        adversarial_experiment = get_covering_adversarial_experiment(AdversarialExperimentParams(
            window_size=WINDOW_SIZE,
            hop_size=HOP_SIZE,
            min_snr=min_snr,
            max_iter=max_iter,
            required_target_pred_confidence=required_target_pred_confidence,
            learning_rate=learning_rate,
        ))
        success_rates_list.append(get_attack_success_rate(adversarial_experiment, max_iter, required_target_pred_confidence))
    success_rates = np.array(success_rates_list)
    plt.figure()
    plt.plot(np.log10(learning_rate_values), 100 * success_rates)
    plt.xlabel('Learning Rate (log10 scale)')
    plt.ylabel('Attack Success Rate (%)')
    plt.show()


def plot_mean_num_iter_to_success(
        min_snr: Optional[float],
        max_iter: int,
        required_target_pred_confidence: float,
        learning_rate_values: List[float],
        ) -> None:
    mean_num_iter_to_success_list: List[float] = []
    for learning_rate in learning_rate_values:
        adversarial_experiment = get_covering_adversarial_experiment(AdversarialExperimentParams(
            window_size=WINDOW_SIZE,
            hop_size=HOP_SIZE,
            min_snr=min_snr,
            max_iter=max_iter,
            required_target_pred_confidence=required_target_pred_confidence,
            learning_rate=learning_rate,
        ))
        mean_num_iter_to_success_list.append(get_mean_num_iter_to_success(adversarial_experiment, max_iter, required_target_pred_confidence))
    mean_num_iter_to_success = np.array(mean_num_iter_to_success_list)
    plt.figure()
    plt.plot(np.log10(learning_rate_values), mean_num_iter_to_success)
    plt.xlabel('Learning Rate (log10 scale)')
    plt.ylabel('Number of iterations until success')
    plt.show()


# def main() -> None:
#     plot_mean_num_iter_to_success(
#         min_snr=60.,
#         max_iter=50,
#         required_target_pred_confidence=0.9,
#         learning_rate_values=[0.000001, 0.00001, 0.0001],
#     )


def main() -> None:
    max_iter = 50
    required_target_pred_confidence = 0.9

    min_snr = 60.
    learning_rate = 1e-4

    adversarial_experiment_params = AdversarialExperimentParams(
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        min_snr=min_snr,
        max_iter=max_iter,
        required_target_pred_confidence=required_target_pred_confidence,
        learning_rate=learning_rate,
    )
    adversarial_experiment = get_covering_adversarial_experiment(adversarial_experiment_params)
    success_rate = get_attack_success_rate(adversarial_experiment, max_iter, required_target_pred_confidence)
    average_iterations = get_mean_num_iter_to_success(adversarial_experiment, max_iter, required_target_pred_confidence)
    print(f'Success rate: {100 * success_rate:.2f}%')
    print(f'Average iterations to success: {average_iterations:.2f}')


if __name__ == '__main__':
    main()
