import torch
import torch.multiprocessing as mp
from robomimic.utils.file_utils import *
import tqdm
from adversarial_attacks.fgsm import FGSM
import wandb
import numpy as np

def rollout_worker_epsilon(args):
    vanilla_bc_model_path, adversary, epsilon, i = args
    # Load the model
    policy, ckpt_dict = policy_from_checkpoint(ckpt_path=vanilla_bc_model_path, verbose=False)
    env, _ = env_from_checkpoint(ckpt_path=vanilla_bc_model_path, verbose=False)
    # do a rollout on clean environment
    policy.start_episode()
    ob_dict = env.reset()
    results = {}
    success = {k: False for k in env.is_success()}  # success metrics
    horizon = 400
    terminate_on_success = False
    step_i = 0
    try:
        for step_i in range(horizon):
            # get action from policy
            ac = policy(ob=ob_dict)
            if adversary == "fgsm":
                # FGSM attack
                ob_adv = FGSM(policy.policy.nets["policy"], ob_dict, ac, epsilon, device='cuda')
                ac = policy(ob=ob_adv)
            # play action
            ob_dict, _, done, _ = env.step(ac)
            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]
            if done or (terminate_on_success and success["task"]):
                break
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
    return float(success["task"]), step_i + 1

def evaluate_epsilon_range(vanilla_bc_model_path, number_of_rollouts, adversary=None):
    wandb.init(project="adversarial_evaluation", entity="sagar8")

    epsilons = np.logspace(-3, 0, 15)  # Generate 15 logarithmically spaced epsilon values from 0.001 to 1

    for epsilon in epsilons:
        args = [(vanilla_bc_model_path, adversary, epsilon, i) for i in range(number_of_rollouts)]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm.tqdm(pool.imap(rollout_worker_epsilon, args), total=number_of_rollouts))

        success_rates, horizons = zip(*results)
        success_rate = sum(success_rates) / number_of_rollouts
        avg_horizon = sum(horizons) / number_of_rollouts

        print(f"Epsilon: {epsilon:.3f}, Success Rate: {success_rate:.3f}, Average Horizon: {avg_horizon:.2f}")
        
        wandb.log({"epsilon": epsilon, "success_rate": success_rate, "avg_horizon": avg_horizon})

    wandb.finish()

# vanilla_bc_model_path = '../bc_trained_models/image/bc_image/20240509144611/models/model_epoch_500_Lift_success_0.9.pth'
number_of_rollouts = 10
evaluate_epsilon_range(vanilla_bc_model_path, number_of_rollouts, adversary="fgsm")