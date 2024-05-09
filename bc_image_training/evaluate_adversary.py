import torch
import torch.multiprocessing as mp
from robomimic.utils.file_utils import *
import tqdm
from adversarial_attacks.fgsm import FGSM

def rollout_worker(args):
    vanilla_bc_model_path, adversary, i = args
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
    try:
        for step_i in range(horizon):
            # get action from policy
            ac = policy(ob=ob_dict)
            if adversary == "fgsm":
                # FGSM attack
                epsilon = 0.1
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
    return float(success["task"])

def rollout_parallel(vanilla_bc_model_path, number_of_rollouts, adversary=None):
    args = [(vanilla_bc_model_path, adversary, i) for i in range(number_of_rollouts)]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap(rollout_worker, args), total=number_of_rollouts))
    success_rate = sum(results) / number_of_rollouts
    print("Success Rate on adversarial environment: ", success_rate)
    return success_rate

vanilla_bc_model_path = '../bc_trained_models/image/bc_image/20240509144611/models/model_epoch_500_Lift_success_0.9.pth'
number_of_rollouts = 10
success_rate = rollout_parallel(vanilla_bc_model_path, number_of_rollouts, adversary="fgsm")

