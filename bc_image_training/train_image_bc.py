import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train
from tests.test_bc import convert_config_for_images

# make default BC config
config = config_factory(algo_name="bc")
config = convert_config_for_images(config)

# set config attributes here that you would like to update
config.experiment.name = "bc_image"
config.train.data = "../datasets/lift/ph/image_v141.hdf5"
config.train.output_dir = "../bc_trained_models/image"
config.train.num_epochs = 500
config.algo.gmm.enabled = False

# debug
debug = False
if debug:
    # shrink length of training to test whether this run is likely to crash
    config.unlock()
    config.lock_keys()

    # train and validate (if enabled) for 3 gradient steps, for 2 epochs
    config.experiment.epoch_every_n_steps = 3
    config.experiment.validation_epoch_every_n_steps = 3
    config.train.num_epochs = 2

    # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
    config.experiment.rollout.rate = 1
    config.experiment.rollout.n = 2
    config.experiment.rollout.horizon = 10

    # send output to a temporary directory
    config.train.output_dir = "/tmp/bc_image"

# logging
config.experiment.logging.log_wandb = True
config.experiment.logging.wandb_proj_name = "Adversarial Attacks"

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# launch training run
train(config, device=device)
