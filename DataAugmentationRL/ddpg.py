# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import glob
import os
import random
import time
from distutils.util import strtobool

import gym, panda_gym, gymnasium

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml
from torch.utils.tensorboard import SummaryWriter

from DataAugmentationRL.aug_functions.common import TranslateGoal, TranslateObject
from DataAugmentationRL.utils.evaluate import Evaluate
from DataAugmentationRL.utils.replay_buffer import ReplayBuffer
from DataAugmentationRL.utils.utils import Actor, QNetwork, make_env, get_latest_run_id


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="DataAugmentationRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-sync-freq", type=int, default=100)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--results-dir", "-f", type=str, default="results", help="directory in which results will be saved")
    parser.add_argument("--results-subdir", "-s", type=str, default="", help="results will be saved to <results_dir>/<env_id>/<subdir>/")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PandaPush-v3", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.95, help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.05, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256, help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.2, help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=10e3, help="timestep to start learning")
    parser.add_argument("--update-freq", type=int, default=2, help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="noise clip parameter of the Target Policy Smoothing Regularization")

    parser.add_argument("--aug-function", type=str, default='translate_object')
    parser.add_argument("--aug-n", type=int, default=1)
    parser.add_argument("--aug-ratio", type=float, default=1)

    parser.add_argument("--eval-freq", type=int, default=10e3)
    parser.add_argument("--eval-episodes", type=int, default=80)



    args = parser.parse_args()
    # fmt: on
    return args

if __name__ == "__main__":
    registered_gymnasium_envs = gymnasium.envs.registry  # pytype: disable=module-attr
    gym.envs.registry.update(registered_gymnasium_envs)
    torch.set_num_threads(1)

    args = parse_args()

    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    save_dir = f"{args.results_dir}/{args.env_id}/{args.results_subdir}"
    run_id = get_latest_run_id(save_dir) + 1
    save_dir += f"/run_{run_id}"


    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name='tmp',
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"{args.env_id}/{args.aug_function}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, save_dir)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )

    rb_aug = ReplayBuffer(
        args.buffer_size*args.aug_n,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )

    eval_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, save_dir)])
    eval_module = Evaluate(model=actor, eval_env=eval_envs, n_eval_episodes=args.eval_episodes, log_path=save_dir, device=device)

    # aug_function = TranslateGoal(env=envs.envs[0].unwrapped)
    aug_function = TranslateObject(env=envs.envs[0].unwrapped)

    # Save config
    with open(os.path.join(save_dir, "config.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    obs_list, next_obs_list, action_list, reward_list, terminated_list, truncated_list, info_list = [], [], [], [], [], [], []

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            if np.random.random() < 0.3:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(device))
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)


        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs
        if "final_observation" in infos:
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(infos["_final_observation"]):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminateds, truncateds, infos)

        for i in range(args.aug_n):
            aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_truncated, aug_info = \
                aug_function.augment(aug_n=1, obs=obs, next_obs=real_next_obs, action=actions, reward=rewards, terminated=terminateds, truncated=truncateds, infos=infos)

            if aug_obs is not None:
                rb_aug.add(aug_obs, aug_next_obs, aug_action, aug_reward[0], aug_terminated[0], aug_truncated[0], aug_info[0])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.update_freq == (args.update_freq-1):
                batch_obs = rb.sample(args.batch_size)
                batch_aug = rb_aug.sample(int(args.aug_ratio*args.batch_size))

                batch_observations = torch.concat((batch_obs.observations, batch_aug.observations))
                batch_actions = torch.concat((batch_obs.actions, batch_aug.actions))
                batch_next_observations = torch.concat((batch_obs.next_observations, batch_aug.next_observations))
                batch_rewards = torch.concat((batch_obs.rewards, batch_aug.rewards))
                batch_dones = torch.concat((batch_obs.dones, batch_aug.dones))

                with torch.no_grad():
                    next_state_actions = target_actor(batch_next_observations)
                    qf1_next_target = qf1_target(batch_next_observations, next_state_actions)
                    next_q_value = batch_rewards.flatten() + (1 - batch_dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

                qf1_a_values = qf1(batch_observations, batch_actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                q_optimizer.zero_grad()
                qf1_loss.backward()
                q_optimizer.step()

                actor_loss = -qf1(batch_observations, actor(batch_observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step > 0 and global_step % args.wandb_sync_freq == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step % args.eval_freq == 0:
            current_time = time.time()-start_time
            print(f"Training time: {int(current_time)} \tsteps per sec: {int(global_step / current_time)}")
            returns, successes = eval_module.evaluate(global_step)

            writer.add_scalar("charts/return", returns, global_step)
            writer.add_scalar("charts/success_rate", successes, global_step)



    envs.close()
