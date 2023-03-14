import os

import numpy as np
import torch


class Evaluate:
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        model,
        eval_env,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        device=None,
    ):
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.device = device

        self.model = model
        self.eval_env = eval_env
        self.best_model_save_path = log_path

        # Logs will be written in ``evaluations.npz``
        os.makedirs(name=log_path, exist_ok=True)
        if log_path is not None:
            self.log_path = os.path.join(log_path, "evaluations")
        self.evaluations_returns = []
        self.evaluations_timesteps = []
        self.evaluations_successes = []
        # For computing success rate
        self._is_success_buffer = []

    def evaluate(self, t):
        # if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
        returns, successes = self._evaluate()

        if self.log_path is not None:
            self.evaluations_timesteps.append(t)
            self.evaluations_returns.append(returns)
            self.evaluations_successes.append(successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                returns=self.evaluations_returns,
                successes=self.evaluations_successes,
            )

            mean_reward, std_reward = np.mean(returns), np.std(returns)
            mean_success, std_success = np.mean(successes), np.std(successes)

            self.last_mean_reward = mean_reward

            print(f"Eval num_timesteps={t}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Eval num_timesteps={t}, " f"episode_success={mean_success:.2f} +/- {std_success:.2f}")

            if mean_reward > self.best_mean_reward:
                print("New best mean reward!")
                torch.save(self.model, os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward

        return mean_reward, mean_success

    def _evaluate(self):
        eval_returns = []
        eval_successes = []
        
        obs, _ = self.eval_env.reset()    
        for episode_i in range(self.n_eval_episodes):
            ep_returns = []
            ep_successes = []
            done = False
            step = 0
            while not done:
                step += 1
                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    actions = self.model(torch.Tensor(obs).to(self.device))
                    actions = actions.cpu().numpy().clip(self.eval_env.single_action_space.low, self.eval_env.single_action_space.high)
    
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminateds, truncateds, infos = self.eval_env.step(actions)
                done = terminateds[0] or truncateds[0]

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
                
                ep_returns.append(rewards[0])
                ep_successes.append(terminateds[0])
            
            eval_returns.append(np.sum(ep_returns))
            eval_successes.append(np.sum(ep_successes)*100)

        return eval_returns, eval_successes
            