from copy import deepcopy
from typing import Dict, List, Any

import numpy as np
# from stable_baselines3.common.vec_env import VecNormalize

class AugmentationFunction:

    def __init__(self, env=None, **kwargs):
        self.env = env
        self.is_her = True
        self.aug_n = None

    def _deepcopy_transition(
            self,
            augmentation_n: int,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
            infos: List[Dict[str, Any]],
    ):
        aug_obs = np.tile(obs, (augmentation_n,1))
        aug_next_obs = np.tile(next_obs, (augmentation_n,1))
        aug_action = np.tile(action, (augmentation_n,1))
        aug_reward = np.tile(reward, (augmentation_n,1))
        aug_termianted = np.tile(terminated, (augmentation_n,1))
        aug_truncated = np.tile(truncated, (augmentation_n,1))
        aug_infos = np.tile([infos], (augmentation_n,1))

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_termianted, aug_truncated, aug_infos

    def _check_observed_constraints(self, obs, next_obs, reward, **kwargs):
        return True

    def augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminated: np.ndarray,
                 truncated: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):

        if not self._check_observed_constraints(obs, next_obs, reward):
            return None, None, None, None, None, None, None

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_truncated, aug_infos = \
            self._deepcopy_transition(aug_n, obs, next_obs, action, reward, terminated, truncated, infos)

        for i in range(aug_n):
            self._augment(aug_obs, aug_next_obs, aug_action, aug_reward, terminated, truncated, aug_infos, **kwargs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_terminated, aug_truncated, aug_infos

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminated: np.ndarray,
                 truncated: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):
        raise NotImplementedError("Augmentation function not implemented.")


class GoalAugmentationFunction(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        # self.goal_length = self.env.goal_idx.shape[-1]
        self.desired_goal_mask = None
        self.achieved_goal_mask = None
        # self.robot_mask = None
        # self.object_mask = None

    def _sample_goals(self, next_obs, **kwargs):
        raise NotImplementedError()

    def _sample_goal_noise(self, n, **kwargs):
        raise NotImplementedError()

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

    def _is_at_goal(self, achieved_goal, desired_goal, **kwargs):
        raise NotImplementedError()

    def _compute_reward(self, achieved_goal, desired_goal, **kwargs):
        raise NotImplementedError()

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminated: np.ndarray,
                 truncated: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 **kwargs,
                 ):

        new_goal = self._sample_goals(next_obs, p=p, **kwargs)
        obs[:, self.desired_goal_mask] = new_goal
        next_obs[:, self.desired_goal_mask] = new_goal

        achieved_goal = next_obs[:, self.achieved_goal_mask]
        terminated[:] = self._is_at_goal(achieved_goal, new_goal)
        truncated[:] = (~terminated) & truncated

        reward[:] = self._compute_reward(achieved_goal, new_goal)
        # self._set_done_and_info(done, infos, at_goal)

class PandaGoalAugmentationFunction(GoalAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.achieved_goal_mask = self.env.achieved_idx
        self.desired_goal_mask = self.env.goal_idx

    def _is_at_goal(self, achieved_goal, desired_goal, **kwargs):
        return self.env.task.is_success(achieved_goal, desired_goal).astype(bool)

    def _compute_reward(self, achieved_goal, desired_goal, infos=None, **kwargs):
        return self.env.task.compute_reward(achieved_goal, desired_goal, infos)

class TranslateGoal(PandaGoalAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_goals(self, next_obs, **kwargs):
        n = next_obs.shape[0]
        return self.env.task._sample_n_goals(n)


#######################################################################################################################
#######################################################################################################################


class ObjectAugmentationFunction(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        # self.goal_length = self.env.goal_idx.shape[-1]
        self.desired_goal_mask = None
        self.achieved_goal_mask = None
        self.robot_mask = None
        self.object_mask = None

    def _sample_object(self, n, **kwargs):
        raise NotImplementedError()

    def _sample_objects(self, obs, next_obs, **kwargs):
        raise NotImplementedError()

    def _is_at_goal(self, achieved_goal, desired_goal, **kwargs):
        raise NotImplementedError()

    def _compute_reward(self, achieved_goal, desired_goal, **kwargs):
        raise NotImplementedError()

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminated: np.ndarray,
                 truncated: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 **kwargs,
                 ):

        new_obj, new_next_obj = self._sample_objects(obs, next_obs, p=p, **kwargs)
        obs[:, self.object_mask] = new_obj
        next_obs[:, self.object_mask] = new_next_obj
        obs[:, 9:-3] = 0
        next_obs[:, 9:-3] = 0

        # new_goal = self.env.task._sample_n_goals(next_obs.shape[0])
        # obs[:, -3:] = new_goal
        # next_obs[:, -3:] = new_goal
        # achieved_goal = new_next_obj
        # desired_goal = new_goal

        achieved_goal = next_obs[:, self.achieved_goal_mask]
        desired_goal = next_obs[:, self.desired_goal_mask]

        terminated[:] = self._is_at_goal(achieved_goal, desired_goal)
        truncated[:] = (~terminated) & truncated

        reward[:] = self._compute_reward(achieved_goal, desired_goal)
        # self._set_done_and_info(done, infos, at_goal)


class PandaObjectAugmentationFunction(ObjectAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.aug_threshold = np.array([0.06, 0.06, 0.06])  # largest distance from center to block edge = 0.02

        self.achieved_goal_mask = self.env.achieved_idx
        self.desired_goal_mask = self.env.goal_idx
        self.object_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        obj_pos_idx = np.argmax(self.env.obj_idx)
        self.object_mask[obj_pos_idx:obj_pos_idx + 3] = True

        self.obj_size = 3

    def _is_at_goal(self, achieved_goal, desired_goal, **kwargs):
        return self.env.task.is_success(achieved_goal, desired_goal).astype(bool)

    def _compute_reward(self, achieved_goal, desired_goal, infos=None, **kwargs):
        return self.env.task.compute_reward(achieved_goal, desired_goal, infos)

    def _sample_objects(self, obs, next_obs, **kwargs):
        n = obs.shape[0]

        mask = np.ones(n, dtype=bool)
        independent_obj = np.empty(shape=(n, self.obj_size))
        independent_next_obj = np.empty(shape=(n, self.obj_size))

        sample_new_obj = True
        while sample_new_obj:
            new_obj = self._sample_object(1, **kwargs)
            new_next_obj = new_obj
            independent_obj[mask] = new_obj
            independent_next_obj[mask] = new_next_obj

            is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)
            mask[mask] = ~(is_independent & next_is_independent)
            sample_new_obj = np.any(mask)

        return independent_obj, independent_next_obj


    # def _sample_objects(self, obs, next_obs, **kwargs):
    #     n = obs.shape[0]
    #
    #     sample_new_obj = True
    #     while sample_new_obj:
    #         new_obj = self._sample_object(1, **kwargs)
    #         new_next_obj = new_obj
    #
    #         diff = np.abs((obs[:, :3] - new_obj))
    #         next_diff = np.abs((next_obs[:, :3] - new_next_obj))
    #
    #         is_independent = np.any(diff > self.aug_threshold, axis=-1)
    #         next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
    #
    #         sample_new_obj = ~(is_independent & next_is_independent)
    #
    #     return new_obj, new_next_obj


    def _check_independence(self, obs, next_obs, new_obj, new_next_obj, mask):
        new_obj = new_obj
        new_next_obj = new_next_obj

        diff = np.abs((obs[mask, :3] - new_obj))
        next_diff = np.abs((next_obs[mask, :3] - new_next_obj))

        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

        # Stop sampling when new_obj is independent.
        return is_independent, next_is_independent

    def _check_at_goal(self, new_next_obj, desired_goal, mask_dependent):
        at_goal = self.env.task.is_success(new_next_obj[mask_dependent], desired_goal[mask_dependent]).astype(bool)
        return at_goal

    def _check_observed_constraints(self, obs, next_obs, reward, **kwargs):
        diff = np.abs((obs[:, :3] - obs[:, self.object_mask]))
        next_diff = np.abs((next_obs[:, :3] - next_obs[:,self.object_mask]))
        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
        observed_is_independent = is_independent & next_is_independent
        return np.all(observed_is_independent)

class TranslateObject(PandaObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_object(self, n, **kwargs):
        new_obj = self.env.task._sample_n_objects(n)
        return new_obj

    # def _check_observed_constraints(self, obs, next_obs, reward, **kwargs):
    #     return True