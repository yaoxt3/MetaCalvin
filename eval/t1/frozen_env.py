"""ProtoEnv adapter that restores exact reset states from a T1 manifest."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from env.protoenv import ProtoEnv


class FrozenProtoEnv(ProtoEnv):
    """Use manifest-provided robot/scene observations at every trial reset."""

    def _restore_trial_state(self, trial_index: int) -> None:
        try:
            state = self._t1_episode["trial_reset_states"][trial_index]
        except IndexError as error:
            raise ValueError(
                f"Episode {self._t1_episode.get('episode_id')} has no reset state "
                f"for trial {trial_index}"
            ) from error
        robot_obs = np.asarray(state["robot_obs"], dtype=np.float64)
        scene_obs = np.asarray(state["scene_obs"], dtype=np.float64)
        if robot_obs.shape != (15,) or scene_obs.shape != (24,):
            raise ValueError(
                f"Invalid frozen reset shapes: robot={robot_obs.shape}, scene={scene_obs.shape}"
            )
        self.base_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    def episode_reset_from_manifest(self, episode: Mapping[str, Any]):
        self._t1_episode = dict(episode)
        self.trial_count = 0
        self.trial_step_count = 0
        self.completed_instructions = 0
        self.trial_done = False
        self.last_reward = 0
        self.task_success_dict = {task: False for task in self.all_tasks}
        self.episode_completed_tasks = []

        self.initial_state = dict(episode["environment_state"])
        self.meta_task = str(episode["meta_task"])
        self.used_objects = list(episode["feedback_state"])
        self.sampled_tasks = self.state_to_tasks(self.initial_state, self.used_objects)
        if self.sampled_tasks != list(episode["candidate_tasks"]):
            raise ValueError(
                "Manifest candidate order no longer matches ProtoEnv.state_to_tasks: "
                f"computed={self.sampled_tasks}, manifest={episode['candidate_tasks']}"
            )
        if self.meta_task not in self.sampled_tasks:
            raise ValueError(f"Manifest meta-task {self.meta_task!r} is not feasible")

        object_descriptions = self.get_object_descriptions(self.used_objects, self.initial_state)
        from env.env_utils import create_environment_description

        self.environment_description = create_environment_description(object_descriptions)
        templates = episode["feedback_templates"]
        self.first_trial_template = str(templates["first_trial_template"])
        self.further_trial_template = str(templates["further_trial_template"])
        self.reward_template = str(templates["reward_template"])
        self.sampled_task_instructions = dict(episode["sampled_task_instructions"])

        self._restore_trial_state(0)
        self.start_info = self.base_env.get_info()
        feedback = self.get_feedback(
            trial_end=True,
            provide_train_feedback=self.provide_train_feedback,
            provide_lm_feedback=self.provide_lm_feedback,
        )
        self.instruction_success_dict = {}
        return self.base_env.get_obs(), feedback, self.environment_description, self.start_info

    def trial_reset_from_manifest(self, instruction=None):
        self.trial_count += 1
        self.trial_step_count = 0
        self.completed_instructions = 0
        self.trial_done = False
        self.last_reward = 0

        if self.provide_feedback:
            feedback = self.get_feedback(
                instruction,
                trial_end=True,
                provide_train_feedback=self.provide_train_feedback,
                provide_lm_feedback=self.provide_lm_feedback,
            )
            self.instruction_success_dict = {}
        else:
            feedback = None

        self.task_success_dict = {task: False for task in self.all_tasks}
        self._restore_trial_state(self.trial_count)
        observation = self.base_env.get_obs()
        info = self.base_env.get_info()
        return observation, feedback, self.environment_description, info

