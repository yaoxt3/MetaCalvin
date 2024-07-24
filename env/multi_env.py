import sys
sys.path.append("./calvin/calvin_models/")
sys.path.append("./calvin/calvin_env/")
sys.path.append("./calvin/calvin_env/tacto")
sys.path.append("./calvin/calvin_models/calvin_agent/")
sys.path.append("./open_flamingo/")
sys.path.append("./robot_flamingo/")

import numpy as np
import torch
import time
import re
import hydra
import random

from math import pi

from env.env_utils import *

from pathlib import Path
from omegaconf import OmegaConf
import pybullet as p

import itertools


class ProtoEnv():
    def __init__(self, base_env, task_oracle, num_trials, trial_steps,
                provide_feedback=True, provide_train_feedback=False, provide_lm_feedback=True, reset_robot_pos=True):
        super(ProtoEnv, self).__init__()

        self.base_env = base_env
        self.task_oracle = task_oracle

        
        self.all_subtasks = ['move_slider_left', 'move_slider_right', 'open_drawer', 'close_drawer', 'turn_on_lightbulb',
                        'turn_off_lightbulb', 'turn_on_led', 'turn_off_led']
        self.static_objects = ['slider', 'drawer', 'led', 'lightbulb']
        self.objects = self.static_objects

        self.train_tasks, self.eval_tasks = self.build_train_and_eval_tasks()
        print(self.train_tasks, self.eval_tasks)
        print(len(self.train_tasks))

        self.task_instruction_feedback = {'move_slider_left': 'move the door to the left', 'move_slider_right': 'move the door to the right',
                                            'open_drawer': 'pull the handle to open the drawer', 'close_drawer': 'push the handle to close the drawer', 'turn_on_lightbulb': 'turn on the lightbulb',
                                            'turn_off_lightbulb': 'turn off the lightbulb', 'turn_on_led': 'turn on the led', 'turn_off_led': 'turn off the led',
                                            'lift_red_block_table': 'lift the red block from the table', 'lift_pink_block_table': 'lift the pink block from the table'}
         


        self.num_trials = num_trials
        self.trial_steps = trial_steps

        self.trial_count = 1
        self.trial_step_count = 0

        self.trial_done = False
        self.last_reward = 0

        self.provide_feedback = provide_feedback
        self.provide_train_feedback = provide_train_feedback
        self.provide_lm_feedback = provide_lm_feedback
        self.reset_robot_pos = reset_robot_pos

    def episode_reset(self, feedback_state=None, environment_state=None, meta_task=None, train=True):
        # Reset statistics
        self.trial_count = 0
        self.trial_step_count = 0
        self.completed_instructions = 0
        self.trial_done = False
        self.last_reward = 0

        # Reset subtask tracking
        self.subtask_success_dict = {task: False for task in self.all_subtasks}
        self.episode_completed_subtasks = []

        # Sample which task to perform
        self.initial_state, self.meta_task, self.environment_description = self.create_dynamic_env(feedback_state, environment_state, meta_task)

        # Sample robot position and reset environment
        if self.reset_robot_pos:
            self.reset_robot_position()
        robot_obs, scene_obs = self.get_env_state_for_initial_condition(self.initial_state)
        self.base_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = self.base_env.get_info()

        # Get first feedback and observation
        if self.provide_feedback:
            feedback = self.get_feedback(trial_end=True, provide_train_feedback=self.provide_train_feedback, provide_lm_feedback=self.provide_lm_feedback)
            self.instruction_success_dict = {}
        else:
            feedback = None

        observation = self.base_env.get_obs()
        return observation, feedback, self.environment_description, self.start_info

    def trial_reset(self, instruction=None):
        '''Reset environment for a new trial'''
        self.trial_count += 1
        self.trial_step_count = 0
        self.completed_instructions = 0
        self.trial_done = False
        self.last_reward = 0

        if self.provide_feedback:
            feedback = self.get_feedback(instruction, trial_end=True, provide_train_feedback=self.provide_train_feedback, provide_lm_feedback=self.provide_lm_feedback)
            self.instruction_success_dict = {}
        else:
            feedback = None

        self.subtask_success_dict = {task: False for task in self.all_subtasks}
        if self.reset_robot_pos:
            self.reset_robot_position()
        robot_obs, scene_obs = self.get_env_state_for_initial_condition(self.initial_state)
        self.base_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        observation = self.base_env.get_obs()
        info = self.base_env.get_info()

        return observation, feedback, self.environment_description, info

    def step(self, action, instruction=None):
        '''
            Perform step in the environment based on the action.

           - Relative actions should be 7 Tuple (x,y,z, euler_x, euler_y, euler_z, gripper):
                - tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50
                - tcp orientation (3): euler angles x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20
                - gripper_action (1): binary (close = -1, open = 1)

            - Absolute actions should be 3 Tuple ((x,y,z), (euler_x, euler_y, euler_z), (gripper)):
                - tcp position (3): x,y,z in absolute world coordinates
                - tcp orientation (3): euler angles x,y,z in absolute world coordinates
                - gripper_action (1): binary (close = -1, open = 1)
        '''
        feedback = None
        reward = 0
        self.trial_step_count += 1

        if not len(action) == 3:
            action = action.squeeze()

        if self.trial_step_count >= self.trial_steps or self.trial_done:
            self.trial_done = True

        observation, _, _, info = self.base_env.step(action)
        task_info = [list(self.task_oracle.get_task_info_for_set(self.start_info, info, {task})) for task in self.all_subtasks]
        for subtask in self.all_subtasks:
            check, updated = self.task_info_check(subtask, task_info)
            subtask_str = subtask.replace('_', ' ')
            if check and subtask_str not in self.episode_completed_subtasks:
                # Add completed task to episode completed tasks
                if subtask_str not in self.episode_completed_subtasks:
                    self.episode_completed_subtasks.append(subtask_str)
                
                # If meta task, track reward and end trial if second meta task is completed
                if subtask in self.meta_task:

                    if self.provide_feedback:
                        self.instruction_success_dict[instruction] = {'inst': instruction, 'rew': 1}
                    reward += 1
                    print('meta subtask completed')
                    feedback = self.get_feedback(instruction, info, provide_train_feedback=self.provide_train_feedback, provide_lm_feedback=self.provide_lm_feedback)
                    if reward == 2:
                        reward = 4
                        self.trial_done = True
                # Else, track reward and provide feedback
                else:
                    if updated:
                        if self.provide_feedback and self.trial_done is False:
                            self.instruction_success_dict[instruction] = {'inst': instruction, 'rew': 0}
                            feedback = self.get_feedback(instruction, info, provide_train_feedback=self.provide_train_feedback, provide_lm_feedback=self.provide_lm_feedback)

        num_completed = len(self.instruction_success_dict)

        if self.trial_done or num_completed == 2:
            self.last_reward = reward
            self.trial_done = True
        
        step_info = {'info': info, 'task_completed': True if len(self.instruction_success_dict) > 0 else False}
        return observation, reward, feedback, self.trial_count, self.trial_done, step_info

    
    def get_feedback(self, instruction=None, info=None, trial_end=False, provide_train_feedback=False, provide_lm_feedback=True):
        lm_feedback = None
        train_feedback = None
        alt_feedback = None
        
        if self.provide_feedback is True:
            if trial_end:
                # Provide feedback for first trial
                env_status = self.get_env_status(self.start_info)
                if self.trial_count == 0:
                    lm_feedback = 'start of training blabla'
                    train_feedback = 'first trial'

                # Feedback for consecutive trials 
                else:
                    lm_feedback = 'continue training'
                    train_feedback = 'second trial'
            else:
                # Feedback after first completed instruction in environment
                lm_feedback = 'completed task'
                train_feedback = 'completed instruction'
        
        feedback = {'lm_feedback': lm_feedback, 'train_feedback': train_feedback, 'alt_feedback': alt_feedback}

        return feedback
    
    def create_dynamic_env(self, feedback_state=None, environment_state=None, meta_task=None):
        used_objects = self.static_objects
        if environment_state is None:
            initial_state = {}
            for static_object in self.static_objects:
                if static_object in ['led', 'lightbulb']:
                    initial_state[static_object] = random.choice([0, 1])
                elif static_object in ['drawer']:
                    initial_state[static_object] = random.choice(['open', 'closed'])
                else:
                    initial_state[static_object] = random.choice(['left', 'right'])
            
           
            initial_state['red_block'] = 'table'
            initial_state['pink_block'] = 'slider_right' 
            initial_state['blue_block'] = 'slider_left'

            initial_state['drawer'] = 'closed'
            initial_state['lightbulb'] = 0

            random.shuffle(used_objects)
            self.state_to_tasks(initial_state, used_objects)
            while True:
                meta_task = random.sample(self.sampled_subtasks, 2)
                if meta_task not in self.eval_tasks:
                    break
            print(meta_task)

        else:
            assert feedback_state is not None, "Feedback state is required for environment state"
            initial_state = environment_state
            used_objects = feedback_state
            self.state_to_tasks(initial_state, used_objects)
            if meta_task is None:
                while True:
                    meta_task = random.sample(self.sampled_subtasks, 2)
                    if meta_task not in self.eval_tasks:
                        break
            else:
                assert meta_task in self.sampled_subtasks, "Meta task must be possible for environment state"

        object_descriptions = self.get_object_descriptions(used_objects, initial_state)
        environment_description = [ENVIRONMENT_DESCRIPTION[0]['content'].format(object_descriptions[0], object_descriptions[1], object_descriptions[2]), ENVIRONMENT_DESCRIPTION[1]]
        
        return initial_state, meta_task, environment_description
    
    def build_train_and_eval_tasks(self):
        '''Create all possible combinations of two subtasks to create a meta task'''
        all_tasks = []
        all_tasks.extend(list(itertools.combinations(self.all_subtasks, 2)))
        all_tasks = [list(task) for task in all_tasks]
        # remove all tasks where static object appears in both subtasks
        for task in all_tasks:
            for obj in self.objects:
                if obj in task[0] and obj in task[1]:
                    all_tasks.remove(task)

        eval_tasks = [['move_slider_right', 'turn_off_led'], ['close_drawer', 'turn_on_lightbulb'], ['turn_on_led', 'turn_off_lightbulb'],
                      ['move_slider_left', 'open_drawer']]
        train_tasks = [task for task in all_tasks if task not in eval_tasks]
        return train_tasks, eval_tasks
    
    def get_object_descriptions(self, objects, initial_state):
        object_descriptions = []
        self.used_objects = objects
        for obj in objects:
            if obj in ['red_block', 'pink_block', 'blue_block']:
                block_state = 'slider' if 'slider' in initial_state[obj] else 'table'
                obj = obj.replace('_', ' ')
                object_descriptions.append('A {0} that lies on the {1} and can be picked up. Possible instructions: "lift the {0} from the table", "lift the {0} from the slider'.format(obj, block_state))
            elif obj in ['slider']:
                slider_state = initial_state[obj]
                object_descriptions.append('A {0} door that is currently at the {1} and can be moved to the left or right. Possible instructions: "move the door to the left", "move the door to the right"'.format(obj, slider_state))
            elif obj in ['drawer']:
                drawer_state = initial_state[obj]
                object_descriptions.append('A {0} that is currently {1} and can be opened or closed by pulling or pushing the handle. Possible instructions: "pull the handle to open the drawer", "close the drawer by pushing the handle"'.format(obj, drawer_state))
            elif obj in ['led']:
                led_state = initial_state[obj]
                led_state = 'on' if led_state else 'off'
                object_descriptions.append('A {0} that is currently {1} and can be turned on or off by pushing the button. Possible instructions: "push the button to turn on the led", "push the button to turn off the led"'.format(obj, led_state))
            elif obj in ['lightbulb']:
                lightbulb_state = initial_state[obj]
                lightbulb_state = 'on' if lightbulb_state else 'off'
                object_descriptions.append('A {0} that is currently {1} and can be turned on or off by pushing or pulling a handle. Possible instructions: "Turn on the lightbulb by pushing the handle", "Turn off the lightbulb by pulling the handle"'.format(obj, lightbulb_state))
        return object_descriptions
    
    def get_env_status(self, info):
        info = info['scene_info']
        light_status = info['lights']['lightbulb']['logical_state']
        led_status = info['lights']['led']['logical_state']
        drawer_status = info['doors']['base__drawer']['current_state']
        slider_status = info['doors']['base__slide']['current_state']

        drawer_status = 'open' if drawer_status >= 0.2 else 'closed'
        slider_status = 'left' if slider_status >= 0.25 else 'right'

        #Generate status string for the objects
        object_status_list = []
        for obj in self.used_objects:
            if obj in self.static_objects:
                if obj == 'led':
                    obj_status = 'The LED is currently {0}'.format('on' if led_status else 'off')
                elif obj == 'lightbulb':
                    obj_status = 'The lightbulb is currently {0}'.format('on' if light_status else 'off')
                elif obj == 'drawer':
                    obj_status = 'The drawer is currently {0}'.format(drawer_status)
                elif obj == 'slider':
                    obj_status = 'The slider door is currently at the {0}'.format(slider_status)
            else:
                location = self.initial_state[obj]
                if location == 'table':
                    location = 'on the table'
                else:
                    location = 'in the slider'
                obj_status = '{0} is currently {1} '.format(obj, location)
            object_status_list.append(obj_status)
        
        env_status_string = 'The current status of the environment: {0}; {1}; {2}.'.format(object_status_list[0], object_status_list[1], object_status_list[2])
        return env_status_string
    
    def task_info_check(self, task, task_info):
        '''Adpater for the info from the task oracle. Task oracle completed tasks cancel out.
        Also checks if task is completed for the first time to provide feedback'''
        task_info = [task for sublist in task_info for task in sublist]
        task_pairs = [['open_drawer', 'close_drawer'], ['turn_on_lightbulb', 'turn_off_lightbulb'], ['turn_on_led', 'turn_off_led'], 
                      ['move_slider_left', 'move_slider_right']]
        block_tasks = ['lift_red_block_table', 'lift_pink_block_table', 'lift_blue_block_table', 'lift_red_block_slider', 'lift_pink_block_slider', 'lift_blue_block_slider']

        if task in block_tasks:
            if task in task_info:
                if self.subtask_success_dict[task] is False:
                    self.subtask_success_dict[task] = True
                    return True, True
                else:
                    return True, False
            else:
                if self.subtask_success_dict[task] is False:
                    return False, False
                else:
                    return True, False 
        else:
            for pair in task_pairs:
                if task in pair:
                    pair.remove(task)
                    partner_task = pair[0]
                    if task in task_info:
                        if self.subtask_success_dict[task] is False:
                            self.subtask_success_dict[task] = True
                            return True, True
                        else:
                            return True, False

                    elif self.subtask_success_dict[task]:
                        return True, False
                    elif self.subtask_success_dict[partner_task] is True and not partner_task in task_info:
                        self.subtask_success_dict[task] = True
                        return True, True
                    else:
                        return False, False
    
    def reset_robot_position(self):
        '''reset robot initial position after each trial'''
        base_position = [-0.34, -0.46, 0.24]
        '''
        variance_x = np.random.choice(np.arange(-0.05, 0.05, 0.001))
        variance_y = np.random.choice(np.arange(-0.02, 0.02, 0.001))
        variance_z = np.random.choice(np.arange(-0.02, 0.02, 0.001))
        new_position = [base_position[0] + variance_x, base_position[1] + variance_y, base_position[2] + variance_z]
        
        base_position = [-0.34, -0.46, 0.24]
        if self.initial_state['drawer'] == 'open':
            variance_x = np.random.randint(-15, 25) * 0.01
        else:
            variance_x = np.random.randint(-15, 25) * 0.01
        variance_y = np.random.randint(-0, 5) * 0.01
        variance_z = np.random.randint(-5, 5) * 0.01
        '''
        variance_x = np.random.randint(-15, 25) * 0.01
        variance_y = np.random.randint(-8, 8) * 0.0125
        variance_z = np.random.randint(-0, 12) * 0.0125
        new_position = [base_position[0] + variance_x, base_position[1] + variance_y, base_position[2] + variance_z]
        
        print(new_position)
        
        num_bodies = p.getNumBodies()
        for i in range(num_bodies):
            body_info = p.getBodyInfo(i)
            body_id = p.getBodyUniqueId(i)
            body_name = body_info[1].decode("utf-8")
            if body_name == 'panda':
                p.resetBasePositionAndOrientation(body_id, new_position, [0.0, 0.0, 0.0, 1.0])
    
    def get_env_state_for_initial_condition(self, initial_condition):
        robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,  # tcp x,y,z in world coordinates
            3.09045411,
            -0.02908596,
            1.50013585, # euler angles x,y,z in world coordinates
            0.07999963, # gripper width
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928, # joint angles
            -1.0,       # gripper state
        ]
        )
        block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
        block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
        block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
        block_table = block_table = [
            np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
            np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
        ]

        

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.07 + np.random.uniform(-0.008, 0.008)
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = [block_table[0][0] + np.random.uniform(0.0, 0.15), block_table[0][1] + np.random.uniform(-0.005, 0.005),
                              block_table[0][2]]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = [block_table[1] + np.random.uniform(-0.05, 0.0), block_table[1][1] + np.random.uniform(-0.005, 0.005),
                                block_table[1][2] + np.random.uniform(-0.005, 0.005)] 
        else:
            scene_obs[12:15] = [block_table[0][0] + np.random.uniform(0.0, 0.1), block_table[0][1] + np.random.uniform(-0.005, 0.005),
                               block_table[0][2] + np.random.uniform(-0.005, 0.005)]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        elif initial_condition["red_block"] == "table" or initial_condition['blue block'] == 'table':
            scene_obs[18:21] = [block_table[1] + np.random.uniform(-0.05, 0.0), block_table[1][1] + np.random.uniform(-0.005, 0.005),
                                block_table[1][2] + np.random.uniform(-0.005, 0.005)]
        else:
            scene_obs[12:15] = [block_table[0][0] + np.random.uniform(0.0, 0.1), block_table[0][1] + np.random.uniform(-0.005, 0.005),
                               block_table[0][2] + np.random.uniform(-0.005, 0.005)] 
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

        return robot_obs, scene_obs
    
    def state_to_tasks(self, initial_state, objects):
        self.sampled_subtasks = []
        self.sampled_subtasks_completed = []
        for obj in objects:
            state = initial_state[obj]
            if obj in ['red_block', 'pink_block', 'blue_block']:
                # We have to make sure that the block is not hidden behind the slider door
                if 'slider' in state:
                    slider_location = state.split('_')[1]
                    if slider_location != initial_state['slider']:
                        self.sampled_subtasks.append('lift_{0}_{1}'.format(obj, state))
                else:
                    self.sampled_subtasks.append('lift_{0}_{1}'.format(obj, state))
            elif obj in ['slider']:
                obj_state = 'left' if state == 'right' else 'right'
                self.sampled_subtasks.append('move_slider_{0}'.format(obj_state))
            elif obj in ['drawer']:
                obj_state = 'open' if state == 'closed' else 'close'
                self.sampled_subtasks.append('{0}_drawer'.format(obj_state))
            elif obj in ['led', 'lightbulb']:
                obj_state = 'on' if state == 0 else 'off'
                self.sampled_subtasks.append('turn_{0}_{1}'.format(obj_state, obj))
    
    
    def render(self, mode="rgb_array"):
        return self.base_env.render(mode=mode)

