import sys
sys.path.append("./calvin/calvin_models/")
sys.path.append("./calvin/calvin_env/")
sys.path.append("./calvin/calvin_env/tacto")
sys.path.append("./calvin/calvin_models/calvin_agent/")
sys.path.append("./open_flamingo/")
sys.path.append("./robot_flamingo/")

import numpy as np
import hydra
import random

from math import pi

from env.env_utils import *

from pathlib import Path
from omegaconf import OmegaConf
import pybullet as p

import itertools


class ProtoEnv():
    def __init__(self, base_env, task_oracle, num_trials, trial_steps, instructions_per_trial=1, num_objs=3,
                 provide_feedback=True, provide_train_feedback=False, provide_lm_feedback=True, reset_robot_pos=True):
        super(ProtoEnv, self).__init__()

        self.base_env = base_env
        self.task_oracle = task_oracle
        self.num_objs = num_objs
        if self.num_objs == 5:
            self.train_tasks = ['move_slider_left', 'move_slider_right', 'open_drawer', 'close_drawer', 'turn_on_lightbulb',
                            'turn_off_lightbulb', 'turn_on_led', 'turn_off_led', 'lift_red_block_table']
            self.block_objects_train = ['red_block']
            self.static_objects = ['slider', 'drawer', 'led', 'lightbulb']
            self.objects = self.block_objects_train + self.static_objects
        elif self.num_objs == 7:
            self.train_tasks = ['move_slider_left', 'move_slider_right', 'open_drawer', 'close_drawer', 'turn_on_lightbulb',
                            'turn_off_lightbulb', 'turn_on_led', 'turn_off_led', 'lift_red_block_table', 'lift_pink_block_table', 'lift_blue_block_table',
                            'lift_red_block_slider', 'lift_pink_block_slider', 'lift_blue_block_slider']
            self.block_objects_train = ['red_block', 'pink_block', 'blue_block']
            self.static_objects = ['slider', 'drawer', 'led', 'lightbulb']
            self.objects = self.block_objects_train + self.static_objects
        else:
            self.train_tasks = ['open_drawer', 'close_drawer', 'turn_on_lightbulb', 'turn_off_lightbulb', 'move_slider_left', 'move_slider_right']
            self.block_objects_train = []
            self.static_objects = ['drawer', 'lightbulb', 'slider']
            self.objects = self.static_objects

        self.train_combinations, self.eval_combinations = self.get_object_combinations()

        self.task_instruction_feedback = {'move_slider_left': 'move the door to the left', 'move_slider_right': 'move the door to the right',
                                            'open_drawer': 'pull the handle to open the drawer', 'close_drawer': 'push the handle to close the drawer', 'turn_on_lightbulb': 'turn on the lightbulb',
                                            'turn_off_lightbulb': 'turn off the lightbulb', 'turn_on_led': 'turn on the led', 'turn_off_led': 'turn off the led',
                                            'lift_red_block_table': 'lift the red block from the table', 'lift_pink_block_table': 'lift the pink block from the table',
                                            'lift_blue_block_table': 'lift the blue block from the table', 'lift_red_block_slider': 'lift the red block from the slider',
                                            'lift_pink_block_slider': 'lift the pink block from the slider', 'lift_blue_block_slider': 'lift the blue block from the slider'}
         
        self.all_tasks = self.train_tasks

        self.num_trials = num_trials
        self.trial_steps = trial_steps
        self.instructions_per_trial = instructions_per_trial



        self.provide_feedback = provide_feedback
        self.provide_train_feedback = provide_train_feedback
        self.provide_lm_feedback = provide_lm_feedback
        self.reset_robot_pos = reset_robot_pos

    def episode_reset(self, feedback_state=None, environment_state=None, meta_task=None, train=True):
        self.trial_count = 0
        self.trial_step_count = 0
        self.completed_instructions = 0
        self.trial_done = False
        self.last_reward = 0
        noise = False

        if train:
            combinations = self.train_combinations
        else:
            combinations = self.eval_combinations

        self.task_success_dict = {task: False for task in self.all_tasks}
        self.episode_completed_tasks = []

        # Sample which task to perform
        self.initial_state, self.meta_task, self.environment_description = self.create_dynamic_env(combinations, feedback_state, environment_state, meta_task)

        if self.reset_robot_pos:
            self.reset_robot_position()
            noise = True

        robot_obs, scene_obs = self.get_env_state_for_initial_condition(self.initial_state, noise)
        self.base_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = self.base_env.get_info()

        # Sample templates for feedback for the tasks for variable alternative feedback
        # Sample templates for feedback for the tasks for variable alternative feedback
        self.first_trial_template = build_template_first_trial(False if train else True)
        self.further_trial_template = build_template_further_trial(False if train else True)
        
        if train:
            self.reward_template = random.choice(REWARD_TEMPLATES)
        else:
            self.reward_template = random.choice(REWARD_TEMPLATES_EVAL)

        self.sampled_task_instructions = {key: random.choice(TASK_INSTRUCTIONS[key]) for key in TASK_INSTRUCTIONS.keys()} 
        
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
        noise = False

        if self.provide_feedback:
            feedback = self.get_feedback(instruction, trial_end=True, provide_train_feedback=self.provide_train_feedback, provide_lm_feedback=self.provide_lm_feedback)
            self.instruction_success_dict = {}
        else:
            feedback = None

        self.task_success_dict = {task: False for task in self.all_tasks}
        if self.reset_robot_pos:
           self.reset_robot_position()
           noise = True
        robot_obs, scene_obs = self.get_env_state_for_initial_condition(self.initial_state, noise)
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
        task_info = [list(self.task_oracle.get_task_info_for_set(self.start_info, info, {task})) for task in self.all_tasks]
        for subtask in self.all_tasks:
            check, updated = self.task_info_check(subtask, task_info)
            if check:
                # Add completed task to episode completed tasks
                subtask_str = subtask.replace('_', ' ')
                if subtask_str not in self.episode_completed_tasks:
                    self.episode_completed_tasks.append(subtask_str)
                
                # If meta task, track reward and end trial with trial feedback
                if subtask == self.meta_task:
                    if self.provide_feedback:
                        self.instruction_success_dict[instruction] = {'inst': instruction, 'rew': 1}
                    self.trial_done = True
                    reward = 1
                    print('meta task completed')
                # Else, track reward and provide feedback
                else:
                    if updated:
                        if self.provide_feedback and self.trial_done is False:
                            self.instruction_success_dict[instruction] = {'inst': instruction, 'rew': 0}
                            feedback = self.get_feedback(instruction, info, provide_train_feedback=self.provide_train_feedback, provide_lm_feedback=self.provide_lm_feedback)

        self.completed_instructions = len(self.instruction_success_dict)

        if self.trial_done or self.completed_instructions == self.instructions_per_trial:
            self.last_reward = reward
            self.trial_done = True
        
        step_info = {'info': info, 'task_completed': True if len(self.instruction_success_dict) > 0 else False}
        return observation, reward, feedback, self.trial_count, self.trial_done, step_info

    def render(self, mode="rgb_array"):
        return self.base_env.render(mode=mode)
    
    def get_feedback(self, instruction=None, info=None, trial_end=False, provide_train_feedback=False, provide_lm_feedback=True):
        lm_feedback = None
        train_feedback = None
        alt_feedback = None
        
        if self.provide_feedback is True:
            if trial_end:
                # Provide feedback at trial end
                env_status = self.get_env_status(self.start_info)
                if self.trial_count == 0:
                    # Feedback for the first trial
                    if provide_lm_feedback:
                        lm_feedback = {"role": "user", "content": "Beginning of trial 1: Please give an instruction for this trial. {0} Remember to use the template: Explanation: <your explanation>  Notes: <your notes>  Instruction: <your instruction>'." \
                        "Remember to keep your instructions clear, concise and short. If you found a rewarding instruction repeat it for all future trials. Remember to strictly follow the guidelines given to you.".format(env_status)}
                    if provide_train_feedback:
                        used_objects = ', '.join([obj.replace('_', ' ') for obj in self.used_objects])
                        train_feedback = 'first trial. episode objects: {0}'.format(used_objects)
                        alt_feedback = self.first_trial_template.format(used_objects)
                else:
                    # Feedback for the consecutive trials
                    if provide_lm_feedback:
                        assert instruction is not None, "Instruction is required for feedback"
                        completed_tasks = ''
                        for key, value in self.instruction_success_dict.items():
                            reward_string = 'was rewarding' if bool(value['rew']) else 'was not rewarding'
                            completed_tasks += key + reward_string + '; '

                        if not bool(self.instruction_success_dict):
                            completed_tasks = 'No tasks were completed.'

                        lm_feedback = {"role": "user", "content": "Trial {0} completed. The following instructions were completed: {1}. Please give an instruction for the next trial. {2} You must adhere to the following template: Explanation: <your explanation>  Notes: <your notes>  Instruction: <your instruction>'." \
                        "Remember to keep your instructions clear, concise and short. If you found a rewarding instruction repeat it for all future trials. Remember that you are strictly instructed to follow the guidelines given to you.".format(self.trial_count, completed_tasks, env_status)}
                    
                    if provide_train_feedback:
                    
                        completed_tasks_trial = [key for key, value in self.task_success_dict.items() if value is True]

                        # Train feedback for success
                        if self.meta_task in completed_tasks_trial:
                            reward_string = 'was rewarding last trial'
                            task_instruction = self.task_instruction_feedback[self.meta_task]
                            train_feedback = task_instruction + ' ' + reward_string + '; '

                            # Alt feedback for success
                            alt_feedback = self.reward_template.format(self.sampled_task_instructions[self.meta_task])

                        else:
                            used_objects_episode = [obj.replace('_', ' ') for obj in self.used_objects]
                            used_objects_str = ', '.join(used_objects_episode)
                            explored_objects = []
                            
                            for obj in self.used_objects:
                                for task in self.episode_completed_tasks:
                                    if obj in ['red_block', 'blue_block', 'pink_block']:
                                        obj = obj.replace('_', ' ')
                                    if obj in task:
                                        explored_objects.append(obj)

                            explored_objects_str = ', '.join(explored_objects)
                            if len(explored_objects) == 0:
                                explored_objects_str = 'none'
                            meta_task_str = self.meta_task.replace('_', ' ')
                            if meta_task_str in self.episode_completed_tasks and meta_task_str not in completed_tasks_trial:
                                train_feedback = 'no reward last trial. episode objects: {0}. explored objects: {1}. {2} was rewarding in previous trials'.format(used_objects_str, explored_objects_str, self.task_instruction_feedback[self.meta_task])
                            else:
                                train_feedback = 'no reward last trial. episode objects: {0}. explored objects: {1}'.format(used_objects_str, explored_objects_str)
                            
                            # Alt feedback
                            alt_feedback = self.further_trial_template.format(used_objects_str, explored_objects_str)

            # TODO: Fix and update this!!
            else:
                # Feedback if multiple tasks can be completed per trial
                env_status = self.get_env_status(info)
                last_action = [key for key, value in self.task_success_dict.items() if value is True][-1]
                last_action = self.task_instruction_feedback[last_action]
                if self.last_reward:
                    if provide_lm_feedback:
                        print(instruction.replace('.', '_'))
                        lm_feedback = {"role": "user", "content": instruction.replace('.', '_') + " was rewarding. {0} Give further instructions to find other rewarding tasks. Remember to use the template: Explanation: <your explanation>  Notes: <your notes>  Instruction: <your instruction>." \
                        "Keep your instructions clear, concise and short. Remember to strictly follow the guidelines given to you. Do not include anything else in your instruction besides the name of the object and the action. Do not include instructions for future trials.".format(env_status)}
                    if provide_train_feedback:
                        train_feedback = "Action {0} was rewarding.".format(last_action)
                else:
                    if provide_lm_feedback:
                        print(instruction.replace('.', ' '))
                        lm_feedback = {"role": "user", "content": instruction.replace('.', ' ') + " was not rewarding. {0} Give a new instruction for this trial. Remember to use the template: Explanation: <your explanation>  Notes: <your notes>  Instruction: <your instruction>." \
                        "Keep your instructions clear, concise and short. Remember to strictly follow the guidelines given to you. Do not include anything else in your instruction besides the name of the object and the action. Do not include instructions for future trials.".format(env_status)}
                    if provide_train_feedback:
                        train_feedback = "Action {0} was not rewarding.".format(last_action)
        
        feedback = {'lm_feedback': lm_feedback, 'train_feedback': train_feedback, 'alt_feedback': alt_feedback}

        return feedback
    
    def create_dynamic_env(self, combinations, feedback_state=None, environment_state=None, meta_task=None):

        if environment_state is None:
            used_objects = random.choice(combinations)
            initial_state = {}
            for static_object in self.static_objects:
                if static_object in ['led', 'lightbulb']:
                    initial_state[static_object] = random.choice([0, 1])
                elif static_object in ['drawer']:
                    initial_state[static_object] = random.choice(['open', 'closed'])
                else:
                    initial_state[static_object] = random.choice(['left', 'right'])
            #Hack for now
            if self.num_objs == 3:
                initial_state['led'] = 0
            else:
                initial_state['led'] = random.choice([0, 1])
            

            if self.num_objs == 7:
                locations = ['table', 'slider_left', 'slider_right']
                for block in self.block_objects_train:
                    # First check for all blocks that they are not behind the slider if they are to be used in the episode
                    if block in used_objects:
                        location_used_block = random.choice([loc for loc in locations if initial_state['slider'] not in loc])
                        locations.remove(location_used_block)
                        initial_state[block] = location_used_block
                
                for block in [blk for blk in self.block_objects_train if blk not in used_objects]:
                    sampled_location = random.choice(locations)
                    locations.remove(sampled_location)
                    initial_state[block] = sampled_location
            else:
                initial_state['red_block'] = 'table'
                initial_state['pink_block'] = 'slider_right' 
                initial_state['blue_block'] = 'slider_left'

            random.shuffle(used_objects)
            self.sampled_tasks = self.state_to_tasks(initial_state, used_objects)
            meta_task = random.choice(self.sampled_tasks)

        else:
            assert feedback_state is not None, "Feedback state is required for environment state"
            initial_state = environment_state
            used_objects = feedback_state
            self.sampled_tasks = self.state_to_tasks(initial_state, used_objects)
            if meta_task is None:
                meta_task = random.choice(self.sampled_tasks)
            else:
                assert meta_task in self.sampled_tasks, "Meta task must be possible for environment state"

        object_descriptions = self.get_object_descriptions(used_objects, initial_state)

        environment_description = create_environment_description(object_descriptions)
        
        return initial_state, meta_task, environment_description
    
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
                else:
                    obj_status = 'The slider door is currently at the {0}'.format(slider_status)
            else:
                location = self.initial_state[obj]
                if location == 'table':
                    location = 'on the table'
                else:
                    location = 'in the slider'
                obj_status = '{0} is currently {1} '.format(obj, location)
            object_status_list.append(obj_status)
        
        # Generate string containing the status of the environment to ground the LLM
        if len(object_status_list) == 3:
            env_status_string = 'The current status of the environment: {0}; {1}; {2}.'.format(
                object_status_list[0], object_status_list[1], object_status_list[2]
            )
        else:
            env_status_string = 'The current status of the environment: {0}; {1}; {2}; {3}.'.format(
                object_status_list[0], object_status_list[1], object_status_list[2], object_status_list[3]
            )
            
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
                if self.task_success_dict[task] is False:
                    self.task_success_dict[task] = True
                    return True, True
                else:
                    return True, False
            else:
                if self.task_success_dict[task] is False:
                    return False, False
                else:
                    return True, False 
        else:
            for pair in task_pairs:
                if task in pair:
                    pair.remove(task)
                    partner_task = pair[0]
                    if task in task_info:
                        if self.task_success_dict[task] is False:
                            self.task_success_dict[task] = True
                            return True, True
                        else:
                            return True, False

                    elif self.task_success_dict[task]:
                        return True, False
                    elif self.task_success_dict[partner_task] is True and not partner_task in task_info:
                        self.task_success_dict[task] = True
                        return True, True
                    else:
                        return False, False
    
    def reset_robot_position(self):
        '''reset robot initial position after each trial'''
        base_position = [-0.34, -0.46, 0.29]
        variance_x = np.random.randint(-15, 15) * 0.01
        variance_y = np.random.randint(-2, 2) * 0.01
        variance_z = np.random.randint(-5, 5) * 0.01
        
        new_position = [base_position[0] + variance_x, base_position[1] + variance_y, base_position[2] + variance_z]
        #new_position = base_position
        print(new_position)
        
        num_bodies = p.getNumBodies()
        for i in range(num_bodies):
            body_info = p.getBodyInfo(i)
            body_id = p.getBodyUniqueId(i)
            body_name = body_info[1].decode("utf-8")
            if body_name == 'panda':
                p.resetBasePositionAndOrientation(body_id, new_position, [0.0, 0.0, 0.0, 1.0])
    
    def get_env_state_for_initial_condition(self, initial_condition, noise):
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

        # Whether to randomly sample block positions or not
        if noise:
            block_noise = [np.random.uniform(0.0, 0.15), np.random.uniform(-0.005, 0.005)]
        else:
            block_noise = [0.0, 0.0]

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
            scene_obs[6:9] = [block_table[0][0] + block_noise[0], block_table[0][1],
                            block_table[0][2] + block_noise[1]]
        
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        else:
            if initial_condition["red_block"] == "table":
                scene_obs[12:15] = [block_table[1] + block_noise[0], block_table[1][1],
                                block_table[1][2] + block_noise[1]] 
            else:
                scene_obs[12:15] = [block_table[0][0] + block_noise[0], block_table[0][1],
                                block_table[0][2] + block_noise[1]]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            if initial_condition["red_block"] == "table" or initial_condition['blue_block'] == 'table':
                scene_obs[18:21] = [block_table[1] + block_noise[0], block_table[1][1],
                                    block_table[1][2] + block_noise[1]]
            else:
                scene_obs[18:21] = [block_table[0][0] + block_noise[0], block_table[0][1],
                                block_table[0][2] + block_noise[1]] 
                                
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

        return robot_obs, scene_obs

    
    def state_to_tasks(self, initial_state, objects):
        sampled_tasks = []
        self.sampled_tasks_completed = []
        for obj in objects:
            state = initial_state[obj]
            if obj in ['red_block', 'pink_block', 'blue_block']:
                # We have to make sure that the block is not hidden behind the slider door
                if 'slider' in state:
                    slider_location = state.split('_')[1]
                    if slider_location != initial_state['slider']:
                        sampled_tasks.append('lift_{0}_slider'.format(obj))
                else:
                    sampled_tasks.append('lift_{0}_{1}'.format(obj, state))
            elif obj in ['slider']:
                obj_state = 'left' if state == 'right' else 'right'
                sampled_tasks.append('move_slider_{0}'.format(obj_state))
            elif obj in ['drawer']:
                obj_state = 'open' if state == 'closed' else 'close'
                sampled_tasks.append('{0}_drawer'.format(obj_state))
            elif obj in ['led', 'lightbulb']:
                obj_state = 'on' if state == 0 else 'off'
                sampled_tasks.append('turn_{0}_{1}'.format(obj_state, obj))
        
        return sampled_tasks
    
    def get_object_combinations(self):
        train_combinations = []
        eval_combinations = []

        all_combinations = []
        num_sampled = 4 if self.num_objs == 7 else 3
        all_combinations.extend(list(itertools.combinations(self.objects, num_sampled)))
        all_combinations = [list(combination) for combination in all_combinations]

        if self.num_objs == 5:
            # create eval subset of combinations of objects not seen during training
            for combination in all_combinations:
                if 'slider' in combination and 'led' in combination:
                    eval_combinations.append(combination)
                else:
                    train_combinations.append(combination)
        elif self.num_objs == 7:
            for combination in all_combinations:
                if 'slider' in combination and 'led' in combination or 'red block' in combination and 'blue block' in combination:
                    eval_combinations.append(combination)
                else:
                    if 'red_block' in combination and 'blue_block' in combination and 'pink_block' in combination:
                        continue
                    else:
                        train_combinations.append(combination)
                
        else:
            train_combinations = all_combinations
            eval_combinations = all_combinations
        print(len(train_combinations), len(eval_combinations))
        return train_combinations, eval_combinations




    



if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from pathlib import Path

    from calvin.calvin_env.calvin_env.envs.play_table_env import get_env

    base_env = get_env(Path("./calvin/dataset/calvin_debug_dataset/validation"), show_gui=False)
    task_cfg = OmegaConf.load("./calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    env = ProtoEnv(base_env, task_oracle, num_trials=2, trial_steps=1000, provide_feedback=True, num_objs=7)
    obs, feedback, environment_description = env.episode_reset(train=True)
    obs, feedback, environment_description = env.episode_reset(train=True)