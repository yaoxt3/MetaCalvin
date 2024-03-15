import numpy as np
import time
import random
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    collect_plan,
    get_env_state_for_initial_condition,
)


class ProtoEnv():
    def __init__(self, base_env, task_oracle, num_trials, trial_steps, instructions_per_trial=2, provide_feedback=True, provide_train_feedback=False):
        super(ProtoEnv, self).__init__()

        self.base_env = base_env
        self.task_oracle = task_oracle

        self.train_tasks = ['move_slider_left', 'move_slider_right', 'open_drawer', 'close_drawer', 'turn_on_lightbulb',
                            'turn_off_lightbulb', 'turn_on_led', 'lift_red_block_table', 'lift_pink_block_table']
        self.eval_tasks = ['lift_blue_block_table', 'turn_off_led',]
        self.block_objects_train = ['red_block', 'pink_block']
        self.static_objects = ['slider', 'drawer', 'led', 'lightbulb']
        self.all_tasks = self.train_tasks + self.eval_tasks 


        self.num_trials = num_trials
        self.trial_steps = trial_steps
        self.instructions_per_trial = instructions_per_trial

        self.trial_count = 1
        self.trial_step_count = 0

        self.trial_done = False
        self.last_reward = 0

        self.provide_feedback = provide_feedback
        self.provide_train_feedback = provide_train_feedback

    def episode_reset(self, train=True):
        self.trial_count = 0
        self.trial_step_count = 0
        self.completed_instructions = 0
        self.trial_done = False

        if train:
            self.tasks = self.train_tasks
            #self.objects = self.train_objects
        else:
            self.tasks = self.eval_tasks
            #self.objects = self.eval_objects

        self.task_success_dict = {task: False for task in self.all_tasks}

        #Sample which task to perform
        self.initial_state, self.meta_task, self.environment_description = self.create_dynamic_env()

        robot_obs, scene_obs = get_env_state_for_initial_condition(self.initial_state)
        self.base_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = self.base_env.get_info()
        
        if self.provide_feedback:
            feedback = self.get_feedback(trial_end=True, provide_train_feedback=self.provide_train_feedback)
            self.instruction_success_dict = {}
        else:
            feedback = None

        observation = self.base_env.get_obs()
        return observation, feedback, self.environment_description

    def trial_reset(self, instruction):
        self.trial_count += 1
        self.trial_step_count = 0
        self.completed_instructions = 0
        self.trial_done = False

        self.task_success_dict = {task: False for task in self.all_tasks}

        if self.provide_feedback:
            feedback = self.get_feedback(instruction, trial_end=True, provide_train_feedback=self.provide_train_feedback)
            self.instruction_success_dict = {}
        else:
            feedback = None

        robot_obs, scene_obs = get_env_state_for_initial_condition(self.initial_state)
        self.base_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        observation = self.base_env.get_obs()
        return observation, feedback, self.environment_description

    def step(self, action, instruction=None):
        feedback = None
        self.trial_step_count += 1
        action = action.squeeze()

        if self.trial_step_count >= self.trial_steps or self.trial_done:
            self.trial_done = True

        observation, _, _, info = self.base_env.step(action)
        reward = 0
        task_info = [list(self.task_oracle.get_task_info_for_set(self.start_info, info, {task})) for task in self.all_tasks]
        for subtask in self.all_tasks:
            check, updated = self.task_info_check(subtask, task_info)
            if check:
                if subtask == self.meta_task:
                    if self.provide_feedback:
                        self.instruction_success_dict[instruction] = {'inst': instruction, 'rew': 1}
                    self.trial_done = True
                    reward = 1
                    print('meta task completed')
                else:
                    if updated:
                        if self.provide_feedback and self.trial_done is False:
                            self.instruction_success_dict[instruction] = {'inst': instruction, 'rew': 0}
                            feedback = self.get_feedback(instruction, info, provide_train_feedback=self.provide_train_feedback)

        self.completed_instructions = len(self.instruction_success_dict)

        if self.trial_done or self.completed_instructions == self.instructions_per_trial:
            self.last_reward = reward
            self.trial_done = True
        
        step_info = {'info': info, 'task_completed': True if len(self.instruction_success_dict) > 0 else False}
        return observation, reward, feedback, self.trial_count, self.trial_done, step_info

    def render(self, mode="rgb_array"):
        return self.base_env.render(mode=mode)
    
    def get_feedback(self, instruction=None, info=None, trial_end=False, provide_train_feedback=False):
        lm_feedback = None
        train_feedback = None
        
        if self.provide_feedback is True:
            if trial_end:
                env_status = self.get_env_status(self.start_info)
                if self.trial_count == 0:
                    if self.trial_count == 0:
                        lm_feedback = {"role": "user", "content": "Beginning of trial 1: Please give an instruction for this trial. {0} Remember to use the template: 'Explanation: <your explanation> Notes: <your notes> Instruction: <your instruction>'." \
                        "Remember to keep your instructions clear, concise and short. If you found a rewarding instruction repeat it for all future trials. Remember to strictly follow the guidelines given to you. Do not include anything else in your instruction besides the name of the object and the action. Do not include instructions for future trials.".format(env_status)}
                        if provide_train_feedback:
                            train_feedback = "Trial 1. Objects of interest are the {0}, {1} and {2}.".format(self.used_objects[0], self.used_objects[1], self.used_objects[2])
                else:
                    assert instruction is not None, "Instruction is required for feedback"
                    completed_tasks = ''
                    for key, value in self.instruction_success_dict.items():
                        reward_string = 'was rewarding' if bool(value['rew']) else ' was not rewarding'
                        completed_tasks += key.replace('.', ' ') + reward_string + '; '

                    if not bool(self.instruction_success_dict):
                        completed_tasks = 'No tasks were completed.'

                    lm_feedback = {"role": "user", "content": "Trial {0} completed. The following instructions were completed: {1}. Please give an instruction for the next trial. {2} Remember to use the template: 'Explanation: <your explanation> Notes: <your notes> Instruction: <your instruction>'." \
                    "Remember to keep your instructions clear, concise and short. If you found a rewarding instruction repeat it for all future trials. Remember to strictly follow the guidelines given to you. Do not include anything else in your instruction besides the name of the object and the action. Do not include instructions for future trials.".format(self.trial_count, completed_tasks, env_status)}
                    if provide_train_feedback:
                        train_feedback = "End of trial {0}. Completed actions: {1}. Objects of interest are the {2}, {3} and {4}.".format(self.trial_count, completed_tasks, self.used_objects[0], self.used_objects[1], self.used_objects[2])
            else:
                env_status = self.get_env_status(info)
                if self.last_reward:
                    lm_feedback = {"role": "user", "content": instruction.replace('.', '_') + " was rewarding. {0} Give further instructions to find other rewarding tasks. Remember to use the template: 'Explanation: <your explanation> Notes: <your notes> Instruction: <your instruction>'." \
                    "Keep your instructions clear, concise and short. Remember to strictly follow the guidelines given to you. Do not include anything else in your instruction besides the name of the object and the action. Do not include instructions for future trials.".format(env_status)}
                    if provide_train_feedback:
                        train_feedback = "Action {0} was rewarding.".format(instruction)
                else:
                    lm_feedback = {"role": "user", "content": instruction.replace('.', ' ') + " was not rewarding. {0} Give a new instruction for this trial. Remember to use the template: 'Explanation: <your explanation> Notes: <your notes> Instruction: <your instruction>'." \
                    "Keep your instructions clear, concise and short. Remember to strictly follow the guidelines given to you. Do not include anything else in your instruction besides the name of the object and the action. Do not include instructions for future trials.".format(env_status)}
                    if provide_train_feedback:
                        train_feedback = "Action {0} was not rewarding.".format(instruction)
        
        feedback = {'lm_feedback': lm_feedback, 'train_feedback': train_feedback}

        return feedback
    
    def create_dynamic_env(self):
        table_object = random.sample(self.block_objects_train, 1)
        table_object = ['pink_block']
        slider_objects = list(set(self.block_objects_train) - set(table_object))
        possible_tasks = [task for task in self.tasks if slider_objects[0] not in task]

        meta_task = random.choice(possible_tasks)
        meta_task = 'lift_pink_block_table'
        initial_state = {}
        meta_object = None
        for static_object in self.static_objects:
            if static_object in meta_task:
                meta_object = static_object
                if static_object in ['led', 'lightbulb']:
                    if 'on' in meta_task:
                        initial_state[static_object] = 0
                    elif 'off' in meta_task:
                        initial_state[static_object] = 1
                elif static_object in ['drawer']:
                    if 'close' in meta_task:
                        initial_state[static_object] = 'open'
                    elif 'open' in meta_task:
                        initial_state[static_object] = 'closed'
                else:
                    if 'right' in meta_task:
                        initial_state[static_object] = 'left'
                    elif 'left' in meta_task:
                        initial_state[static_object] = 'right' 
            else:
                if static_object in ['led', 'lightbulb']:
                    initial_state[static_object] = random.choice([0, 1])
                elif static_object in ['drawer']:
                    initial_state[static_object] = random.choice(['open', 'closed'])
                else:
                    initial_state[static_object] = random.choice(['left', 'right'])

        initial_state[table_object[0]] = 'table'
        initial_state[slider_objects[0]] = 'slider_right'
        initial_state['blue_block'] = 'slider_left'

        table_object = [obj.replace('_', ' ') for obj in table_object]
        if meta_object is None:
            meta_object = table_object[0]

        all_objects = self.static_objects + table_object
        all_objects.remove(meta_object)
        distract_objects = random.sample(all_objects, 2)
        self.used_objects = [meta_object] + distract_objects
        object_discriptions = self.get_object_descriptions(meta_object, distract_objects, initial_state)


        environment_description = [{"role": "user", "content": "You are a helpfull assistant that gives me instructions on which tasks I should complete, given a set of " \
        "objects that I can interact with. It is not known which task is rewarding, so I need to explore the task space to find the rewarding task as efficiently and quickly " \
        "as possible. Once I find the rewarding task I should exploit my knowledge and keep repeating this task. Following objects are present in the environment:"
        " 1) {0}" \
        " 2) {1}" \
        " 3) {2}" \
        "The objects can only be manipulated in the ways that are described above. There are no hidden features or functions " \
        "I have multiple trials to find the rewarding task. The rewarding task stays " \
        "the same throughout the trials but all the objects are put back to their initial states at the beiginning of each trial. After the completion of a task you get feedback if the task was rewarding or not. Additionally at the end of each trial you will get " \
        "information about all completed task in the trial and their reward. In order to be as helpful as possible to me you have to follow these guidelines:" \
        " 1) You should act as a mentor and guide me to the next task based on the current environment state and information gathered from previous trials about completed tasks. " \
        " 2) You have to adhere to the following template when giving instructions: 'Explanation: <your explanation> Notes: <your notes> Instruction: <your instruction>' If you have any explanations or notes than you have to add them in the specified format " \
        " 3) Do not give instructions that include other forms of manipulation for the objects besides the ones mentioned in the object descriptions. You have to adjust your instructions to the current environment state. For example the instruction 'open the drawer' does not make sense if the drawer is currently open " \
        " 4) Keep your instructions clear, concise and precise. Do not add any additional information. An example instruction can be: 'pick up the red block'. All instructions have to follow this format. " \
        " 5) Only use one action in your instruction. For example do not say 'pick up the bink block and move it to the left'. Only use the available actions given to you "
        " 6) Use the information given to you to give instructions that explore the task space as efficiently as possible. Do not repeat instructions for tasks which were not rewarding. " \
        " 7) If you gave an instruction for a task but I did not complete it, you may repeat the instruction again." \
        " 8) If you find a rewarding instruction you must repeat it for all future trials. Do not give an instruction for a different task" \
        "Because you are a helpfull assistant you will strictly follow these instructions and guide me on my mission to find the rewarding task and maximize my total reward. ".format(object_discriptions[0], object_discriptions[1], object_discriptions[2])},
        {"role": "assistant", "content": "I understand your instructions and will strictly follow them. I will use my reasoning skills to give useful instructions about which task" \
        "you should complete in order to maximize the total reward."}]
        
        return initial_state, meta_task, environment_description
    
    def get_object_descriptions(self, meta_object, distract_objects, initial_state):
        object_descriptions = []
        objects = [meta_object] + distract_objects
        #Shuffle the objects
        random.shuffle(objects)
        for obj in [meta_object] + distract_objects:
            if obj in ['red block', 'pink block']:
                object_descriptions.append('A {0} that lies on the table and can be picked up. Possible instructions: "lift the {0} from the table" '.format(obj))
            elif obj in ['slider']:
                slider_state = initial_state[obj]
                object_descriptions.append('A {0} door that is currently at the {1} and can be moved to the left or right by grasping the handle. Possible instructions: "grasp the handle to slide the door to the left", "grasp the handle to move the slider door to the right"'.format(obj, slider_state))
            elif obj in ['drawer']:
                drawer_state = initial_state[obj]
                object_descriptions.append('A {0} that is currently {1} and can be opened or closed by pulling or pushing the handle. Possible instructions: "open the drawer", "close the drawer"'.format(obj, drawer_state))
            elif obj in ['led']:
                led_state = initial_state[obj]
                led_state = 'on' if led_state else 'off'
                object_descriptions.append('A {0} that is currently {1} and can be turned on or off by pushing the button. Possible instructions: "push the button to turn on the led", "push the button to turn off the led"'.format(obj, led_state))
            elif obj in ['lightbulb']:
                lightbulb_state = initial_state[obj]
                lightbulb_state = 'on' if lightbulb_state else 'off'
                object_descriptions.append('A {0} that is currently {1} and can be turned on or off by pushing or pulling a handle. Possible instructions: "Turn on the lightbulb by pushing the handle", "Turn of the lightbulb by pulling the handle"'.format(obj, lightbulb_state))
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
                obj_status = '{0} is currently on the table'.format(obj)
            object_status_list.append(obj_status)
        
        env_status_string = 'The current status of the environment: {0}; {1}; {2}.'.format(object_status_list[0], object_status_list[1], object_status_list[2])
        return env_status_string
    
    def task_info_check(self, task, task_info):
        '''Adpater for the info from the task oracle. Task oracle completed tasks cancel out.
        Also checks if task is completed for the first time to provide feedback'''

        task_info = [task for sublist in task_info for task in sublist]
        task_paris = [['open_drawer', 'close_drawer'], ['turn_on_lightbulb', 'turn_off_lightbulb'], ['turn_on_led', 'turn_off_led'], 
                      ['move_slider_left', 'move_slider_right']]
        block_tasks = ['lift_red_block_table', 'lift_pink_block_table', 'lift_blue_block_table']

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
            for pair in task_paris:
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
                    
if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from pathlib import Path

    from calvin.calvin_env.calvin_env.envs.play_table_env import get_env

    base_env = get_env(Path("/home/richard/Documents/GitHub/RoboTest/calvin/dataset/calvin_debug_dataset/validation"), show_gui=False)
    task_cfg = OmegaConf.load("/home/richard/Documents/GitHub/RoboTest/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    env = ProtoEnv(base_env, task_oracle, num_trials=2, trial_steps=1000, provide_feedback=True)
    obs, feedback, environment_description = env.episode_reset(train=True)
