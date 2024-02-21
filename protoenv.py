import numpy as np
from calvin_agent.evaluation.utils import (
    collect_plan,
    get_env_state_for_initial_condition,
)


class ProtoEnv():
    def __init__(self, base_env, task_oracle, num_trials, trial_steps, provide_feedback=True):
        super(ProtoEnv, self).__init__()

        self.base_env = base_env
        self.task_oracle = task_oracle

        self.initial_state = {'led': 0, 'lightbulb': 0, 'slider': 'right', 'drawer': 'open', 'red_block': 'table', 'blue_block': 'table', 'pink_block': 'slider_left', 'grasped': 0}
        self.tasks = ('move_slider_left', 'turn_on_led', 'push_red_block_right')

        #Placeholder for now, generate this dynamically later
        '''
        self.environment_description = [{
                "role": "user",
                "content": "You are a friendly, competent and helpful robot controler. You are in charge of giving language instructions to a robot about which task \
                it should perform. The robot is able to grasp objects and move them around. The robot is infont of a table. On this table are mutltiple objects. \
                A button to turn an LED on or off. A slider door which can be moved left or right. A red block which can be moved left, right or can be picked up. You should give \
                concise, short instructions to the robot. You are controling the robot for \
                one episode, which consists of four trials. At the beginning of the episode one task is randomly sampled to give a reward when completed. This task stays the same \
                for all trials. Only this specific task gives a reward. All other tasks give no reward at all. Your job is to find the rewarding task in as few trials as possible and then exploit your knowledge to maximize the reward. \
                In each trial you can give the robot one instruction. The robot will then try to perform the task based on your instruction. The robot is only able to complete on task per trial. \
                After completion of a trial, you get feedback about whether the task performed by the robot was rewarding or not. You can then give the robot another instruction \
                for the next trial. After completion of each trial, the robot and the environment is reset to the intial state. If you find a task that is rewarding, you have to continue giving the robot instructions to perform the same task again for all trials till the end of the episode. If you find a task that is not \
                rewarding, you can give the robot instructions to perform another task. You first think about which task you choose and then give the corresponding instruction to the robot \
                You have to use the following template to give your explanation, notes and instructions: 'Explanation: <your explanation> Notes: <your notes> Task: <your instruction>' \
                Make sure your instructions are clear, concise and short. An example instruction could be 'Pick up the red block and move it left' or 'Press the button to ligh the LED'. Do not include anything else in your instruction besides the \
                instruction. Only choose one instruction per trial. Do not include anything else besides your explanation and the instruction in your output! Only output a single explanation and a single instruction."
            },
            {"role": "assistant", "content": "I understand my job and will execute it to the best of my abilities."}]
        '''

        self.environment_description = [
            {
                "role": "user",
                "content": "You are a friendly, competent and helpful robot controler. You are in charge of giving language instructions to a robot about which task \
                it should perform.  You can choose between three different instructions: move_slider_left, turn_on_led, push_red_block_right. You are controling the robot for \
                one episode, which consists of four trials. At the beginning of the episode one task is randomly sampled to give a reward when completed. This task stays the same \
                for all trials. Only this specific task gives a reward. All other tasks give no reward at all. Your job is to find the rewarding task in as few trials as possible and then exploit your knowledge to maximize the reward. \
                In each trial you can give the robot one instruction. The robot will then try to perform the task based on your instruction. The robot is only able to complete on task per trial. \
                After completion of a trial, you get feedback about whether the task performed by the robot was rewarding or not. You can then give the robot another instruction \
                for the next trial. If you find a task that is rewarding, you have to continue giving the robot instructions to perform the same task again for all trials till the end of the episode. If you find a task that is not \
                rewarding, you can give the robot instructions to perform another task. You first think about which task you choose and then give the corresponding instruction to the robot \
                You have to use the following template to give your explanation and instructions: 'Explanation: <your explanation> Task: <your instruction>' \
                Only use one the following instructions: move_slider_left, turn_on_led, push_red_block_right. Do not include anything else in your instruction besides the \
                instruction choices given to you. Only choose one instruction per trial. Do not include anything else besides your explanation and the instruction in your output! Only output a single explanation and a single instruction."
            },
            {
                "role": "assistant", "content": "I understand my job and will execute it to the best of my abilities."
            },
            {
                "role": "user", "content": "Beginning Trial 1: Please give the robot an instruction for this trial. Remember to use the template: 'Explanation: <your explanation> Notes: <your notes> Task: <your instruction>'. \
                Remember to keep your instructions clear, concise and short. Do not include anything else in your instruction. "
            },]


        self.task_success_dict = {task: False for task in self.tasks}

        self.num_trials = num_trials
        self.trial_steps = trial_steps

        self.trial_count = 1
        self.trial_step_count = 0

        self.trial_done = False
        self.last_reward = 0

        self.provide_feedback = provide_feedback

    def episode_reset(self):
        self.trial_count = 0

        #Sample which task to perform
        self.meta_task = np.random.choice(self.tasks)

        if self.provide_feedback:
            feedback = self.get_feedback()
        else:
            feedback = None

        robot_obs, scene_obs = get_env_state_for_initial_condition(self.initial_state)
        self.base_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = self.base_env.get_info()
        observation = self.base_env.get_obs()
        return observation, feedback, self.environment_description

    def trial_reset(self, instruction):
        self.trial_count += 1
        self.trial_step_count = 0
        self.trial_done = False
        self.task_success_dict = {task: False for task in self.tasks}

        if self.provide_feedback:
            feedback = self.get_feedback(instruction)
        else:
            feedback = None

        robot_obs, scene_obs = get_env_state_for_initial_condition(self.initial_state)
        self.base_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        observation = self.base_env.get_obs()
        return observation, feedback, self.environment_description

    def step(self, action, instruction=None):
        self.trial_step_count += 1
        action = action.squeeze()

        if self.trial_step_count >= self.trial_steps or self.trial_done:
            self.trial_done = True
            #observation = self.trial_reset(instruction)
        
        observation, _, _, info = self.base_env.step(action)
        reward = 0
        for subtask in self.tasks:
            current_task_info = self.task_oracle.get_task_info_for_set(self.start_info, info, {subtask})
            if len(current_task_info) > 0:
                self.task_success_dict[subtask] = True
                self.trial_done = True

                if subtask == self.meta_task:
                    reward = 1
                    print('meta task completed')
        if self.trial_done:
            self.last_reward = reward
        return observation, reward, self.trial_count, self.trial_done

    def render(self, mode="rgb_array"):
        return self.base_env.render(mode=mode)
    
    def get_feedback(self, instruction=None):
        if self.trial_count == 0:
            feedback = {"role": "user", "content": "Beginning Trial 1: Please give the robot an instruction for this trial. Remember to use the template: 'Explanation: <your explanation> Notes: <your notes> Task: <your instruction>'. \
                        Remember to keep your instructions clear, concise and short. Do not include anything else in your instruction. "}
        else:
            assert instruction is not None, "Instruction is required for feedback"
            if self.last_reward:
                feedback = instruction + " was rewarding in Trial {0}. Beginning Trial {1}: Please give the robot an instruction for this trial. Remember to use the template: 'Explanation: <your explanation> Notes: <your notes> Task: <your instruction>'. \
                Remember to keep your instructions clear, concise and short. Do not include anything else in your instruction. ".format(self.trial_count, self.trial_count+1)
            else:
                feedback = instruction + " was not rewarding in Trial {0}. Beginning Trial {1}: Please give the robot an instruction for this trial. Remember to use the template: 'Explanation: <your explanation> Notes: <your notes> Task: <your instruction>'. \
                Remember to keep your instructions clear, concise and short. Do not include anything else in your instruction. ".format(self.trial_count, self.trial_count+1)
        return feedback