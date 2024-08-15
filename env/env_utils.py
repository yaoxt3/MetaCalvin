import random


import random

def get_feedback_template(used_objects, explored_tasks_episode, explored_tasks_trial, meta_task, trial_one=False):
    meta_task_string = meta_task.replace('_', ' ')
    explored_tasks_trial = [task.replace('_', ' ') for task in explored_tasks_trial]
    explored_objects = []

    for obj in used_objects:
        for task in explored_tasks_episode:
            if obj in task:
                explored_objects.append(obj)
    
    used_objects_str = str(', '.join(used_objects))
    explored_objects = str(', '.join(explored_objects))

    if trial_one:
        template = build_template_first_trial()
        reward_template = template.format(used_objects_str)
    elif meta_task_string in explored_tasks_trial:
        meta_task_template = random.choice(TASK_INSTRUCTIONS[meta_task])
        reward_template = random.choice(REWARD_TEMPLATES).format(meta_task_template)
    else:
        if meta_task_string in explored_tasks_episode and meta_task_string not in explored_tasks_trial:
            meta_task_template = random.choice(TASK_INSTRUCTIONS[meta_task])
            template = build_template_further_no_meta()
            reward_template = template.format(used_objects_str, explored_objects, meta_task_template)
        else:
            if len(explored_objects) == 0:
                explored_objects = 'none'
                template = build_template_further_no_explored()
                reward_template = template.format(used_objects_str)
            else:
                template = build_template_further_trial()
                reward_template = template.format(used_objects_str, explored_objects)
    
    return reward_template

def build_template_first_trial(eval=False):
    if eval:
        beginning = random.choice(FIRST_TRIAL_TEMPLATES_BEGINNINGS_EVAL)
        middle = random.choice(TEMPLATES_MIDDLE_EVAL)
        ending = random.choice(FIRST_TRIAL_TEMPLATES_ENDINGS_EVAL)
    else:
        beginning = random.choice(FIRST_TRIAL_TEMPLATES_BEGINNINGS)
        middle = random.choice(TEMPLATES_MIDDLE)
        ending = random.choice(FIRST_TRIAL_TEMPLATES_ENDINGS)
    return beginning + ' ' + middle + ' ' + ending

def build_template_further_trial(eval=False):
    if eval:
        beginning = random.choice(FURTHER_TRIAL_BEGINNINGS_EVAL)
        middle = random.choice(TEMPLATES_MIDDLE_EVAL)
        ending = random.choice(FURTHER_TRIAL_ENDINGS_EVAL)
    else:
        beginning = random.choice(FURTHER_TRIAL_BEGINNINGS)
        middle = random.choice(TEMPLATES_MIDDLE)
        ending = random.choice(FURTHER_TRIAL_ENDINGS)
    return beginning + ' ' + middle + ' ' + ending

def build_template_further_no_explored(eval=False):
    if eval:
        beginning = random.choice(FURTHER_TRIAL_BEGINNINGS_EVAL)
        middle = random.choice(TEMPLATES_MIDDLE_EVAL)
        ending = random.choice(FIRST_TRIAL_TEMPLATES_ENDINGS_EVAL)
    else:
        beginning = random.choice(FURTHER_TRIAL_BEGINNINGS)
        middle = random.choice(TEMPLATES_MIDDLE)
        ending = random.choice(FIRST_TRIAL_TEMPLATES_ENDINGS)
    return beginning + ' ' + middle + ' ' + ending

def build_template_further_no_meta(eval=False):
    if eval:
        beginning = random.choice(FURTHER_TRIAL_BEGINNINGS_EVAL)
        middle = random.choice(TEMPLATES_MIDDLE_EVAL)
        ending = random.choice(FURTHER_TRIAL_NO_EXPLORED_ENDINGS_EVAL)
    else:
        beginning = random.choice(FURTHER_TRIAL_BEGINNINGS)
        middle = random.choice(TEMPLATES_MIDDLE)
        ending = random.choice(FURTHER_TRIAL_NO_EXPLORED_ENDINGS)
    return beginning + ' ' + middle + ' ' + ending


def create_environment_description(objects):
    if len(objects) not in [3, 4]:
        raise ValueError("The number of objects must be either 3 or 4.")
    
    object_descriptions = "".join([f" {i+1}) {obj}" for i, obj in enumerate(objects)])
    
    environment_description = [
        {"role": "user", "content": "You are a helpful assistant that gives me instructions on which tasks I should complete, given a set of " \
        "objects that I can interact with. It is not known which task is rewarding, so I need to explore the task space to find the rewarding task as efficiently and quickly " \
        "as possible. Once I find the rewarding task I should exploit my knowledge and keep repeating this task. Following objects are present in the environment:" + \
        object_descriptions + \
        " The objects can only be manipulated in the ways that are described above. There are no hidden features or functions " \
        "I have multiple trials to find the rewarding task. The rewarding task stays " \
        "the same throughout the trials but all the objects are put back to their initial states at the beginning of each trial. After the completion of a task you get feedback if the task was rewarding or not. Additionally at the end of each trial you will get " \
        "information about all completed tasks in the trial and their reward. In order to be as helpful as possible to me you have to follow these guidelines:" \
        " 1) You should act as a mentor and guide me to the next task based on the current environment state and information gathered from previous trials about completed tasks. " \
        " 2) You have to adhere to the following template when giving instructions: 'Explanation: <your explanation>  Notes: <your notes>  Instruction: <your instruction>' If you have any explanations or notes then you have to add them in the specified format " \
        " 3) Do not give instructions that include other forms of manipulation for the objects besides the ones mentioned in the object descriptions. You have to adjust your instructions to the current environment state. For example the instruction 'open the drawer' does not make sense if the drawer is currently open " \
        " 4) Keep your instructions clear, concise, and precise. Do not add any additional information. An example instruction can be: 'pick up the red block'. All instructions have to follow this format. " \
        " 5) Only use one action in your instruction. For example, do not say 'pick up the pink block and move it to the left'. Only use the available actions given to you " \
        " 6) Use the information given to you to give instructions that explore the task space as efficiently as possible. Do not repeat instructions for tasks which were not rewarding. " \
        " 7) If you gave an instruction for a task but I did not complete it, you may repeat the instruction again." \
        " 8) If you find a rewarding instruction you must repeat it for all future trials. Do not give an instruction for a different task" \
        " 9) Explore the objects in the order they are given to you. Explore the first object first. If it is not rewarding, explore the second object. If this is not rewarding explore the third object" \
        "Because you are a helpful assistant you will strictly follow these instructions and guide me on my mission to find the rewarding task and maximize my total reward. "},
        {"role": "assistant", "content": "I understand your instructions and will strictly follow them. I will use my reasoning skills to give useful instructions about which task " \
        "you should complete in order to maximize the total reward."}]
    
    return environment_description




REWARD_TEMPLATES = ["{0} was rewarding in the last trial.",
                    "The last trial was rewarding with {0}.",
                    "Last trial success: {0} was rewarding.",
                    "In the previous trial, {0} proved to be rewarding.",
                    "The previous trial was successful; {0} was rewarding."]

REWARD_TEMPLATES_EVAL = ["Success in the last trial: {0} was rewarding.",
                    "Last trial reward: {0} was rewarding.",
                    "The last attempt was rewarding with {0}.",
                    "{0} brought a reward in the previous trial.",
                    "The last trial yielded a reward from {0}.",]

FIRST_TRIAL_TEMPLATES_BEGINNINGS = ["This is the first attempt.",
                                    "Starting with trial one.",
                                    "First trial:",
                                    "Initiating the first trial.",
                                    "Commencing trial one.",]

FIRST_TRIAL_TEMPLATES_BEGINNINGS_EVAL = ["This is the initial trial.",
                                    "First trial in progress.",
                                    "Beginning the first trial.",
                                    "This is the first trial.",
                                    "Starting first trial:",]

TEMPLATES_MIDDLE = ["Objects to investigate: {}.",
                    "Explore the following objects: {}.",
                    "Focus on exploring these objects: {}.",
                    "Your task is to explore these objects: {}.",
                    "Please investigate the following objects: {}.",]

TEMPLATES_MIDDLE_EVAL = ["Objects to be explored: {}.",
                    "Objects designated for exploration: {}.",
                    "Explore these objects: {}.",
                    "The objects to explore are: {}.",
                    "Examine the following objects: {}.",]

FIRST_TRIAL_TEMPLATES_ENDINGS = ["Objects that have been explored yet: none.",
                                "Previous objects that have been explored: none.",
                                "Explored objects so far: none.",
                                "Previously explored objects: none.",
                                "Objects that have been explored yet: none.",]

FIRST_TRIAL_TEMPLATES_ENDINGS_EVAL = ["Explored objects: none.",
                                "Currently explored objects: none.",
                                "Already explored: none.",
                                "Previously explored objects: none.",
                                "Explored objects so far: none.",]

FURTHER_TRIAL_BEGINNINGS = ["The previous trial yielded no reward.",
                            "Last trial results: no reward.",
                            "No reward was achieved in the last trial.",
                            "The last attempt resulted in no reward.",
                            "Previous trial had no reward.",]

FURTHER_TRIAL_BEGINNINGS_EVAL = ["There was no reward in the previous trial.",
                            "No reward was found in the last trial.",
                            "The last trial did not yield a reward.",
                            "Last trial outcome: no reward.",
                            "No reward achieved in the last trial.",]

FURTHER_TRIAL_ENDINGS = ["Previously explored objects: {}.",
                        "Objects already explored: {}.",
                        "These objects have been explored: {}.",
                        "Already explored: {}.",
                        "Objects previously explored: {}.",]

FURTHER_TRIAL_ENDINGS_EVAL = ["Please explore these objects: {}.",
                        "Previously explored objects: {}.",
                        "Explored objects so fare: {}.",
                        "Already explored objects include: {}.",
                        "Previously explored objects are: {}.",]

FURTHER_TRIAL_NO_EXPLORED_ENDINGS = ["{} was already rewarding in previous trials.",
                                     "Note that {} was rewarding in prior trials.",
                                    "Previously, {} was rewarding.",
                                    "Recall that {} was rewarding in earlier trials.",
                                    "Remember, {} has been rewarding before.",]

FURTHER_TRIAL_NO_EXPLORED_ENDINGS_EVAL = ["In previous trials, {} was rewarding.",
                                    "{} has proven rewarding in past trials.",
                                    "Note that {} was rewarding in earlier trials.",
                                    "Remember, {} was rewarding before.",
                                    "{} has been rewarding in past trials.",]

FIRST_TRIAL_TEMPLATES = ["This is the first attempt. Objects to investigate: {}. Objects that have been explored yet: none.",
"Starting with trial one. Explore the following objects: {}. Previous objects that have been explored: none.",
"First trial: Focus on exploring these objects: {}. Explored objects so far: none.",
"Initiating the first trial. Your task is to explore these objects: {}. Previously explored objects: none.",
"Commencing trial one. Please investigate the following objects: {}. Objects that have been explored yet: none.",
"This is the initial trial. Objects to be explored: {}. Explored objects: none.",
"First trial in progress. Objects designated for exploration: {}. Currently explored objects: none.",
"Beginning the first trial. Explore these objects: {}. Already explored: none.",
"This is the first trial. The objects to explore are: {}. Previously explored objects: none.",
"First trial: Examine the following objects: {}. Explored objects so far: none.",]

FURTHER_TRIAL_TEMPLATES_NO_META = ["No reward in the last trial. Objects to investigate: {}. Previously explored objects: {}.",
"The previous trial yielded no reward. Focus on these objects: {}. Objects already explored: {}.",
"Last trial results: no reward. Objects of interest: {}. Explored objects: {}.",
"No reward was achieved in the last trial. Please explore: {}. Already explored: {}.",
"The last attempt resulted in no reward. Investigate these objects: {}. Objects previously explored: {}.",
"Previous trial had no reward. Objects for exploration: {}. Objects that have been explored: {}.",
"There was no reward in the previous trial. Examine the following objects: {}. Explored so far: {}.",
"No reward was found in the last trial. Focus on exploring: {}. Already explored objects include: {}.",
"The last trial did not yield a reward. Objects to explore: {}. Already explored objects: {}.",
"Last trial outcome: no reward. Objects of interest are: {}. Previously explored objects are: {}.",]

FURTHER_TRIAL_TEMPLATES_NO_EXPLORED = ["No reward in the last trial. Objects to investigate: {}. Objects that have been explored yet: none.",
"The previous trial yielded no reward. Focus on these objects: {}. Previous objects that have been explored: none.",
"Last trial results: no reward. Objects of interest: {}. Explored objects so far: none.",
"No reward was achieved in the last trial. Please explore: {}. Previously explored objects: none.",
"The last attempt resulted in no reward. Investigate these objects: {}. Objects that have been explored yet: none.",
"Previous trial had no reward. Objects for exploration: {}. Explored objects: none.",
"There was no reward in the previous trial. Examine the following objects: {}. Currently explored objects: none.",
"No reward was found in the last trial. Focus on exploring: {}. Already explored: none.",
"The last trial did not yield a reward. Objects to explore: {}. Already explored objects: none.",
"Last trial outcome: no reward. Objects of interest are: {}. Previously explored objects are: none."]

FURTHER_TRIAL_TEMPLATES_META = ["No reward last trial. Objects of interest: {}. Already explored objects: {}. {} was already rewarding in previous trials.",
"The last trial yielded no reward. Focus on these objects: {}. Previously explored objects: {}. Note that {} was rewarding in prior trials.",
"Last trial had no reward. Investigate the following objects: {}. Explored objects: {}. Previously, {} was rewarding.",
"No reward from the last trial. Objects to explore: {}. Already explored: {}. Recall that {} was rewarding in earlier trials.",
"The previous trial was unrewarding. Objects of interest: {}. Explored objects: {}. Remember, {} has been rewarding before.",
"There was no reward in the last trial. Objects to investigate: {}. Previously explored objects: {}. {} has proven rewarding in past trials.",
"No reward in the last attempt. Examine these objects: {}. Already explored: {}. In previous trials, {} was rewarding.",
"The last trial resulted in no reward. Focus on: {}. Already explored objects include: {}. Note that {} was rewarding in earlier trials.",
"No reward achieved in the last trial. Explore these objects: {}. Objects already explored: {}. Remember, {} was rewarding before.",
"Last trial yielded no reward. Objects to focus on: {}. Previously explored objects: {}. {} has been rewarding in past trials.",]

TASK_INSTRUCTIONS = {'move_slider_left': ['moving the door to the left', 'sliding the door to the left', 'grabing the handle and moving the door to the left', 'pulling the handle to move the door to the left'],
                     'move_slider_right': ['moving the door to the right', 'sliding the door to the right', 'grabing the handle and move the door to the right', 'pulling the handle to move the door to the right'],
                    'open_drawer': ['opening the drawer', 'opening the drawer by pulling the handle', 'pulling the handle to open the drawer', 'pulling to open the drawer'],
                    'close_drawer': ['closing the drawer', 'closing the drawer by pushing the handle', 'pushing the handle to close the drawer', 'pushing to close the drawer'], 
                    'turn_on_lightbulb': ['turning on the lightbulb', 'turning on the lightbulb by pushing the handle', 'pushing the switch to turn on the lightbulb', 'toggling the switch to turn on the lightbulb'],
                    'turn_off_lightbulb': ['turning off the lightbulb', 'turning off the lightbulb by pushing the handle', 'pushing the switch to turn off the lightbulb', 'toggling the switch to turn off the lightbulb'], 
                    'turn_on_led': ['turning on the led', 'turning on the led by pushing the button', 'pressing on the button to turn on the led', 'pushing the button to turn on the led'], 
                    'turn_off_led': ['turning off the led', 'turning off the led by pushing the button', 'pressing on the button to turn off the led' 'pushing the button to turn off the led'],
                    'lift_red_block_table': ['lifting the red block from the table', 'lifting the red block from the table by grasping it', 'lifting the red block', 'picking up the red block from the table'],
                    'lift_pink_block_table': ['lifting the pink block from the table', 'lifting the pink block from the table by grasping it', 'lifting the pink block', 'picking up the pink block from the table'],
                    'lift_blue_block_table': ['lifting the blue block from the table', 'lifting the blue block from the table by grasping it', 'lifting the blue block', 'picking up the blue block from the table'],
                    'lift_red_block_slider': ['lifting the red block from the slider', 'lifting the red block from the slider by grasping it', 'lifting the red block', 'picking up the red block from the slider'],
                    'lift_pink_block_slider': ['lifting the pink block from the slider', 'lifting the pink block from the slider by grasping it', 'lifting the pink block', 'picking up the pink block from the slider'],
                    'lift_blue_block_slider': ['lifting the blue block from the slider', 'lifting the blue block from the slider by grasping it', 'lifting the blue block', 'picking up the blue block from the slider'],}

TASK_INSTRUCTION_FEEDBACK = {'move the door to the left': 'move_slider_left', 'move the door to the right': 'move_slider_right',
                            'pull the handle to open the drawer': 'open_drawer', 'push the handle to close the drawer': 'close_drawer', 'turn on the lightbulb': 'turn_on_lightbulb',
                            'turn off the lightbulb': 'turn_off_lightbulb', 'turn on the led': 'turn_on_led', 'turn off the led': 'turn_off_led',
                            'lift the red block from the table': 'lift_red_block_table', 'lift the pink block from the table': 'lift_pink_block_table'}


if __name__ == '__main__':
    for i in range(20):
        template = build_template_further_no_meta()
        print(template)