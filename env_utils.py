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
        reward_template = random.choice(FURTHER_TRIAL_TEMPLATES_META).format(used_objects_str, explored_objects, meta_task_template)
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
    
    return reward_template

def build_template_first_trial():
    beginning = random.choice(FIRST_TRIAL_TEMPLATES_BEGINNINGS)
    middle = random.choice(TEMPLATES_MIDDLE)
    ending = random.choice(FIRST_TRIAL_TEMPLATES_ENDINGS)
    return beginning + ' ' + middle + ' ' + ending

def build_template_further_trial():
    beginning = random.choice(FURTHER_TRIAL_BEGINNINGS)
    middle = random.choice(TEMPLATES_MIDDLE)
    ending = random.choice(FURTHER_TRIAL_ENDINGS)
    return beginning + ' ' + middle + ' ' + ending

def build_template_further_no_explored():
    beginning = random.choice(FURTHER_TRIAL_BEGINNINGS)
    middle = random.choice(TEMPLATES_MIDDLE)
    ending = random.choice(FIRST_TRIAL_TEMPLATES_ENDINGS)
    return beginning + ' ' + middle + ' ' + ending

def build_template_further_no_meta():
    beginning = random.choice(FURTHER_TRIAL_BEGINNINGS)
    middle = random.choice(TEMPLATES_MIDDLE)
    ending = random.choice(FURTHER_TRIAL_NO_EXPLORED_ENDINGS)
    return beginning + ' ' + middle + ' ' + ending








REWARD_TEMPLATES = ["{0} was rewarding in the last trial.",
                    "The last trial was rewarding with {0}.",
                    "Last trial success: {0} was rewarding.",
                    "In the previous trial, {0} proved to be rewarding.",
                    "The previous trial was successful; {0} was rewarding.",
                    "Success in the last trial: {0} was rewarding.",
                    "Last trial reward: {0} was rewarding.",
                    "The last attempt was rewarding with {0}.",
                    "{0} brought a reward in the previous trial.",
                    "The last trial yielded a reward from {0}.",]

FIRST_TRIAL_TEMPLATES_BEGINNINGS = ["This is the first attempt.",
                                    "Starting with trial one.",
                                    "First trial:",
                                    "Initiating the first trial.",
                                    "Commencing trial one.",
                                    "This is the initial trial.",
                                    "First trial in progress.",
                                    "Beginning the first trial.",
                                    "This is the first trial.",
                                    "Starting first trial:",]

TEMPLATES_MIDDLE = ["Objects to investigate: {}.",
                    "Explore the following objects: {}.",
                    "Focus on exploring these objects: {}.",
                    "Your task is to explore these objects: {}.",
                    "Please investigate the following objects: {}.",
                    "Objects to be explored: {}.",
                    "Objects designated for exploration: {}.",
                    "Explore these objects: {}.",
                    "The objects to explore are: {}.",
                    "Examine the following objects: {}.",]

FIRST_TRIAL_TEMPLATES_ENDINGS = ["Objects that have been explored yet: none.",
                                "Previous objects that have been explored: none.",
                                "Explored objects so far: none.",
                                "Previously explored objects: none.",
                                "Objects that have been explored yet: none.",
                                "Explored objects: none.",
                                "Currently explored objects: none.",
                                "Already explored: none.",
                                "Previously explored objects: none.",
                                "Explored objects so far: none.",]

FURTHER_TRIAL_BEGINNINGS = ["The previous trial yielded no reward.",
                            "Last trial results: no reward.",
                            "No reward was achieved in the last trial.",
                            "The last attempt resulted in no reward.",
                            "Previous trial had no reward.",
                            "There was no reward in the previous trial.",
                            "No reward was found in the last trial.",
                            "The last trial did not yield a reward.",
                            "Last trial outcome: no reward.",
                            "No reward achieved in the last trial.",]

FURTHER_TRIAL_ENDINGS = ["Previously explored objects: {}.",
                        "Objects already explored: {}.",
                        "These objects have been explored: {}.",
                        "Already explored: {}.",
                        "Objects previously explored: {}.",
                        "Please explore these objects: {}.",
                        "Previously explored objects: {}.",
                        "Explored objects so fare: {}.",
                        "Already explored objects include: {}.",
                        "Previously explored objects are: {}.",]

FURTHER_TRIAL_NO_EXPLORED_ENDINGS = ["{} was already rewarding in previous trials.",
                                     "Note that {} was rewarding in prior trials.",
                                    "Previously, {} was rewarding.",
                                    "Recall that {} was rewarding in earlier trials.",
                                    "Remember, {} has been rewarding before.",
                                    "In previous trials, {} was rewarding.",
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
                    'lift_pink_block_table': ['lifting the pink block from the table', 'lifting the pink block from the table by grasping it', 'lifting the pink block', 'picking up the pink block from the table'],}


if __name__ == '__main__':
    for i in range(20):
        template = build_template_further_no_meta()
        print(template)