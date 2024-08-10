import random


def build_template_first_trial(eval=False):
    beginning = random.choice(FIRST_TRIAL_TEMPLATES_BEGINNINGS_EVAL) if eval else random.choice(FIRST_TRIAL_TEMPLATES_BEGINNINGS)
    end = random.choice(TEMPLATES_END_EVAL) if eval else random.choice(TEMPLATES_END)
    return beginning + " " + end

def build_template_further_trial(eval=False):
    beginning = random.choice(FURTHER_TRIAL_TEMPLATES_BEGINNINGS_EVAL) if eval else random.choice(FURTHER_TRIAL_TEMPLATES_BEGINNINGS)
    middle = random.choice(FURTHER_TRIAL_MIDDLE_EVAL) if eval else random.choice(FURTHER_TRIAL_MIDDLE)
    end = random.choice(TEMPLATES_END_EVAL) if eval else random.choice(TEMPLATES_END)
    return beginning + " " + middle + " " + end

def build_template_completed_task(eval=False):
    beginning = random.choice(COMPLETED_TASK_FEEDBACK_EVAL) if eval else random.choice(COMPLETED_TASK_FEEDBACK)
    end = random.choice(TEMPLATES_END_EVAL) if eval else random.choice(TEMPLATES_END)
    return beginning + " " + end


ENVIRONMENT_DESCRIPTION = [{"role": "user", "content": "You are a helpful assistant that gives me instructions on which tasks I should complete, given a set of " \
"objects that I can interact with. It is not known which tasks are rewarding, so I need to explore the task space to find the rewarding tasks as efficiently and quickly " \
"as possible. Once I find the rewarding tasks, I should exploit my knowledge and keep repeating these tasks. The following objects are present in the environment:" \
" 1) {0}" \
" 2) {1}" \
" 3) {2}" \
" 4) {3}" \
"The objects can only be manipulated in the ways that are described above. There are no hidden features or functions. " \
"There are two rewarding tasks in total. I have multiple trials to find the rewarding tasks. The rewarding tasks stay the same throughout the trials, but all the objects are put back to their initial states at the beginning of each trial. " \
"After the completion of a task, you get feedback if the task was rewarding or not. Additionally, at the end of each trial, you will get " \
"information about all completed tasks in the trial and their reward. In order to be as helpful as possible to me, you have to follow these guidelines:" \
" 1) You should act as a mentor and guide me to the next task based on the current environment state and information gathered from previous trials about completed tasks. " \
" 2) You have to adhere to the following template when giving instructions: 'Explanation: <your explanation> Notes: <your notes> Instruction: <your instruction>' If you have any explanations or notes then you have to add them in the specified format. " \
" 3) Do not give instructions that include other forms of manipulation for the objects besides the ones mentioned in the object descriptions. You have to adjust your instructions to the current environment state. For example, the instruction 'open the drawer' does not make sense if the drawer is currently open. " \
" 4) Keep your instructions clear, concise, and precise. Do not add any additional information. An example instruction can be: 'pick up the red block'. All instructions have to follow this format. " \
" 5) Only use one action in your instruction. For example, do not say 'pick up the pink block and move it to the left'. Only use the available actions given to you. " \
" 6) Use the information given to you to give instructions that explore the task space as efficiently as possible. Do not repeat instructions for tasks that were not rewarding. " \
" 7) The goal is to find the two rewarding task as quickly as possible. Once they are found they should both be repeated once per trial." \
" 8) Explore the objects in the order they are given to you. Explore the first object first. If it is not rewarding, explore the second object. If this is not rewarding, explore the third object, and so on. " \
" 9) All objects can only be interacted with once per trial. Once you have moved an object in the trial you can not move it again. This rule also applies for rewarding objects. " \
"Because you are a helpful assistant, you will strictly follow these instructions and guide me on my mission to find the rewarding tasks and maximize my total reward."}, \
{"role": "assistant", "content": "I understand your instructions and will strictly follow them. I will use my reasoning skills to give useful instructions about which task " \
"you should complete in order to maximize the total reward."}]

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

TEMPLATES_END = ["Objects to investigate: {}.",
                    "Explore the following objects: {}.",
                    "Focus on exploring these objects: {}.",
                    "Your task is to explore these objects: {}.",
                    "Please investigate the following objects: {}.",]

TEMPLATES_END_EVAL = ["Objects to be explored: {}.",
                    "Objects designated for exploration: {}.",
                    "Explore these objects: {}.",
                    "The objects to explore are: {}.",
                    "Examine the following objects: {}.",]

FURTHER_TRIAL_TEMPLATES_BEGINNINGS = ["Trial {} finished. Rewarding tasks completed in the last trial: {}.",
                                      "Trial {} concluded. Last trial's rewarding tasks: {}.",
                                      "Trial {} done. Rewarding tasks from the previous trial: {}.",
                                      "Trial {} wrapped up. Completed rewarding tasks in the previous trial: {}.",
                                      "Trial {} completed. Rewarding tasks achieved last trial: {}.",]

FURTHER_TRIAL_TEMPLATES_BEGINNINGS_EVAL = ["Trial {} is over. The rewarding tasks completed in the previous trial were: {}.",
                                          "Trial {} has ended. Rewarding tasks from the last session: {}.",
                                          "Trial {} complete. Last trial's rewarding tasks accomplished: {}.",
                                          "Trial {} ended. Previous trial's rewarding tasks: {}.",
                                          "Trial {} is done. Rewarding tasks identified last trial: {}."]

FURTHER_TRIAL_MIDDLE = ["Completed non-rewarding tasks last trial: {}.",
                        "Unrewarding tasks finished in the previous trial: {}.",
                        "Tasks without reward from the last trial: {}.",
                        "Last trial's completed unrewarding tasks: {}.",
                        "Unrewarding tasks accomplished last trial: {}."]

FURTHER_TRIAL_MIDDLE_EVAL = ["Previous trial's non-rewarding tasks: {}.",
                            "Non-rewarding tasks identified in the last trial: {}.",
                            "Unrewarding tasks in last trial: {}.",
                            "Tasks completed without reward in the last trial: {}.",
                            "Unrewarding tasks from the previous trial: {}."]


COMPLETED_TASK_FEEDBACK = ["A task has been completed.",
                           "Task accomplished.",
                           "You have finished a task.",
                           "Task successfully completed.",
                           "One task completed."]

COMPLETED_TASK_FEEDBACK_EVAL = ["You have completed a task.",
                                "Task was completed.",
                                "Task executed.",
                                "Task finished",
                                "The task has been completed."]

TASK_INSTRUCTIONS = {'move_slider_left': ['moving the door to the left', 'sliding the door to the left', 'grabing the handle and moving the door to the left', 'pulling the handle to move the door to the left'],
                     'move_slider_right': ['moving the door to the right', 'sliding the door to the right', 'grabing the handle and move the door to the right', 'pulling the handle to move the door to the right'],
                    'open_drawer': ['opening the drawer', 'opening the drawer by pulling the handle', 'pulling the handle to open the drawer', 'pulling to open the drawer'],
                    'close_drawer': ['closing the drawer', 'closing the drawer by pushing the handle', 'pushing the handle to close the drawer', 'pushing to close the drawer'], 
                    'turn_on_lightbulb': ['turning on the lightbulb', 'turning on the lightbulb by pushing the handle', 'pushing the switch to turn on the lightbulb', 'toggling the switch to turn on the lightbulb'],
                    'turn_off_lightbulb': ['turning off the lightbulb', 'turning off the lightbulb by pushing the handle', 'pushing the switch to turn off the lightbulb', 'toggling the switch to turn off the lightbulb'], 
                    'turn_on_led': ['turning on the led', 'turning on the led by pushing the button', 'pressing on the button to turn on the led', 'pushing the button to turn on the led'], 
                    'turn_off_led': ['turning off the led', 'turning off the led by pushing the button', 'pressing on the button to turn off the led', 'pushing the button to turn off the led'],
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

