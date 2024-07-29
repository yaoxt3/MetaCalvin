from time import sleep
from builtins import print


class OracleAgent():
    def __init__(self, policy_model, reasoning_llm=None, vlm=None, task_llm=None):
        self.policy_model = policy_model
        self.reasoning_llm = reasoning_llm
        self.vlm = vlm
        self.task_llm = task_llm

        self.reasoning_hist = []
        self.subtask_instruction = None
        self.num_instruction_step = 0

        self.planned_actions = []

    def step(self, obs, feedback=None, use_reasoning=True, annotation=None, env_description=None, instruction_step=False):
        '''Take a step in the environment'''
        if instruction_step:
            self.instruction, self.output = self.get_instruction(feedback, use_reasoning, annotation, env_description)
            if self.task_llm is not None:
                if self.subtask_instruction is None:
                    self.subtask_instruction = self.instruction
                else:
                    self.instruction = self.subtask_instruction[0]
                    self.subtask_instruction = self.subtask_instruction[1:]
            self.num_instruction_step += 1

        if self.policy_model.replan != -1 and i % self.policy_model.replan == 0:
            if self.policy_model.model.refresh != -1:
                self.policy_model.model.lang_encoder.lm_head.hidden_state = None
                self.policy_model.model.lang_encoder.lm_head.history_memory = self.policy_model.model.lang_encoder.lm_head.history_memory[-self.policy_model.refresh:]
            else:
                self.policy_model.reset()
        
        action = self.policy_model.step(obs, self.instruction, (len(self.planned_actions) == 0))
        if len(self.planned_actions) == 0:
            if action.shape == (7,):
                self.planned_actions.append(action)
            else:
                self.planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = self.planned_actions.pop(0)
        if self.policy_model.use_diff:
            self.policy_model.action_hist_queue.append(action)
        
        return action, self.instruction, self.output


    def get_instruction(self, feedback=None, use_reasoning=True, annotation=None, env_description=None):
        '''Get the instruction from a description of the environment and the feedback'''
        output = None
        if use_reasoning:
            assert self.reasoning_llm is not None, "Reasoning LLM is required if reasoning is used"
            assert feedback is not None, "Feedback is required if reasoning is used"
            if self.vlm is None:
                instruction, output = self.get_reasoning_output(feedback, env_description, None)
            else:
                vlm_description = self.get_vlm_description()
                instruction, output = self.get_reasoning_output(feedback, env_description, vlm_description)
        else:
            assert annotation is not None, "Task annotation is required if reasoning is not used"
            instruction = annotation

        if self.task_llm is not None:
            instruction = self.get_task_llm_output(instruction)
        
        return instruction, output

    def get_reasoning_output(self, feedback, env_description, vlm_description=None):
        '''Get the reasoning output from the feedback and the environment description'''
        if self.num_instruction_step == 0:
            self.reasoning_hist = []

            if vlm_description is not None:
                self.reasoning_hist.append(vlm_description, feedback)
            else:
                env_description.append(feedback)
                self.reasoning_hist = env_description
        else:
            self.reasoning_hist.append(feedback)
        
        instruction, output = self.reasoning_llm.get_instruction(self.reasoning_hist)
        self.reasoning_hist.append(output)
        return instruction, output


    def get_vlm_description(self):
        '''Get the description of the environment from the VLM'''
        pass
    
    def get_task_llm_output(self, task_instruction):
        '''Get the task instruction from the task LLM'''
        pass

    def reset(self):
        self.policy_model.reset()
        self.reasoning_hist = []
        self.subtask_instruction = None
        self.num_instruction_step = 0
        self.planned_actions = []
        self.output = None
        
        

