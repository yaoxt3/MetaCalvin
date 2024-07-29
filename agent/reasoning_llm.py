import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ReasoningLLM():
    def __init__(
        self, 
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        torch_dtype=torch.float16,
        temperature=0.8,
        **kwargs):

        super(ReasoningLLM, self).__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        self.temperature = temperature

    def get_instruction(self, input):
        '''Generate an instruction from the input'''
        tokenized_input = self.tokenizer.apply_chat_template(input, add_generation_prompt=True, return_tensors='pt')
        output = self.model.generate(tokenized_input.to(self.model.device), max_new_tokens=256, temperature=self.temperature, do_sample=True)
        return self.get_instruction_from_output(output)

    def get_instruction_from_output(self, output):
        '''Detokenize the output of the LLM to get the instruction'''
        output = output[0]
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        assistant_output = output.split('model')[-1]
        instruction = assistant_output.split('Instruction:')[-1].replace('\n', '')
        assistant_output = {"role": "assistant", "content": assistant_output.replace('\n', '')}
        return instruction, assistant_output
        
        