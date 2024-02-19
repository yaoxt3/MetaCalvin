def get_instruction_from_output(output, tokenizer):
    '''Detokenize the output of the LLM to get the instruction'''
    output = output[0]
    output = tokenizer.decode(output, skip_special_tokens=True)
    assistant_output = output.split('[/INST]')[-1]
    #assistant_output = assistant_output.split('Task:')[1].strip().lower().replace("\_", "_")
    assistant_output = assistant_output.split('Task:')[1]
    return assistant_output

def add_output_to_history(history, output, tokenizer):
    '''Add the output of the LLM to the history'''
    output = output[0]
    output = tokenizer.decode(output, skip_special_tokens=True)
    assistant_output = output.split('[/INST]')[-1]
    history.append({"role": "assistant", "content": assistant_output})
    return history

def convert_obs_dict_to_cpu(obs):
    '''Convert the observation dictionary to CPU tensors with nested dictionaries'''
    for key in obs.keys():
        if isinstance(obs[key], dict):
            obs[key] = convert_obs_dict_to_cpu(obs[key])
        else:
            obs[key] = obs[key].to('cpu', torch.float16)
    return obs