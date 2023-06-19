import json
import os


def parse_code_alpaca_instructions(data):
    instructions_list = []

    for entry in data:
        instruction = entry.get('instruction', '')
        input_data = entry.get('input', '')
        
        if input_data:
            instructions_list.append(f"{instruction} {input_data}")
        else:
            instructions_list.append(instruction)

    return instructions_list


def load_code_alpaca_20k():
    with open('data/code_alpaca_20k.json', 'r') as f:
        json_data = json.loads(f.read())
    items = parse_code_alpaca_instructions(json_data) # the items we want to send network requests to
    return items


def load_evolved_instruction():
    with open('evolved_instruction/responses.json', 'r') as f:
        items = json.loads(f.read())
    return items


def load_dataset(name):
    if name in ['code_alpaca_20k', 'code_alpaca', 'codealpaca']:
        return load_code_alpaca_20k()
    elif os.path.exists(f'evolved_instruction/{name}.json'):
        with open(f'evolved_instruction/{name}.json', 'r') as f:
            return json.loads(f.read())
    else:
        raise FileNotFoundError(f'dataset {name} is not found in the preset or the evolved instruction directory, please check its existence and permission.')
