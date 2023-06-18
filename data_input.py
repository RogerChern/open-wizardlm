import json


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
