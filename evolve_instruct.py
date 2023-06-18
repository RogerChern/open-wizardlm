from openai_gpt_request import send_prompt

class AddConstraints:
    def __init__(self):
        with open('prompts/add_constraints_v2.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class Concretizing:
    def __init__(self):
        with open('prompts/concretizing.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class Deepen:
    def __init__(self):
        with open('prompts/deepen.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class InBreadth:
    def __init__(self):
        with open('prompts/in_breadth.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


class IncreasingReasoningSteps:
    def __init__(self):
        with open('prompts/increasing_reasoning_steps.txt', 'r') as f:
            self.prompt_template = f.read()

    def __call__(self, instruction: str) -> str:
        prompt = self.prompt_template.format(instruction=instruction)
        print(prompt)
        try:
            response = send_prompt('', prompt)
        except Exception as e:
            print(e)
            raise e
        return response


if __name__ == '__main__':
    # add_constraint = AddConstraints()
    # response = add_constraint('Write a python script that prints "Hello World"')
    # print(response)

    # concretizing = Concretizing()
    # response = concretizing('Write a function script that prints "Hello World"')
    # print(response)

    deepen = Deepen()
    response = deepen('Write a function script that prints "Hello World"')
    print(response)