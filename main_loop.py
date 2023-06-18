import asyncio
import aiohttp
import json
import os
import signal
from asyncio import Queue
from evolve_instruct_code import RandomSystemPrompt

random_system_prompter = RandomSystemPrompt()


def parse_json_instructions(data):
    instructions_list = []

    for entry in data:
        instruction = entry.get('instruction', '')
        input_data = entry.get('input', '')
        
        if input_data:
            instructions_list.append(f"{instruction} {input_data}")
        else:
            instructions_list.append(instruction)

    return instructions_list

with open('data/code_alpaca_20k.json', 'r') as f:
    json_data =  json.loads(f.read())


items = parse_json_instructions(json_data) # the items we want to send network requests to

# import random
# for instruction in random.choices(items, k=10):
#     print(instruction)
# exit(0)

num_retries = 3
pause_between_retries = 5  # seconds
max_concurrent_requests = 50  # Set the maximum concurrent requests

responses = []
processed_items = []

# Load processed items if any
try:
    with open('processed_items.json', 'r') as f:
        processed_items = json.load(f)
        print(f'processed_items len: {len(processed_items)}')
except FileNotFoundError:
    pass

# Load responses if any
try:
    with open('responses.json', 'r') as f:
        responses = json.load(f)
        print(f'responses len: {len(responses)}')
except FileNotFoundError:
    pass


async def write_to_file(queue):
    while True:
        response, item = await queue.get()  # Wait until a response is available

        # If we receive the sentinel value, exit the loop
        if item is None:
            queue.task_done()
            break

        # append response to responses
        responses.append(response)

        # Save responses with indentation
        with open('responses.json', 'w') as f:
            json.dump(responses, f, indent=4)

        # append item to processed_items and save it
        processed_items.append(item)
        # Save processed items with indentation
        with open('processed_items.json', 'w') as f:
            json.dump(list(processed_items), f, indent=4)

        queue.task_done()  # Indicate that a formerly enqueued task is complete.


async def send_request(session, item, queue, semaphore):
    if item in processed_items:
        return

    for i in range(num_retries):
        await semaphore.acquire()  # Manually acquire the lock
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            }
            payload = {
                "model": "gpt-3.5-turbo-0613",
                "messages": [
                    {"role": "system", "content": random_system_prompter()},
                    {"role": "user", "content": item}
                ]
            }
            
            async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
                response.raise_for_status()
                data = await response.json()
                response_content = data['choices'][0]['message']['content']
                # Add response and item to queue
                await queue.put((response_content, item))
            break
        except aiohttp.ClientError as e:
            semaphore.release()  # Release the lock before sleeping
            print(f"Request failed for {item} with error {e}, attempt {i+1}")
            await asyncio.sleep(pause_between_retries)
        finally:
            semaphore.release()  # Release the lock if not retrying


async def shutdown(signal, loop):
    """Cleanup tasks tied to the service's shutdown."""
    print(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]

    print(f"Cancelling {len(tasks)} tasks")
    for task in tasks:
        print(f"Cancelling {task}")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def main():
    # register signals for interruption
    loop = asyncio.get_running_loop()
    for s in {signal.SIGTERM, signal.SIGINT}:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

    queue = Queue()

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(send_request(session, item, queue, semaphore)) for item in items]
        await asyncio.gather(*tasks, return_exceptions=True)
        # create write_to_file task
        file_writer = asyncio.create_task(write_to_file(queue))
        await queue.join()  # Wait for all items in the queue to be processed.
        await queue.put((None, None))
        await file_writer


# Run the event loop
asyncio.run(main())
