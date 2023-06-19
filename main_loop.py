import asyncio
import aiohttp
import json
import os
import signal
from asyncio import Queue
from evolve_instruct_code import RandomSystemPrompt
from data_input import load_dataset

random_system_prompter = RandomSystemPrompt()

num_retries = 3
pause_between_retries = 5  # seconds
max_concurrent_requests = 50  # Set the maximum concurrent requests

dataset_name = 'code_alpaca_20k'
round_number = 'round2'
data_dir = 'evolved_instruction'
max_num_item = 20
items = load_dataset('evolved_instruction/code_alpaca_20k_round1_responses.json', max_num_item)

# Configuration
config = {
    "processed_items_file_prefix": f'{data_dir}/{dataset_name}_',
    "processed_items_file_postfix": round_number + '_',
    "responses_file_prefix": f'{data_dir}/{dataset_name}_',
    "responses_file_postfix": round_number + '_',
}

# Function to load data from JSON file
def load_json_file(file_prefix, file_postfix, file_name):
    try:
        with open(f'{file_prefix}{file_postfix}{file_name}', 'r') as f:
            data = json.load(f)
            print(f'{file_name} len: {len(data)}')
            return data
    except FileNotFoundError:
        return []


# Load processed items if any
processed_items = load_json_file(
    config["processed_items_file_prefix"], config["processed_items_file_postfix"], 'processed_items.json')

# Load responses if any
responses = load_json_file(
    config["responses_file_prefix"], config["responses_file_postfix"], 'responses.json')


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
        with open(f'{config["responses_file_prefix"]}{config["responses_file_postfix"]}responses.json', 'w') as f:
            json.dump(responses, f, indent=4)

        # append item to processed_items and save it
        processed_items.append(item)
        # Save processed items with indentation
        with open(f'{config["processed_items_file_prefix"]}{config["processed_items_file_postfix"]}processed_items.json', 'w') as f:
            json.dump(processed_items, f, indent=4)

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
        # create write_to_file task
        file_writer = asyncio.create_task(write_to_file(queue))
        await asyncio.gather(*tasks, return_exceptions=True)
        await queue.join()  # Wait for all items in the queue to be processed.
        await queue.put((None, None))
        await file_writer


# Run the event loop
asyncio.run(main())
