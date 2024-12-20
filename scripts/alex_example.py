"""
This script demonstrates how to run the Alex agent in the mineland environment.

Please set your key in OPENAI_API_KEY environment variable before running this script.
Or, you can set the key in the script as follows (not recommended):
"""

import os

import mineland
from mineland.alex import Alex

API_KEY = "12345"
BASE_URL = "https://alien-curious-smoothly.ngrok-free.app/v1"
MODEL_NAME = "default"

os.environ["OPENAI_API_KEY"] = API_KEY

mland = mineland.make(
    task_id="playground",
    agents_count=1,
)

# initialize agents
agents = []
alex = Alex(
    personality="None",  # Alex configuration
    llm_model_name=MODEL_NAME,
    vlm_model_name=MODEL_NAME,
    base_url=BASE_URL,
    bot_name="MineflayerBot0",
    temperature=0.1,
)
agents.append(alex)

obs = mland.reset()

agents_count = len(obs)
agents_name = [obs[i]["name"] for i in range(agents_count)]

# first step to includes respawn events and other initializations
actions = mineland.Action.no_op(agents_count)
obs, code_info, event, done, task_info = mland.step(action=actions)

for i in range(10):
    actions = []
    for idx, agent in enumerate(agents):
        action = agent.run(obs[idx], code_info[idx], done, task_info, verbose=True)
        actions.append(action)

    if i % 10 == 0:
        print("task_info: ", task_info)

    obs, code_info, event, done, task_info = mland.step(action=actions)
