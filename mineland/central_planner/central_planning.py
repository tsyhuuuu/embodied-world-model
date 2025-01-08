import time

from .critic_agent import CriticAgent
from .long_term_planner import LongtermPlanner


class CentralPlanner:
    def __init__(
        self,
        llm_model_name="gpt-4-turbo",
        vlm_model_name="gpt-4-turbo",
        base_url=None,
        max_tokens=512,
        temperature=0,
        save_path="./storage",
        load_path="./load",
        vision=True,
    ):
        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.save_path = save_path + "/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.load_path = load_path
        self.vision = vision
        self.current_progress = None

        print(f"save_path: {self.save_path}")

        self.critic_agent = CriticAgent(
            model_name=self.vlm_model_name,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            save_path=self.save_path,
            vision=self.vision,
        )

        self.long_term_planner = LongtermPlanner(
            model_name=self.vlm_model_name,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            vision=self.vision,
        )

    def set_current_progress(self, agents_info: str) -> None:
        self.current_progress = agents_info

    # TODO: 入出力とpromptを設定
    def generate_long_term_plan(self, obs, task_info) -> None:
        self.long_term_plan = self.long_term_planner.plan(obs, task_info)

    def monitor(self, obs_list, code_info_list, done=None, task_info=None, verbose=False) -> list:
        target_agents_idx = []

        if not done:
            for i, code_info in enumerate(code_info_list):
                if code_info is None or not (code_info.code_error or code_info.is_running):
                    target_agents_idx.append(i)

        if verbose:
            agent_names = [obs_list[i]["name"] for i in target_agents_idx]
            print("==========monitoring==========")
            print(f"target agents: {agent_names}")
            with open(f"{self.save_path}/log.txt", "a+") as f:
                f.write("==========monitoring==========\n")
                f.write(f"target agents: {agent_names}\n")

        return target_agents_idx

    def critic(self, obs, verbose=False):
        short_term_plan = self.memory_library.retrieve_latest_short_term_plan()
        return self.critic_agent.critic(short_term_plan, obs, verbose=verbose)

    # TODO: ここでworkerのplanを生成する
    def generate_worker_plan(self, obs, task_info):
        return None

    def run(self, agent, obs, code_info, task_info=None, verbose=False) -> str:
        # critic
        plan_is_success = False
        critic_info = None
        if verbose:
            print("==========critic==========")
            with open(f"{self.save_path}/log.txt", "a+") as f:
                f.write("==========critic==========\n")

        # TODO: criticの入出力、prompt
        next_step, plan_is_success, critic_info = self.critic(obs, verbose=verbose)

        if verbose:
            print("critic info: " + critic_info)
            print("==========================\n")
            with open(f"{self.save_path}/log.txt", "a+") as f:
                f.write("critic info: " + critic_info + "\n")
                f.write("==========================\n")

        if plan_is_success:
            return None

        plan = self.generate_worker_plan(obs, task_info)

        return plan
