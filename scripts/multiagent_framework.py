import os
import random
import time
from pathlib import Path
from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openai
from dotenv import load_dotenv
from tqdm import tqdm

import mineland
from mineland.alex import Alex


def load_prompt(prompt_dir: str, filename: str) -> str:
    if not filename.endswith(".txt"):
        filename = prompt_dir + f"revised/{filename}.txt"
    directory = os.path.dirname(__file__)
    filepath = os.path.join(directory, filename)

    try:
        with open(filepath, "r") as f:
            prompt = f.read()
        return prompt
    except Exception as e:
        print(f"Error loading prompt from {filepath}: {e}")
        return ""


class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None):
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._return = None

    def run(self) -> None:
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


class AgentGraphPattern(object):
    def __init__(self, agents, threads_num=2, prompt_dir=""):
        self.agents = agents
        self.threads_num = threads_num

        self.actions = None
        self.agents_runs = [
            lambda obs, code_info, done, task_info: agent.run(obs[idx], code_info[idx], done, task_info)
            for idx, agent in enumerate(self.agents)
        ]
        self.agents_info = []

        self.long_term_plan = None
        self.short_term_plans = []

        self.leader_llm = openai.OpenAI()
        self.prompt_dir = prompt_dir

    def reset(self):
        self.agents_runs = [
            lambda obs, code_info, done, task_info: agent.run(obs[idx], code_info[idx], done, task_info)
            for idx, agent in enumerate(self.agents)
        ]

    def pattern_default(self, obs, code_info, done, task_info):
        actions = []
        for idx, agent in enumerate(self.agents):
            action = agent.run(obs[idx], code_info[idx], done, task_info, verbose=True)
            actions.append(action)
        return actions

    def pattern_default_multithread(self, obs, code_info, done, task_info):
        self.reset()
        # Main logic
        not_all_finished = True
        agents_threads = []
        actions_dict = {k: None for k in self.agents_runs}
        while self.agents_runs or agents_threads or not_all_finished:
            # Fill the thread pool
            while len(agents_threads) < self.threads_num and self.agents_runs:
                agent_run = self.agents_runs.pop(0)
                agents_threads.append(CustomThread(target=agent_run, args=[obs, code_info, done, task_info]))
                agents_threads[-1].start()

            # Check completed threads
            for thread in agents_threads:
                if thread.is_alive():
                    continue
                actions_dict[thread._target] = thread.join()

            # Remove completed threads
            agents_threads = [t for t in agents_threads if t.is_alive()]
            not_all_finished = None in list(actions_dict.values())
            if not not_all_finished:
                actions = list(actions_dict.values())
                print(actions)
                return actions

            time.sleep(2e-4)  # これを入れないと何故かプログラムが詰まる

    def pattern_random_multithread(self, obs, code_info, done, task_info, random_agents_num=4):
        self.reset()

        not_all_finished = True
        random_agents_threads = []
        actions_dict = {k: None for k in self.agents_runs}

        random_agents_indices = sorted(random.sample(range(len(self.agents)), random_agents_num))
        random_agents = [self.agents[i] for i in random_agents_num]
        random_agents_runs = [
            lambda obs, code_info, done, task_info: agent.run(obs[idx], code_info[idx], done, task_info)
            for idx, agent in zip(random_agents_indices, random_agents)
        ]

        while random_agents_runs or random_agents_threads or not_all_finished:
            # Fill the thread pool
            while len(random_agents_threads) < self.threads_num and random_agents_runs:
                random_agent_run = random_agents_runs.pop(0)
                random_agents_threads.append(
                    CustomThread(target=random_agent_run, args=[obs, code_info, done, task_info])
                )
                random_agents_threads[-1].start()

            # Check completed threads
            for thread in random_agents_threads:
                if thread.is_alive():
                    continue
                actions_dict[thread._target] = thread.join()

            # Remove completed threads
            random_agents_threads = [t for t in random_agents_threads if t.is_alive()]
            not_all_finished = None in list(actions_dict.values())
            if not not_all_finished:
                actions = list(actions_dict.values())
                return actions

            time.sleep(2e-4)  # これを入れないと何故かプログラムが詰まる

    def pattern_custom_default(self, obs, code_info, done, task_info):
        # 0. Shared Pool (Position, Inventory, Current Plan)
        agents_num = len(obs)
        agent_info = {"name": None, "position": None, "inventory": {}, "current_plan": None}
        actions = [None] * agents_num
        for i, o in enumerate(obs):
            agent_info["name"] = o.name
            agent_info["position"] = o.location_stats["pos"]
            agent_info["inventory"] = o.inventory_all
            try:
                agent_info["current_plan"] = self.agents[i].memory_library.short_term_plan[0]["short_term_plan"]
            except Exception:
                pass

            self.agents_info.append(agent_info)

        # 1. [Leader] Create and Manage long-term plan (Task Tree)
        # if self.long_term_plan:
        #     self.manage_long_term_plan(task_info, option='update')
        # else:
        #     self.manage_long_term_plan(task_info, option='create')

        # for agent in self.agents:
        #     agent.memory_library.long_term_plan = self.long_term_plan

        # 2. [Agents] Action Planning and Execution based on the common dynamic long-term plan
        actions = []
        for idx, agent in enumerate(self.agents):
            agent.current_progress = str(self.agents_info)
            action = agent.run(obs[idx], code_info[idx], done, task_info, verbose=True)
            actions.append(action)

        return actions

    def manage_long_term_plan(self, task_info=None, agent_info=None, option="create"):
        if option == "create":
            system_prompt = load_prompt(self.prompt_dir, "generate_long_term_plan_task_tree")
            user_prompt = ""
            user_prompt += f"Task Info: {str(task_info)}\n"
            user_prompt += f"Agents_num: {len(self.agents)}\n"

        elif option == "update":
            system_prompt = load_prompt(self.prompt_dir, "update_long_term_plan_task_tree")
            user_prompt = ""
            user_prompt += f"Agent Info: {str(agent_info)}\n"
            user_prompt += f"Previous Plan: {self.long_term_plan}"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        response = self.leader_llm.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=2048, temperature=0.7, top_p=0.9
        )

        self.long_term_plan = response.choices[0].message.content


class MultiAgentMineland(object):
    def __init__(
        self,
        task_id,
        agents_num,
        agents_name,
        agents_llm,
        agents_vlm,
        base_url,
        agents_personality,
        agents_role,
        enable_low_level_action,
        threads_num,
        action_pattern,
        save_video_dir,
        prompt_dir,
    ):
        """
        Agent と Environment を初期化

        パラメータ:
            task_id     (str)          タスク
            agents_num  (int)          エージェントの数
            agents_name (str)          エージェントの名前
            agents_llm  (Any)          エージェントのチャット・メモリ用LLMの設定
            agents_vlm  (Any)          エージェントの視覚解析用LLMの設定
            base_url    (str)          エージェント LLM のサーバーURL
            agents_personality (Any)   エージェントの性格属性
            threads_num        (int)   エージェントの行動におけるThread数
            action_pattern     (str)   エージェントの行動の方針
            save_video_dir     (str)   実験動画の保存先

        """

        self.task_id = task_id  # タスクを初期化
        self.agents_num = agents_num  # エージェントの数を初期化
        self.agents_name = agents_name  # エージェントの名前を初期化
        self.agents_llm = agents_llm  # エージェントのチャット・メモリ用LLMの設定を初期化
        self.agents_vlm = agents_vlm  # エージェントの視覚解析用LLMの設定を初期化
        self.base_url = base_url  # エージェント LLM のサーバーURL
        self.agents_personality = agents_personality  # エージェントの性格属性を初期化

        # initialize agents
        self.agents = [
            Alex(
                personality=None,  # personality は自動で set_personality_from_bgi2 したいなら None にしておく
                bgi2_scores=self.agents_personality[agent_id],  # ここで BGI2スコアを渡す
                llm_model_name=self.agents_llm[agent_id],
                vlm_model_name=self.agents_vlm[agent_id],
                base_url=self.base_url[agent_id],
                bot_name=self.agents_name[agent_id],
                temperature=0.1,
                role=agent_role,
            )
            for agent_id, agent_role in zip(range(agents_num), agents_role)
        ]

        # initialize environment
        self.mland = mineland.make(
            task_id=task_id, agents_count=agents_num, ticks_per_step=10, enable_low_level_action=enable_low_level_action
        )

        n_cols = int(np.ceil(np.sqrt(agents_num)))
        n_rows = int(np.ceil(agents_num / n_cols))
        self.axes = (
            plt.subplots(n_rows, n_cols, figsize=(12, 8))[1].flatten()
            if agents_num > 1
            else np.array([plt.subplots(n_rows, n_cols, figsize=(12, 8))[1]])
        )

        # other parameters
        self.step = 0
        self.images = [[] for _ in range(agents_num)]
        self.action_pattern = action_pattern
        self.save_video_dir = save_video_dir
        self.agent_achievements = [{"name": None, "inventory": {}, "achievements": []}] * agents_num

        # structure of the multi-agents
        self.patterns = AgentGraphPattern(agents=self.agents, threads_num=threads_num, prompt_dir=prompt_dir)

    def show_agent_perspectives(self, obs):
        """
        すべてエージェントの視覚情報を表示
        """

        agents_name = [obs[i]["name"] for i in range(self.agents_num)]
        agents_num = len(agents_name)

        for agent_id in range(agents_num):
            self.axes[agent_id].clear()
            self.axes[agent_id].imshow(np.transpose(obs[agent_id].rgb, (1, 2, 0)))
            self.axes[agent_id].axis("off")
            self.axes[agent_id].set_title(agents_name[agent_id])

            if self.step % 1 == 0:
                self.images[agent_id].append(np.transpose(obs[agent_id].rgb, (1, 2, 0)))

        plt.pause(2e-4)

    def images_to_video(self, video_name, fps=8):
        for agent_id, images in enumerate(self.images):
            frame = images[0]
            height, width, _ = frame.shape
            video_path = f"{self.save_video_dir}/{video_name}_{self.agents_name[agent_id]}.mp4"
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            for image in tqdm(images):
                video.write(image)

            video.release()

    def get_resume_low_actions(self):
        """
        テスト用のプログラム
        エージェントは何の行動も取らない
        enable_low_level_action==Trueのときのみ実行可能
        """
        return mineland.LowLevelAction.no_op(self.agents_num)

    def get_random_low_actions(self):
        """
        テスト用のプログラム
        エージェントはランダムの行動を取る
        enable_low_level_action==Trueのときのみ実行可能
        """
        return mineland.LowLevelAction.random_op(self.agents_num)

    def get_intelligent_actions(self, obs, code_info, done, task_info, option="default"):
        """
        エージェントはLLMで生成した行動を取る
        """
        if option == "default":
            actions = self.patterns.pattern_default(obs, code_info, done, task_info)
        elif option == "custom_roles_default":
            actions = self.patterns.pattern_custom_default(obs, code_info, done, task_info)
        elif option == "default_multithread":
            actions = self.patterns.pattern_default_multithread(obs, code_info, done, task_info)
        elif option == "random_multithread":
            actions = self.patterns.pattern_random_multithread(obs, code_info, done, task_info, random_agents_num=4)
        else:
            raise ValueError(f"Invalid option: {option}")

        return actions

    def summarize(self):
        """
        タスク解決までの出力をもとに作成する、エージェントがどう協力しあって、タスクを解決したかをまとめた文章 [to be continue]

        output:
        - agent_achievements: エージェントの実績のまとめ
        """
        return self.agent_achievements

    def test(self, total_steps, option="stay"):
        """
        LLMを使用せず、実験の流れをテストするためのプログラム

        option:
        - 'stay':   agents はなんの行動も取らない
        - 'random': agents はランダムな行動も取る

        """

        obs = self.mland.reset()
        start_time = time.time()

        code_infos = []
        task_infos = []
        for i in range(total_steps):  # 5000
            print(f"===== i = {i} =====")
            self.step = i
            if option == "stay":
                actions = self.get_resume_low_actions()
            elif option == "random":
                actions = self.get_random_low_actions()

            obs, code_info, event, done, task_info = self.mland.step(action=actions)
            self.show_agent_perspectives(obs=obs)

            code_infos.append(code_info)
            task_infos.append(task_info)

        end_time = time.time()
        self.mland.close()

        task_duration = end_time - start_time
        self.images_to_video(self.task_id)

        return (task_duration, code_infos, task_infos)

    def run_task(self, total_steps):
        """
        事前に設定したパラメータをもとに、マルチエージェントを行うプログラム

        option:
        - 'stay':   agents はなんの行動も取らない
        - 'random': agents はランダムな行動も取る

        """

        obs = self.mland.reset()
        start_time = time.time()

        for step in range(total_steps):
            print("=" * 15 + f" step = {step} " + "=" * 15)
            self.step = step

            if step == 0:
                # skip the first step which includes respawn events and other initializations
                actions = [mineland.Action(type=mineland.Action.RESUME, code="") for _ in range(self.agents_num)]
            else:
                # run agents
                actions = self.get_intelligent_actions(obs, code_info, done, task_info, option=self.action_pattern)

            obs, code_info, event, done, task_info = self.mland.step(action=actions)
            self.show_agent_perspectives(obs=obs)

            if step % 2 == 1:
                for i, agent in enumerate(self.agents):
                    self.agent_achievements[i]["name"] = obs[i]["name"]
                    self.agent_achievements[i]["inventory"] = obs[i]["inventory_all"]
                    try:
                        if (
                            agent.memory_library.short_term_plan[0]["short_term_plan"]
                            not in self.agent_achievements[i]["achievements"]
                        ):
                            self.agent_achievements[i]["achievements"].append(
                                agent.memory_library.short_term_plan[0]["short_term_plan"]
                            )
                    except Exception:
                        pass

        end_time = time.time()
        self.mland.close()

        task_duration = end_time - start_time
        task_summary = self.summarize()
        self.images_to_video(self.task_id)

        return (task_duration, task_summary)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    env_path = os.path.join(current_dir.parent, ".env")
    load_dotenv(dotenv_path=env_path)

    """ 1. パラメータ設計 """
    NUM_AGENTS = 2  # <-- REVISE HERE
    TOTAL_STEPS = 50  # <-- REVISE HERE
    SAVE_VIDEO_DIR = os.path.join(current_dir.parent, "my_scripts/images/task4")
    PROMPT_DIR = os.path.join(current_dir.parent, "alex/prompt_template/")

    # 1.1. While using openai api
    CONFIGS = {
        "task_id": "techtree_1_iron_pickaxe",  # <-- REVISE HERE ('playground')
        "agents_num": NUM_AGENTS,
        "agents_name": [f"Bot{i}" for i in range(NUM_AGENTS)],  # <-- REVISE HERE
        "agents_llm": ["gpt-4o mini" for i in range(NUM_AGENTS)],  # <-- REVISE HERE
        "agents_vlm": ["gpt-4o" for i in range(NUM_AGENTS)],  # <-- REVISE HERE
        "base_url": [None for i in range(NUM_AGENTS)],  # <-- REVISE HERE
        "agents_personality": [None for i in range(NUM_AGENTS)],  # <-- REVISE HERE
        "agents_role": ["leader"] + ["default" for i in range(NUM_AGENTS - 1)],  # <-- REVISE HERE
        "enable_low_level_action": False,  # <-- REVISE HERE
        "threads_num": NUM_AGENTS,
        "action_pattern": "custom_roles_default",  # <-- REVISE HERE
        "save_video_dir": SAVE_VIDEO_DIR,
        "prompt_dir": PROMPT_DIR,
    }

    # 1.2. While using Qwen2-VL-7B-instruct via omniverse server
    # BASE_URL = "https://alien-curious-smoothly.ngrok-free.app/v1"
    # MODEL_NAME = "default"
    # CONFIGS = {
    #     'task_id':                 'techtree_1_golden_pickaxe',                         # <-- REVISE HERE ('build_a_chicken_farm')
    #     'agents_num':              NUM_AGENTS,
    #     'agents_name':             [f"Bot{i}" for i in range(NUM_AGENTS)],              # <-- REVISE HERE
    #     'agents_llm':              [MODEL_NAME for i in range(NUM_AGENTS)],             # <-- REVISE HERE
    #     'agents_vlm':              [MODEL_NAME for i in range(NUM_AGENTS)],             # <-- REVISE HERE
    #     'base_url':                [BASE_URL   for i in range(NUM_AGENTS)],             # <-- REVISE HERE
    #     'agents_personality':      [None       for i in range(NUM_AGENTS)],             # <-- REVISE HERE
    #     'agents_role':             ['leader']+['default' for i in range(NUM_AGENTS-1)], # <-- REVISE HERE
    #     'enable_low_level_action': False,                                               # <-- REVISE HERE
    #     'threads_num':             3,                                                   # <-- REVISE HERE
    #     'action_pattern':          'custom_roles_default',                              # <-- REVISE HERE
    #     'save_video_dir':          SAVE_VIDEO_DIR,
    #     'prompt_dir':              PROMPT_DIR
    # }

    """ 2. 指定したタスクでマルチエージェント """
    plt.ion()
    mam = MultiAgentMineland(**CONFIGS)
    # summary = mam.test(TOTAL_STEPS, option="random")         # enable_low_level_action: True
    summary = mam.run_task(TOTAL_STEPS)  # enable_low_level_action: False
    plt.ioff()

    """ 3. 結果をまとめ・解析（まとめ）[to be continue] """
    # print("===== TASK DURATION =====")
    print(summary[0])

    # print("===== TASK  SUMMARY =====")
    print(summary[1])
