import mineland
from mineland.alex import Alex

import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from threading import Thread



class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None):
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


class AgentGraphPattern(object):

    def __init__(self, agents, threads_num=4):
        self.agents      = agents
        self.threads_num = threads_num

        self.actions     = None
        self.agents_runs = [lambda obs, code_info, done, task_info: agent.run(obs[idx], code_info[idx], done, task_info) for idx, agent in enumerate(self.agents)]


    def reset(self):
        self.agents_runs = [lambda obs, code_info, done, task_info: agent.run(obs[idx], code_info[idx], done, task_info) for idx, agent in enumerate(self.agents)]

    def pattern_default(self, obs, code_info, done, task_info):
        self.reset()
        # Main logic
        not_all_finished = True
        agents_threads   = []
        actions_dict     = {k: None for k in self.agents_runs}
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
            agents_threads   = [t for t in agents_threads if t.is_alive()]
            not_all_finished = (None in list(actions_dict.values()))
            if not not_all_finished: 
                actions      = list(actions_dict.values())
                return actions

            time.sleep(1e-4)   # これを入れないと何故かプログラムが詰まる
    

    def pattern_random(self, obs, code_info, done, task_info, random_agents_num=4):
        self.reset()

        not_all_finished      = True
        random_agents_threads = []
        actions               = {k: None for k in self.agents_runs}
        
        random_agents_indices = sorted(random.sample(range(len(self.agents)), random_agents_num))
        random_agents         = [self.agents[i] for i in random_agents_num]
        random_agents_runs    = [lambda obs, code_info, done, task_info: agent.run(obs[idx], code_info[idx], done, task_info) for idx, agent in zip(random_agents_indices, random_agents)]

        while random_agents_runs or random_agents_threads or not_all_finished:
            # Fill the thread pool
            while len(random_agents_threads) < self.threads_num and random_agents_runs:
                random_agent_run = random_agents_runs.pop(0)
                random_agents_threads.append(CustomThread(target=random_agent_run, args=[obs, code_info, done, task_info]))
                random_agents_threads[-1].start()

            # Check completed threads
            for thread in random_agents_threads:
                if thread.is_alive():
                    continue
                actions[thread._target] = thread.join()

            # Remove completed threads
            random_agents_threads   = [t for t in random_agents_threads if t.is_alive()]
            not_all_finished = (None in list(actions.values()))
            time.sleep(1e-4)   # これを入れないと何故かプログラムが詰まる

        actions = list(actions.values())

        return actions



class MultiAgentMineland(object):

    def __init__(self, task_id, agents_num, agents_name, 
                       agents_llm, agents_vlm, base_url, 
                       agents_personality, enable_low_level_action, 
                       threads_num, action_pattern, save_video_dir
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

        self.task_id            = task_id              # タスクを初期化
        self.agents_num         = agents_num           # エージェントの数を初期化
        self.agents_name        = agents_name          # エージェントの名前を初期化
        self.agents_llm         = agents_llm           # エージェントのチャット・メモリ用LLMの設定を初期化
        self.agents_vlm         = agents_vlm           # エージェントの視覚解析用LLMの設定を初期化
        self.base_url           = base_url             # エージェント LLM のサーバーURL
        self.agents_personality = agents_personality   # エージェントの性格属性を初期化
        
        # initialize agents
        self.agents = [
            Alex(personality=self.agents_personality[agent_id],    # Alex configuration
                 llm_model_name=self.agents_llm[agent_id],
                 vlm_model_name=self.agents_vlm[agent_id],
                 base_url=self.base_url[agent_id],
                 bot_name=self.agents_name[agent_id],
                 temperature=0.1)
            for agent_id in range(agents_num)
        ]
        self.patterns = AgentGraphPattern(agents=self.agents, threads_num=threads_num)

        # initialize environment
        self.mland = mineland.make(
            task_id=task_id,
            agents_count=agents_num,
            ticks_per_step=10,
            enable_low_level_action=enable_low_level_action
        )

        n_cols    = int(np.ceil(np.sqrt(agents_num)))
        n_rows    = int(np.ceil(agents_num / n_cols))
        self.axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))[1].flatten() if agents_num > 1 else np.array([plt.subplots(n_rows, n_cols, figsize=(12, 8))[1]]) 

        # other parameters
        self.step           = 0
        self.images         = [[] for _ in range(agents_num)]
        self.action_pattern = action_pattern
        self.save_video_dir = save_video_dir


    def show_agent_perspectives(self, obs):
        """
          すべてエージェントの視覚情報を表示
        """

        agents_name = [obs[i]['name'] for i in range(self.agents_num)]
        agents_num  = len(agents_name)

        for agent_id in range(agents_num):
            self.axes[agent_id].clear()
            self.axes[agent_id].imshow(np.transpose(obs[agent_id].rgb, (1, 2, 0)))
            self.axes[agent_id].axis('off')
            self.axes[agent_id].set_title(agents_name[agent_id])
            
            if self.step % 2 == 0:
                self.images[agent_id].append(np.transpose(obs[agent_id].rgb, (1, 2, 0)))

        plt.pause(0.05)

    def images_to_video(self, video_name, fps=8):
        
        for (agent_id, images) in enumerate(self.images):
            frame = images[0]
            height, width, _ = frame.shape
            video_path = f"{self.save_video_dir}/{video_name}_{self.agents_name[agent_id]}.mp4"
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

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
    

    def get_intelligent_actions(self, obs, code_info, done, task_info, option='default'):
        """
          エージェントはLLMで生成した行動を取る
        """
        if option == 'default':
            actions = self.patterns.pattern_default(obs, code_info, done, task_info)
        if option == 'random':
            actions = self.patterns.pattern_random(obs, code_info, done, task_info, random_agents_num=4)

        return actions
    

    def summarize(self, codes_info, tasks_info):
        """
          タスク解決までの出力をもとに作成する、エージェントがどう協力しあって、タスクを解決したかをまとめた文章 [to be continue]
        """

        pass
 
    
    def test(self, option="stay"):
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
        for i in range(100):  # 5000
            self.step = i
            if option == 'stay':
                actions = self.get_resume_low_actions()
            if option == 'random':       
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


    def run_task(self):

        """
          事前に設定したパラメータをもとに、マルチエージェントを行うプログラム

          option:
          - 'stay':   agents はなんの行動も取らない
          - 'random': agents はランダムな行動も取る

        """

        obs = self.mland.reset()
        start_time = time.time()

        code_infos = []
        task_infos = []
        for i in range(100):  # 5000
            self.step = i
            if i > 0 and i % 10 == 0:
                print("task_info: ", task_info)
            if i == 0:
                # skip the first step which includes respawn events and other initializations
                actions = [mineland.Action(type=mineland.Action.RESUME, code="") for _ in range(self.agents_num)]
            else:
                # run agents
                actions = self.get_intelligent_actions(obs, code_info, done, task_info, option=self.action_pattern)
            
            obs, code_info, event, done, task_info = self.mland.step(action=actions)
            self.show_agent_perspectives(obs=obs)

            code_infos.append(code_info)
            task_infos.append(task_info)
        
        end_time = time.time()
        self.mland.close()

        task_duration = end_time - start_time
        task_summary = None   # self.summarize(code_infos, task_infos)
        self.images_to_video(self.task_id)

        return (task_duration, code_infos)


if __name__ == "__main__":

    """ 1. パラメータ設計 """
    NUM_AGENTS     = 2                        # <-- REVISE HERE
    SAVE_VIDEO_DIR = "/MY/SAVE/VIDEO/DIR"     # <-- REVISE HERE

    # 1.1. While using openai api
    # API_KEY = "MY_API_KEY"
    # os.environ["OPENAI_API_KEY"] = API_KEY
    # CONFIGS = {
    #     'task_id':                 'playground',                                       # <-- REVISE HERE ('playground')
    #     'agents_num':              NUM_AGENTS,
    #     'agents_name':             [f"MineflayerBot{i}" for i in range(NUM_AGENTS)],   # <-- REVISE HERE
    #     'agents_llm':              [f'gpt-4o mini'      for i in range(NUM_AGENTS)],   # <-- REVISE HERE
    #     'agents_vlm':              [f'gpt-4o'           for i in range(NUM_AGENTS)],   # <-- REVISE HERE
    #     'base_url':                [None                for i in range(NUM_AGENTS)],   # <-- REVISE HERE
    #     'agents_personality':      [None                for i in range(NUM_AGENTS)],   # <-- REVISE HERE
    #     'enable_low_level_action': False,                                              # <-- REVISE HERE
    #     'threads_num':             2,                                                  # <-- REVISE HERE
    #     'action_pattern':          'default',                                          # <-- REVISE HERE
    #     'save_video_dir':          SAVE_VIDEO_DIR                                    
    # }

    # 1.2. While using molmo via omniverse server
    API_KEY = "12345"
    os.environ["OPENAI_API_KEY"] = API_KEY
    BASE_URL = "https://alien-curious-smoothly.ngrok-free.app/v1"
    MODEL_NAME = "default"
    CONFIGS = {
        'task_id':                 'harvest_1_copper_ingot',                           # <-- REVISE HERE ('build_a_chicken_farm')
        'agents_num':              NUM_AGENTS,
        'agents_name':             [f"MineflayerBot{i}" for i in range(NUM_AGENTS)],   # <-- REVISE HERE
        'agents_llm':              [MODEL_NAME for i in range(NUM_AGENTS)],            # <-- REVISE HERE
        'agents_vlm':              [MODEL_NAME for i in range(NUM_AGENTS)],            # <-- REVISE HERE
        'base_url':                [BASE_URL   for i in range(NUM_AGENTS)],            # <-- REVISE HERE
        'agents_personality':      [None       for i in range(NUM_AGENTS)],            # <-- REVISE HERE
        'enable_low_level_action': False,                                              # <-- REVISE HERE
        'threads_num':             2,                                                  # <-- REVISE HERE
        'action_pattern':          'default',                                          # <-- REVISE HERE
        'save_video_dir':          SAVE_VIDEO_DIR                                    
    }

    """ 2. 指定したタスクでマルチエージェント """
    plt.ion()
    mam = MultiAgentMineland(**CONFIGS)
    # summary = mam.test(option="random")         # enable_low_level_action: True
    summary = mam.run_task()                  # enable_low_level_action: False
    plt.ioff()
    

    """ 3. 結果をまとめ・解析（まとめ）[to be continue] """
    # print("===== TASK DURATION =====")
    print(summary[0])

    # print("===== TASK  SUMMARY =====")
    print(summary[1])
