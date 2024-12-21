import mineland
from mineland.alex import Alex

import time
import numpy as np
import matplotlib.pyplot as plt

# from openai import OpenAI


class MultiAgentMineland(object):

    def __init__(self, task_id, agents_num, agents_name, agents_llm, agents_vlm, agents_personality, enable_low_level_action):
        
        """
            Agent と Environment を初期化

            パラメータ:
                task_id     (str)          タスク
                agents_num  (int)          エージェントの数
                agents_name (str)          エージェントの名前
                agents_llm  (Any)          エージェントのチャット・メモリ用LLMの設定
                agents_vlm  (Any)          エージェントの視覚解析用LLMの設定
                agents_personality (Any)   エージェントの性格属性

        """

        self.task_id            = task_id              # タスクを初期化
        self.agents_num         = agents_num           # エージェントの数を初期化
        self.agents_name        = agents_name          # エージェントの名前を初期化
        self.agents_llm         = agents_llm           # エージェントのチャット・メモリ用LLMの設定を初期化
        self.agents_vlm         = agents_vlm           # エージェントの視覚解析用LLMの設定を初期化
        self.agents_personality = agents_personality   # エージェントの性格属性を初期化
        
        # initialize agents
        self.agents = [
            Alex(personality=self.agents_personality[agent_id],    # Alex configuration
                 llm_model_name=self.agents_llm[agent_id],
                 vlm_model_name=self.agents_vlm[agent_id],
                 bot_name=self.agents_name[agent_id],
                 temperature=0.1)
            for agent_id in range(agents_num)
        ]

        # initialize environment
        self.mland = mineland.make(
            task_id=task_id,
            agents_count=agents_num,
            ticks_per_step=10,
            enable_low_level_action=enable_low_level_action
        )

        n_cols = int(np.ceil(np.sqrt(agents_num)))
        n_rows = int(np.ceil(agents_num / n_cols))
        self.axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))[1].flatten()
    

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

        plt.pause(0.1)


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
    

    def get_intelligent_actions(self, obs, code_info, done, task_info):
        """
          エージェントはLLMで生成した行動を取る
        """
        actions = []
        for idx, agent in enumerate(self.agents):
            action = agent.run(obs[idx], code_info[idx], done, task_info, verbose=True)
            actions.append(action)

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
        for _ in range(30):  # 5000
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
        for i in range(2):  # 5000
            if i > 0 and i % 10 == 0:
                print("task_info: ", task_info)
            if i == 0:
                # skip the first step which includes respawn events and other initializations
                actions = [mineland.Action(type=mineland.Action.RESUME, code="") for _ in range(self.agents_num)]
            else:
                # run agents
                actions = self.get_intelligent_actions(obs, code_info, done, task_info)

            obs, code_info, event, done, task_info = self.mland.step(action=actions)
            self.show_agent_perspectives(obs=obs)

            code_infos.append(code_info)
            task_infos.append(task_info)
        
        end_time = time.time()
        self.mland.close()

        task_duration = end_time - start_time
        task_summary = self.summarize(code_infos, task_infos)

        return (task_duration, task_summary)



if __name__ == "__main__":
    
    """ 1. パラメータ設計 """
    NUM_AGENTS = 2                                                              # <-- REVISE HERE
    CONFIGS = {
        'task_id':            'playground',                                     # <-- REVISE HERE
        'agents_num':         NUM_AGENTS,
        'agents_name':        [f"MineflayerBot{i}" for i in range(NUM_AGENTS)], # <-- REVISE HERE
        'agents_llm':         [f'gpt-4o mini' for i in range(NUM_AGENTS)],      # <-- REVISE HERE
        'agents_vlm':         [f'gpt-4o' for i in range(NUM_AGENTS)],           # <-- REVISE HERE
        'agents_personality': [None for i in range(NUM_AGENTS)],                # <-- REVISE HERE
        'enable_low_level_action': False                                        # <-- REVISE HERE

    }


    """ 2. 指定したタスクでマルチエージェント """
    plt.ion()
    mam = MultiAgentMineland(**CONFIGS)
    # summary = mam.test()         # enable_low_level_action: True
    summary = mam.run_task()     # enable_low_level_action: False
    plt.ioff()
    

    """ 3. 結果をまとめ・解析（まとめ）[to be continue] """
    # print("===== TASK DURATION =====")
    # print(summary[0])

    # print("===== TASK  SUMMARY =====")
    # print(summary[1])