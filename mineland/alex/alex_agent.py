import time

from .. import Action
from .action.action_agent import *
from .brain.associative_memory import *
from .brain.memory_library import *
from .critic.critic_agent import *
from .self_check.self_check_agent import *


class Alex:
    def __init__(
        self,
        llm_model_name="gpt-4-turbo",
        vlm_model_name="gpt-4-turbo",
        base_url=None,
        max_tokens=512,
        temperature=0,
        save_path="./storage",
        load_path="./load",
        FAILED_TIMES_LIMIT=3,
        bot_name="Alex",
        personality="None",
        vision=True,
        # ↓ BGI2スコアを受け取るパラメータを用意
        bgi2_scores=None,
    ):
        """
        bgi2_scores は辞書形式を想定:
        {
            "O": 0.0〜1.0,  # Openness
            "C": 0.0〜1.0,  # Conscientiousness
            "E": 0.0〜1.0,  # Extraversion
            "A": 0.0〜1.0,  # Agreeableness
            "N": 0.0〜1.0   # Neuroticism
        }
        """

        # personality が None または "None" で、かつ bgi2_scores が渡されていれば、自動的に性格テキストを生成
        if (personality is None or personality == "None") and bgi2_scores is not None:
            personality = self.set_personality_from_bgi2(bgi2_scores)

        self.personality = personality
        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.save_path = save_path + "/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.load_path = load_path
        self.vision = vision
        self.bot_name = bot_name
        self.FAILED_TIMES_LIMIT = FAILED_TIMES_LIMIT

        print(f"save_path: {self.save_path}")
        print(f"Personality set to: {self.personality}")

        self.self_check_agent = SelfCheckAgent(
            FAILED_TIMES_LIMIT=self.FAILED_TIMES_LIMIT,
            save_path=self.save_path,
        )
        self.critic_agent = CriticAgent(
            FAILED_TIMES_LIMIT=self.FAILED_TIMES_LIMIT,
            mode="auto",
            model_name=self.vlm_model_name,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            save_path=self.save_path,
            vision=self.vision,
        )
        self.memory_library = MemoryLibrary(
            model_name=self.vlm_model_name,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            save_path=self.save_path,
            load_path=self.load_path,
            personality=self.personality,
            bot_name=self.bot_name,
            vision=self.vision,
        )
        self.associative_memory = AssociativeMemory(
            model_name=self.vlm_model_name,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            save_path=self.save_path,
            personality=self.personality,
            vision=self.vision,
        )
        self.action_agent = ActionAgent(
            model_name=self.vlm_model_name,
            base_url=self.base_url,
            max_tokens=self.max_tokens * 3,
            temperature=self.temperature,
            save_path=self.save_path,
        )

    def set_personality_from_bgi2(self, bgi2_scores: dict) -> str:
        """
        BGI2のスコアをLow / Medium / Highに区分し、各特性ごとに対応するテキストを返す。
        例として以下のように区分している:
        0.00〜0.32: Low
        0.33〜0.65: Medium
        0.66〜1.00: High
        ※スコアが範囲外の場合には適宜クリップされる想定
        """

        # 各特性のレベル別テキストを定義
        trait_text_map = {
            "O": {
                "Low": "Prefers familiar routines and experiences over novelty",
                "Medium": "Moderately open to new ideas and experiences",
                "High": "Highly imaginative, curious, and adventurous in thinking",
            },
            "C": {
                "Low": "Tends to be disorganized or spontaneous in scheduling",
                "Medium": "Somewhat reliable, balancing planning with flexibility",
                "High": "Very organized, disciplined, and goal-oriented",
            },
            "E": {
                "Low": "Reserved, quiet, and introspective",
                "Medium": "Moderately outgoing and sociable",
                "High": "Highly energetic, talkative, and enjoys social settings",
            },
            "A": {
                "Low": "Competitive or critical in interactions with others",
                "Medium": "Generally cooperative, but can assert needs when necessary",
                "High": "Very empathetic, cooperative, and trusting",
            },
            "N": {
                "Low": "Emotionally stable and calm under pressure",
                "Medium": "Somewhat sensitive to stress, but usually remains composed",
                "High": "Sensitive and prone to experiencing stress or worry",
            },
        }

        # レベル判定用の関数
        def get_level(score: float) -> str:
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0

            if score < 0.33:
                return "Low"
            elif score < 0.66:
                return "Medium"
            else:
                return "High"

        # 出力用文字列を構築
        personality_descriptions = []
        for trait_code in ["O", "C", "E", "A", "N"]:
            score = bgi2_scores.get(trait_code, 0.0)
            level = get_level(score)
            text_for_trait = trait_text_map.get(trait_code, {}).get(level, "Undefined")

            # Openness(0.75: High) => "Highly imaginative..."
            # のような形式でまとめる
            personality_descriptions.append(f"{trait_code}({score:.2f}, {level}): {text_for_trait}")

        # 全特性のテキストをカンマ区切りで連結して personality として返す
        # 実際には改行区切りにしたり、文章形式にしたりも可
        return ", ".join(personality_descriptions)

    def self_check(self, obs, code_info=None, done=None, task_info=None):
        return self.self_check_agent.self_check(
            obs, code_info, done, task_info, associative_memory=self.associative_memory
        )

    def critic(self, obs, verbose=False):
        short_term_plan = self.memory_library.retrieve_latest_short_term_plan()
        return self.critic_agent.critic(short_term_plan, obs, verbose=verbose)

    def perceive(
        self,
        obs,
        plan_is_success,
        critic_info=None,
        code_info=None,
        vision=False,
        verbose=False,
    ):
        self.memory_library.perceive(obs, plan_is_success, critic_info, code_info, vision=vision, verbose=verbose)

    def retrieve(self, obs, verbose=False):
        retrieved = self.memory_library.retrieve(obs, verbose)
        return retrieved

    def plan(self, obs, task_info, retrieved, verbose=False):
        short_term_plan = self.associative_memory.plan(obs, task_info, retrieved, verbose=verbose)
        self.memory_library.add_short_term_plan(short_term_plan, verbose=verbose)
        return short_term_plan

    def execute(self, obs, description, code_info=None, critic_info=None, verbose=False):
        if description == "Code Unfinished":
            return Action(type=Action.RESUME, code="")

        short_term_plan = self.memory_library.retrieve_latest_short_term_plan()
        if description in ["Code Failed", "Code Error"]:
            return self.action_agent.retry(obs, short_term_plan, code_info, verbose=verbose)
        if description == "redo":
            return self.action_agent.redo(obs, short_term_plan, critic_info, verbose=verbose)
        return self.action_agent.execute(obs, short_term_plan, verbose=verbose)

    def run(self, obs, code_info=None, done=None, task_info=None, verbose=False):
        # 1. self check
        if verbose:
            print("==========self check==========")

        next_step, description = self.self_check(obs, code_info, done, task_info)

        if next_step is None:  # タスク完了の場合
            print(description)
            return None

        if verbose:
            print("next step after self check: " + next_step)
            print("description: " + description)
            print("==============================\n")
            if next_step != "action":
                with open(f"{self.save_path}/log.txt", "a+") as f:
                    f.write("==========self check==========\n")
                    f.write("next step after self check: " + next_step + "\n")
                    f.write("description: " + description + "\n")
                    f.write("==============================\n")

        # 2. critic
        plan_is_success = False
        critic_info = None
        if next_step == "critic":
            if verbose:
                print("==========critic==========")
                with open(f"{self.save_path}/log.txt", "a+") as f:
                    f.write("==========critic==========\n")

            next_step, plan_is_success, critic_info = self.critic(obs, verbose=verbose)

            if next_step == "action":
                description = "redo"

            if verbose:
                print("next step after critic: " + next_step)
                print("critic info: " + critic_info)
                print("==========================\n")
                with open(f"{self.save_path}/log.txt", "a+") as f:
                    f.write("next step after critic: " + next_step + "\n")
                    f.write("critic info: " + critic_info + "\n")
                    f.write("==========================\n")

        # 3. brain
        if next_step == "action":
            self.perceive(
                obs,
                plan_is_success,
                critic_info=None,
                code_info=None,
                vision=False,
                verbose=False,
            )

        if next_step == "brain":
            if verbose:
                print("==========brain==========")
                with open(f"{self.save_path}/log.txt", "a+") as f:
                    f.write("==========brain==========\n")

            if description == "Code Failed":
                critic_info = "action failed, maybe the plan is too difficult. please change to an easier plan."

            self.perceive(
                obs,
                plan_is_success,
                critic_info,
                code_info,
                vision=True,
                verbose=verbose,
            )
            self.memory_library.generate_long_term_plan(obs, task_info)
            retrieved = self.retrieve(obs, verbose=verbose)
            self.plan(obs, task_info, retrieved, verbose=verbose)

            next_step = "action"
            description = "execute the plan"

            if verbose:
                print("next step after brain: " + next_step)
                print("description: " + description)
                print("========================\n")
                with open(f"{self.save_path}/log.txt", "a+") as f:
                    f.write("next step after brain: " + next_step + "\n")
                    f.write("description: " + description + "\n")
                    f.write("========================\n")

        # 4. action
        if next_step == "action":
            if verbose:
                print("==========action==========")
                if description != "Code Unfinished":
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write("==========action==========\n")

            act = self.execute(
                obs,
                description,
                code_info=code_info,
                critic_info=critic_info,
                verbose=verbose,
            )

            if verbose:
                print("==========================\n")
                if description != "Code Unfinished":
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write("==========================\n\n\n")

            return act
