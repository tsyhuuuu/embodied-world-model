from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from .prompt_template import load_prompt


class LongtermPlan(BaseModel):
    reasoning: str = Field(description="reasoning")
    long_term_plan: str = Field(description="long-term-plan")


class LongtermPlanner:
    """
    Long-term Planner
    Generate a long-term plan for the ultimate goal.
    """

    def __init__(self, model_name="gpt-4-turbo", base_url=None, max_tokens=1024, temperature=0, vision=True):
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.vision = vision

        vlm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        parser = JsonOutputParser(pydantic_object=LongtermPlan)
        self.chain = vlm | parser

    def render_system_message(self):
        system_prompt = load_prompt("generate_long_term_plan")
        return SystemMessage(content=system_prompt)

    def render_human_message(self, obs, task_info, verbose=False):
        content = []
        text = ""
        text += f"Task: {task_info}\n"
        text += f"Observation: {str(obs)}\n"
        content.append({"type": "text", "text": text})
        if self.vision:
            try:
                image_base64 = obs["rgb_base64"]
                if image_base64 != "":
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "auto",
                            },
                        }
                    )
            except:
                print("No image in observation")
                pass

            try:
                blueprint_base64 = task_info["rgb_base64"]
                if blueprint_base64 and blueprint_base64 != "":
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{blueprint_base64}",
                                "detail": "auto",
                            },
                        }
                    )
            except:
                pass

        human_message = HumanMessage(content=content)

        if verbose:
            human_message.pretty_print()

        return human_message

    def plan(self, obs, task_info):
        system_message = self.render_system_message()
        human_message = self.render_human_message(obs, task_info)

        message = [system_message, human_message]

        long_term_plan = self.chain.invoke(message)
        print(f"\033[31m****Long-term Planner****\n{long_term_plan}\033[0m")

        return long_term_plan


if __name__ == "__main__":
    longterm_planner = LongtermPlanner()
    longterm_planner.render_human_message(obs={"rgb_base64": ""}, task_info="build a house", verbose=True)
