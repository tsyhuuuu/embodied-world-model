from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from ..prompt_template import load_prompt


class skillInfo(BaseModel):
    name: str = Field(description="name")
    description: str = Field(description="description")


class SkillManager:
    def __init__(
        self,
        model_name="gpt-4-turbo",
        base_url=None,
        max_tokens=256,
        temperature=0,
        role='default'
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        parser = JsonOutputParser(pydantic_object=skillInfo)
        self.chain = model | parser
        self.role = role

    def render_system_message(self):
        # prompt = load_prompt("generate_skill_description")
        prompt = load_prompt("generate_skill_description", role=self.role)
        return SystemMessage(content=prompt)

    def render_human_message(self, code_info):
        code = code_info["last_code"]
        human_message = HumanMessage(content=code)
        return human_message

    def generate_skill_info(self, code_info):
        system_message = self.render_system_message()
        human_message = self.render_human_message(code_info)

        message = [system_message, human_message]

        skill_info = self.chain.invoke(message)
        print(f"\033[31m****Skill Manager****\n{skill_info}\033[0m")

        return skill_info
