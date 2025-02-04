from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from ... import Action
from ..prompt_template import load_prompt


class ActionInfo(BaseModel):
    Explain: str = Field(description="Explain")
    Plan: str = Field(description="Plan")
    Code: str = Field(description="Code")


class ActionAgent:
    def __init__(
        self,
        model_name="gpt-4-turbo",
        base_url=None,
        max_tokens=1024,
        temperature=0,
        save_path="./save",
        role='default'
    ):

        model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # self.model = model
        # self.parser = JsonOutputParser(pydantic_object=ActionInfo)
        parser = JsonOutputParser(pydantic_object=ActionInfo)
        self.chain = model | parser
        self.save_path = save_path
        self.role = role

    def render_system_message(self):
        # system_template = load_prompt("high_level_action_template")
        # # FIXME: fix program loading
        # programs = load_prompt("programs")
        # code_example = load_prompt("code_example")
        # response_format = load_prompt("high_level_action_response_format")

        system_template = load_prompt("high_level_action_template", role=self.role)
        # FIXME: fix program loading
        programs = load_prompt("programs", role=self.role)
        code_example = load_prompt("code_example", role=self.role)
        response_format = load_prompt("high_level_action_response_format", role=self.role)

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(
            programs=programs,
            code_example=code_example,
            response_format=response_format,
        )
        assert isinstance(system_message, SystemMessage)
        return system_message

    def render_human_message(
        self, obs, short_term_plan, code_info=None, critic_info=None
    ):
        content = []
        text = ""
        text += f"short-term plan: {short_term_plan}\n"
        text += f"observation: {str(obs).replace(' ', '')}\n"
        if code_info is not None:
            text += f"code info: {code_info}\n"
        if critic_info is not None:
            text += f"critic info: {critic_info}\n"
        content.append(
            {
                "type": "text",
                "text": text,
            }
        )
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
            pass
        human_message = HumanMessage(content=content)
        return human_message

    def execute(self, obs, short_term_plan, max_tries=3, verbose=False):
        system_message = self.render_system_message()
        human_message = self.render_human_message(obs, short_term_plan)

        message = [system_message, human_message]

        try:
            response = self.chain.invoke(message)
            # response = self.model.invoke(message) 
            # response.content = response.content.replace("RESPONSE FORMAT:\n", "")
            # print(response)
            # response = self.parser.invoke(response)
        except:
            max_tries -= 1
            if max_tries > 0:
                return self.execute(obs, short_term_plan, max_tries, verbose)
            else:
                print("parse failed")
                # return {"type": Action.RESUME, "code": ""}
                return Action(type=Action.RESUME, code="")

        if verbose:
            print(f"\033[31m****Action Agent****\n{response}\033[0m")
            with open(f"{self.save_path}/log.txt", "a+") as f:
                f.write(f"****Action Agent****\n{response}\n")
                
        # act = {"type": Action.NEW, "code": response["Code"]}
        act = Action(type=Action.NEW, code=response["Code"])
        return act

    def retry(self, obs, short_term_plan, code_info, max_tries=3, verbose=False):
        system_message = self.render_system_message()
        human_message = self.render_human_message(obs, short_term_plan, code_info)

        message = [system_message, human_message]

        try:
            # response = self.chain.invoke(message)
            response = self.model.invoke(message) 
            response.content = response.content.replace("RESPONSE FORMAT:\n", "")
            # print(response)
            response = self.parser.invoke(response)
        except:
            max_tries -= 1
            if max_tries > 0:
                return self.retry(obs, short_term_plan, code_info, max_tries, verbose)
            else:
                print("parse failed")
                # return {"type": Action.RESUME, "code": ""}
                return Action(type=Action.RESUME, code="")

        if verbose:
            print(f"\033[31m****Action Agent****\n{response}\033[0m")
            with open(f"{self.save_path}/log.txt", "a+") as f:
                f.write(f"****Action Agent****\n{response}\n")

        # act = {"type": Action.NEW, "code": response["Code"]}
        act = Action(type=Action.NEW, code=response["Code"])
        return act

    def redo(self, obs, short_term_plan, critic_info, max_tries=3, verbose=False):
        system_message = self.render_system_message()
        human_message = self.render_human_message(
            obs, short_term_plan, critic_info=critic_info
        )

        message = [system_message, human_message]

        try:
            # response = self.chain.invoke(message)
            response = self.model.invoke(message) 
            response.content = response.content.replace("RESPONSE FORMAT:\n", "")
            # print(response)
            response = self.parser.invoke(response)
        except:
            max_tries -= 1
            if max_tries > 0:
                return self.redo(obs, short_term_plan, critic_info, max_tries, verbose)
            else:
                print("parse failed")
                # return {"type": Action.RESUME, "code": ""}
                return Action(type=Action.RESUME, code="")

        if verbose:
            print(f"\033[31m****Action Agent****\n{response}\033[0m")
            with open(f"{self.save_path}/log.txt", "a+") as f:
                f.write(f"****Action Agent****\n{response}\n")

        # act = {"type": Action.NEW, "code": response["Code"]}
        act = Action(type=Action.NEW, code=response["Code"])
        return act
