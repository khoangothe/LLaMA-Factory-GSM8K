from typing_extensions import TypedDict
from langchain_community.llms import VLLM

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import AnyMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()

CORE_PROMPT = ChatPromptTemplate.from_template( "Please extract core question from this question, only return the most comprehensive and detailed one!\nQuestion: {question}\nOnly return the core question, nothing else")

HINT_PROMPT = ChatPromptTemplate.from_template("{question}\nNote: Please extract the most useful information related to the core question ({core_question}), only extract the most useful information, and list them one by one! Only return the hint, do not solve the problem")

FINAL_PROMPT = ChatPromptTemplate.from_template("question:\n{question}\nHint:\n{key_information}\nCore Question:\n{core_question}\nPlease fully understand the Hint and question information and integrated comprehensively, and then give back the answer and calculation!")

class State(TypedDict):
    question: AnyMessage
    core_question : AnyMessage
    key_information: AnyMessage
    final_answer :  AnyMessage

class CoreQuestionNode:
    def __init__(self, llm: Runnable):
        self.llm = llm
        self.chain = CORE_PROMPT | llm

    def __call__(self, state : State , config: RunnableConfig):
        configuration = config.get("configurable", {})
        core_question = self.chain.invoke(state)
        print(core_question)
        return {"core_question": core_question}


class HintNode:
    def __init__(self, llm: Runnable):
        self.llm = llm
        self.chain = HINT_PROMPT | llm

    def __call__(self, state : State , config: RunnableConfig):
        configuration = config.get("configurable", {})
        key_information = self.chain.invoke(state)
        print(key_information)
        return {"key_information": key_information}

class FinalResponse:
    def __init__(self, llm: Runnable):
        self.llm = llm
        self.chain = FINAL_PROMPT | llm

    def __call__(self, state : State , config: RunnableConfig):
        configuration = config.get("configurable", {})
        response = self.chain.invoke(state)
        return {"final_answer": response}

class QwenDupAgent:
    def __init__(self, llm):
        self.core = CoreQuestionNode(llm)
        self.hint = HintNode(llm)
        self.final = FinalResponse(llm)

    def get_agent_graph(self):
        builder = StateGraph(State)
        builder.add_node("core",  self.core)
        builder.add_node("hint",  self.hint)
        builder.add_node("final",  self.final)

        builder.add_edge(START, "core")
        builder.add_edge("core",  "hint")
        builder.add_edge("hint",  "final")
        builder.add_edge("final",  "__end__")
        graph = builder.compile()
        return graph



if __name__ == "__main__":
    llm = VLLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True,  # mandatory for hf models
        top_p=0.95,
        temperature=0.1,
    )
    graph = QwenDupAgent(llm).get_agent_graph()
    graph.invoke( {"question" : "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"})


