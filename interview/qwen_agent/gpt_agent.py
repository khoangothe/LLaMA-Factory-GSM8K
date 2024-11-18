from typing_extensions import TypedDict
from langchain_community.llms import VLLM

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import AnyMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()

CORE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Please extract core question, only the most comprehensive and detailed one!
            {question}
            """
        ),
    ]
)

HINT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            {question}
            Note: Please extract the most useful information related to the core question ({core_question}), only extract the most useful information, and list them one by one!
            """
        ),
    ]
)

FINAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            question:
            {question}
            Hint: 
            {key_information} 
            Core Question:
            {core_question}
            Please fully understand the Hint and question information and integrated comprehensively, and then give the answer carefully and give details!"
            """
        ),
    ]
)

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
        result = self.chain.invoke(state)
        core_question = result.content
        print(core_question)
        return {"core_question": core_question}


class HintNode:
    def __init__(self, llm: Runnable):
        self.llm = llm
        self.chain = HINT_PROMPT | llm

    def __call__(self, state : State , config: RunnableConfig):
        configuration = config.get("configurable", {})
        result = self.chain.invoke(state)
        key_information = result.content
        print(key_information)
        return {"key_information": key_information}

class FinalResponse:
    def __init__(self, llm: Runnable):
        self.llm = llm
        self.chain = FINAL_PROMPT | llm

    def __call__(self, state : State , config: RunnableConfig):
        configuration = config.get("configurable", {})
        result = self.chain.invoke(state)
        response = result.content
        print(response)
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

def load_azure_client() -> AzureChatOpenAI:
     return AzureChatOpenAI(
        deployment_name="gpt-4o-mini",
        temperature = 0.1      
    )

if __name__ == "__main__":
    llm = VLLM(
        model="Qwen/Qwen2.5-Coder-7B",
        trust_remote_code=True,  # mandatory for hf models
    )

    config = {}
    graph = QwenDupAgent(llm).get_agent_graph()
    graph.invoke({
        "question" : "What is 5 + 5?"
    }, config)
    
