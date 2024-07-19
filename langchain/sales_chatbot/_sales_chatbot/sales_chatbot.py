import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(base_url="https://apikeyplus.com/v1"), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, base_url="https://apikeyplus.com/v1")
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.6}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def special_sales_chat(message, history):
    template = (
        """你是 OOCL 远洋航运订舱 的客服专员，你的回复贴近 OOCL 客服专员的回答风格。
        你将只回答关于 OOCL 远洋航运订舱业务相关的问题，当话题超出范围时，请礼貌地回避这个问题。
        你需要使用提供的上下文来回答问题。如果上下文中没有任何信息可用来回答问题，请礼貌地向客户解释你无法回答这个问题。
        你的回答必须时以“我”开头
        """
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # 将历史消息转换为 LangChain 的消息格式
    history_messages = [msg for msg in history]

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 使用 System、History 和 Human 角色的提示模板构造 ChatPromptTemplate
    # 注意：MessagesPlaceholder 用于在运行时动态插入历史消息
    chat_prompt_template = ChatPromptTemplate.from_messages([
        system_message_prompt,
        MessagesPlaceholder(variable_name="history"),
        human_message_prompt
    ])

    # 为了翻译结果的稳定性，将 temperature 设置为 0
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, base_url="https://apikeyplus.com/v1")
    chain = LLMChain(llm=chat, prompt=chat_prompt_template, verbose=True)

    # 在运行链时，传递历史消息
    response = chain.run({
        "text": message,
        "history": history_messages  # 这里传递的是转换后的历史消息
    })

    return response

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = False

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果

    try:
        if ans["source_documents"] or enable_chat:
            print(f"[result]{ans['result']}")
            print(f"[source_documents]{ans['source_documents']}")
            return ans["result"]
        # 否则输出套路话术
        else:
            return special_sales_chat(message, history)
    except Exception as e:
        return special_sales_chat(message, history)
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="航运仓位销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 航运仓位销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
