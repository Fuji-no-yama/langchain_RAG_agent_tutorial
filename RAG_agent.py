import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter

import langchain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import load_tools, AgentExecutor, create_openai_tools_agent
from add_document import initialize_vectorstore
from langchain.tools.retriever import create_retriever_tool

load_dotenv(override=True) #.envファイルの中身を環境変数に設定
langchain.debug = True

def create_QAchain(): #ここの部分がどうすればいいのか本当にわからない
    system_prompt1 = ChatPromptTemplate.from_template(
        "一連の会話であるhistoryとquestionを用いて, questionをweb検索に適した正確な質問にしてください。"\
        "必要ない場合はquestionをそのまま出力してください。\n"\
        "######"\
        "question:{question}\n"\
        "history:{history}"
    )
    system_prompt2 = ChatPromptTemplate.from_template(
        "以下のquestionにcontextで与えられた情報のみを用いて3行程度で答えてください。"\
        "わからない場合は「わかりません」と出力してください。ですます調を用いて答えてください。"\
        "context:{context}\n"\
        "質問:{question}"
    )

    vectorstore = initialize_vectorstore()
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    chain1 = (
        {
            "history":RunnableLambda(st.session_state.chain_memory.load_memory_variables) | itemgetter("history"), 
            "question":RunnablePassthrough(),
        }
        | system_prompt1
        | model
        | StrOutputParser()
    )
    chain2 = (
        {
            "context":vectorstore.as_retriever(search_kwargs={"k": 3}),
            "question":chain1,
        }
        | system_prompt2
        | model
        | StrOutputParser()
    )
    return chain2

def create_agent_chain(): #エージェントを作る関数
    callback = [StreamlitCallbackHandler(st.container())]
    chat = ChatOpenAI(
        model = os.environ["OPENAI_API_MODEL"],
        temperature = float(os.environ["OPENAI_API_TEMPERATURE"]),
        streaming = True,
        callbacks = callback
    )
    tools = load_tools(tool_names=["ddg-search", "wikipedia"], callbacks=callback) #ツールを作成

    system_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "あなたは聞かれた質問に答える優秀なAIアシスタントです。"
            ),
            MessagesPlaceholder(variable_name="history"), 
            HumanMessagePromptTemplate.from_template("{question}"), #invoke時にquestionに入力を入れるようにする
            MessagesPlaceholder(variable_name = "agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm=chat, tools=tools, prompt=system_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, callbacks=callback, memory=st.session_state.chain_memory)


st.title("RAG or Agent streamlit-app")
USE_RAG = st.toggle('USE RAG (if it\'s off, then use Agent)')

if "conversation_memory" not in st.session_state: #会話履歴表示用memoryを初期化
    st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)

if "chain_memory" not in st.session_state: #chain用memoryを初期化(上限2個まで)
    st.session_state.chain_memory = ConversationBufferWindowMemory(k=2 ,memory_key="history", return_messages=True)

if "qa_chain" not in st.session_state: #RAGの初期化
    st.session_state.qa_chain = create_QAchain()

if "agent" not in st.session_state: #agentの初期化
    st.session_state.agent = create_agent_chain()


for message in st.session_state.conversation_memory.load_memory_variables({})["history"]: #メッセージ履歴の表示
    if(message.type=="ai"):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    else:
        with st.chat_message("user"):
            st.markdown(message.content)

prompt = st.chat_input("質問はありますか?") #chat入力欄の表示

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"): #応答の生成と表示
        if USE_RAG:
            agent_result = st.session_state.qa_chain.invoke(prompt)
            responce = agent_result
            st.markdown(responce)
            st.session_state.chain_memory.save_context({"input":prompt}, {"output":responce}) #RAGはagentと違ってメモリを結びつけていないため手動でメモリ追加
        else:
            agent_result = st.session_state.agent.invoke({"question":prompt})
            responce = agent_result["output"]
            st.markdown(responce)
    st.session_state.conversation_memory.save_context({"input":prompt}, {"output":responce}) #会話履歴表示用memoryへの追加
    
