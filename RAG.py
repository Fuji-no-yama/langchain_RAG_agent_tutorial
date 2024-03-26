import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter

import langchain
from langchain import hub
from langchain.chains import LLMChain, RetrievalQA
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import load_tools, AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import StreamlitCallbackHandler
from add_document import initialize_vectorstore

load_dotenv(override=True) #.envファイルの中身を環境変数に設定
langchain.debug = True

def create_QAchain():
    system_prompt = ChatPromptTemplate.from_template(
        """
        以下の質問にcontextで与えられた情報とhistoryのみを用いて3行程度で答えてください。
        わからない場合は「わかりません」と出力してください。
        context:{context}
        history:{history}
        質問:{question}
        """
    )
    vectorstore = initialize_vectorstore()
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = (
        {
            "context":vectorstore.as_retriever(),
            "history":RunnableLambda(st.session_state.memory.load_memory_variables) | itemgetter("history"), 
            "question":RunnablePassthrough(),
        }
        | system_prompt
        | model
        | StrOutputParser()
    )
    return chain


st.title("RAG streamlit-app")


if "messages" not in st.session_state:  #メッセージ履歴の初期化
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "qa_chain" not in st.session_state: #チェインの初期化
    st.session_state.qa_chain = create_QAchain()


for message in st.session_state.messages: #メッセージ履歴の表示
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("質問はありますか?") #chat入力欄の表示

if prompt:
    st.session_state.messages.append({"role":"user", "content":prompt}) #メッセージ履歴への追加(ユーザ)

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"): #応答の生成と表示
        agent_result = st.session_state.qa_chain.invoke(prompt)
        responce = agent_result
        st.markdown(responce)

    st.session_state.messages.append({"role":"assistant", "content":responce}) #メッセージ履歴への追加(AI)
    st.session_state.memory.save_context({"inputs":prompt}, {"output":responce})

