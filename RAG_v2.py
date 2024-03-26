import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter

import langchain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import StreamlitCallbackHandler
from add_document import initialize_vectorstore

load_dotenv(override=True) #.envファイルの中身を環境変数に設定
langchain.debug = True

def create_QAchain():
    system_prompt1 = ChatPromptTemplate.from_template(
        "以下のquestionを会話履歴であるhistoryを用いてより正確な質問にしてください。"\
        "わからない場合はquestionをそのまま出力してください。"\
        "######"\
        "question:{question}"\
        "history:{history}"
    )
    system_prompt2 = ChatPromptTemplate.from_template(
        "以下のquestionにcontextで与えられた情報のみを用いて3行程度で答えてください。"\
        "わからない場合は「わかりません」と出力してください。"\
        "context:{context}"\
        "質問:{question}"
    )

    vectorstore = initialize_vectorstore()
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain1 = (
        {
            "history":RunnableLambda(st.session_state.memory.load_memory_variables) | itemgetter("history"), 
            "question":RunnablePassthrough(),
        }
        | system_prompt1
        | model
        | StrOutputParser()
    )
    chain2 = (
        {
            "context":vectorstore.as_retriever(),
            "question":chain1,
        }
        | system_prompt2
        | model
        | StrOutputParser()
    )
    return chain2


st.title("RAG streamlit-app")

if "memory" not in st.session_state: #memoryを初期化
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "qa_chain" not in st.session_state: #チェインの初期化
    st.session_state.qa_chain = create_QAchain()


for message in st.session_state.memory.load_memory_variables({})["history"]: #メッセージ履歴の表示
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
        agent_result = st.session_state.qa_chain.invoke(prompt)
        responce = agent_result
        st.markdown(responce)

    st.session_state.memory.save_context({"inputs":prompt}, {"output":responce}) #memoryへの追加

