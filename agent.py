import os
import streamlit as st
from dotenv import load_dotenv

from langchain import hub
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import load_tools, AgentExecutor, create_openai_tools_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(override=True) #.envファイルの中身を環境変数に設定
#USE_OPENAI = True
USE_AGENT = True

def create_agent_chain(): #エージェントを作る関数
    callback = [StreamlitCallbackHandler(st.container())]
    chat = ChatOpenAI(
        model = os.environ["OPENAI_API_MODEL"], #modelかmodel_nameかは変化していそう
        temperature = float(os.environ["OPENAI_API_TEMPERATURE"]),
        streaming = True,
        callbacks = callback
    )
    tools = load_tools(tool_names=["ddg-search", "wikipedia"], callbacks=callback) #ツールを作成

    system_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human."
            ),
            MessagesPlaceholder(variable_name="chat_history"), #memory定義時に"chat_history"という名前は一致している必要がある
            HumanMessagePromptTemplate.from_template("{question}"), #invoke時にquestionに入力を入れるようにする
            MessagesPlaceholder(variable_name = "agent_scratchpad"),
        ]
    )
    #llm_chain = LLMChain(llm=chat, prompt=system_prompt)
    agent = create_openai_tools_agent(llm=chat, tools=tools, prompt=system_prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, callbacks=callback, memory=memory)

st.title("streamlit-app")
# switch = st.toggle("Activate Chat GPT")
# if(switch):
#     USE_OPENAI = True

if "messages" not in st.session_state:  #メッセージ履歴の初期化
    st.session_state.messages = []

if "agent_chain" not in st.session_state: #エージェントの初期化
    st.session_state.agent_chain = create_agent_chain()
    print("エージェントの初期化を行いました")

for message in st.session_state.messages: #メッセージ履歴の表示
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("what is up?") #chat入力欄の表示


if prompt:
    st.session_state.messages.append({"role":"user", "content":prompt}) #メッセージ履歴への追加(ユーザ)

    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if(USE_AGENT): #さらにagentを使う場合
            agent_result = st.session_state.agent_chain.invoke({"question":prompt})
            #agent_result = st.session_state.agent_chain.run(question=prompt)
            responce = agent_result["output"]
                
        else: #通常の記憶なしチャットを使う場合
            chat = ChatOpenAI(
                model = os.environ["OPENAI_API_MODEL"],
                temperature = os.environ["OPENAI_API_TEMPERATURE"],
            )
            messages = [HumanMessage(content=prompt)]
            responce = chat(messages).content

        st.markdown(responce)

    st.session_state.messages.append({"role":"assistant", "content":responce}) #メッセージ履歴への追加(AI)

