import os
import streamlit as st
from dotenv import load_dotenv
import re
import pandas as pd
import zipfile

import langchain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import load_tools, AgentExecutor, create_openai_tools_agent, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv(override=True) #.envファイルの中身を環境変数に設定
langchain.debug = True #コマンドライン上でLLMとのやり取りを可視化できる。必要なければFalseに

def remove_ruby(filepath): #指定したテキストファイルの余計な部分を正規表現で除去しエンコードをshiftjis->utf8へ変更する関数(青空文庫ファイル専用)
    with open(filepath, 'r', encoding='shift_jis') as input_file:
        content = input_file.read()
    pattern = r'《[^》]*》'
    tmp = re.sub(pattern, '', content)
    pattern = r'-{2,}.*?-{2,}'
    tmp = re.sub(pattern, '', tmp, flags=re.DOTALL)
    pattern = r'［＃.*\n'
    modified_content = re.sub(pattern, '', tmp)
    with open(filepath, 'w', encoding='utf-8') as output_file:
        output_file.write(modified_content)

def format_file(zip_filename): #zip形式のファイルを解凍してremove_rubyに入れる関数(zipの中身のファイル名を返す)
    try:
        with zipfile.ZipFile("/workspace/doc/"+zip_filename, 'r') as zip_ref:
            files = zip_ref.namelist()
            print(files)
            zip_ref.extractall("/workspace/doc")
        os.remove("/workspace/doc/"+zip_filename)
    except:
        st.error("ファイルの整形中にエラーが発生しました")
    if len(files) != 1:
        st.error("1ファイルのみを含む形式のzipファイルをアップロードしてください")
        return
    filename = files[0]
    filepath = "/workspace/doc/"+filename
    remove_ruby(filepath)
    return filename

def setuup_vectorDB(filename): #ファイル名を指定してその青空文庫ファイルをローカルのvectorDBに格納する関数(フォルダがない場合に一度だけ呼び出される)
    loader = TextLoader("/workspace/doc/"+filename)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    dummy_text, dummy_id = "1", 1
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts([dummy_text], embeddings, ids=[dummy_id])
    vectorstore.delete([dummy_id])
    vectorstore.merge_from(FAISS.from_documents(docs, embeddings))
    dir_name = "/workspace/DBs/vector_"+(filename.rsplit('.', 1)[0])
    vectorstore.save_local(dir_name)
    print("作成が完了しました")

def init_RAG_tool(filename): #与えられたファイルに関するRAGツールを作成する関数
    #ツール概要作成用のchain
    if st.session_state.RAG_sourcefiles[filename] == "": #概要が登録されていなければchainで作成し登録
        prompt = ChatPromptTemplate.from_template("以下の文章を読んでタイトルをつけてください。\n #文章:\n{sentence}")
        model = ChatOpenAI(
            model = os.environ["OPENAI_API_MODEL"],
            temperature = float(os.environ["OPENAI_API_TEMPERATURE"])
        )
        chain = prompt | model | StrOutputParser()
        with open("/workspace/doc/"+filename, 'r', encoding='utf-8') as file:
            description = chain.invoke({"sentence": file.read()})
        st.session_state.RAG_sourcefiles[filename] = description
        print(filename+" "+description)
    else: #登録されていたら登録済みの物を使う(毎回chain使うとapi使用料が無駄になってしまう)
        description = st.session_state.RAG_sourcefiles[filename]

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("/workspace/DBs/vector_"+(filename.rsplit('.', 1)[0]), embedding, allow_dangerous_deserialization=True)
    tool = create_retriever_tool(
        vectorstore.as_retriever(search_kwargs={"k": 3}),
        "search_about_"+(filename.rsplit('.', 1)[0]),
        f"{description}について検索して, 関連性が高い文書の一部を返します。",
    )
    return tool

def create_agent_chain(): #エージェントを作る関数
    chat = ChatOpenAI(
        model = os.environ["OPENAI_API_MODEL"],
        temperature = float(os.environ["OPENAI_API_TEMPERATURE"]),
        streaming = True
    )

    tools = load_tools(tool_names=["ddg-search"]) #ツールを初期化し作成
    tools = tools + [init_RAG_tool(x) for x in list(st.session_state.RAG_sourcefiles.keys())]

    system_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "あなたは聞かれた質問に答える優秀なAIアシスタントです。"
            ),
            MessagesPlaceholder(variable_name="history"), 
            HumanMessagePromptTemplate.from_template("{question}"), 
            MessagesPlaceholder(variable_name = "agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm=chat, tools=tools, prompt=system_prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)



#以下がstreamlit起動時に動作する部分

st.title("RAG and Agent streamlit-app")
return_source = st.toggle("Return Source Document") #参照元文書を返すかどうかを表す

if "RAG_sourcefiles" not in st.session_state: #RAGの参照ファイルとその概要(RAGtool作成に必要)の辞書型オブジェクト(docの中身で初期化)
    files = [f for f in os.listdir("/workspace/doc") if not f.startswith(".")] #隠しファイルを除くファイル一覧を取得
    st.session_state.RAG_sourcefiles = {file : "" for file in files}

uploaded_file = st.file_uploader("青空文庫のZIPファイルを選択してください", type=['zip'])
if uploaded_file is not None: #アップロードされたファイルがある場合は適宜処理してソースファイルとして追加する
    bytes_data = uploaded_file.getvalue()
    with open("/workspace/doc/"+uploaded_file.name, "wb") as output_file:
        output_file.write(bytes_data)
    filename = format_file(uploaded_file.name)
    st.session_state.RAG_sourcefiles[filename] = "" #概要を空にして新しく追加
    setuup_vectorDB(filename)
    st.session_state.agent = create_agent_chain() #ファイルが追加された場合agentも作成し直す

df = pd.DataFrame({ #使用可能ツールのテーブル表示
    '使用可能agentツール': ["duck duck go(web検索)"]+list(st.session_state.RAG_sourcefiles.keys()),
})
st.table(df)

with st.spinner("Vector DBの作成中..."):
    for filename in list(st.session_state.RAG_sourcefiles.keys()): #vectorDBのディレクトリがない場合に作成する(初回起動時のみ)
        print(filename+"の作成中")
        if not os.path.isdir("/workspace/DBs/vector_"+(filename.rsplit('.', 1)[0])): #存在しなかった場合作成
            setuup_vectorDB(filename)

if "conversation_memory" not in st.session_state: #会話履歴を初期化
    st.session_state.conversation_memory = []

if "chain_memory" not in st.session_state: #chain用memoryを初期化(上限2個まで)
    st.session_state.chain_memory = ConversationBufferWindowMemory(k=2 ,memory_key="history", return_messages=True)

if "agent" not in st.session_state: #agentの初期化
    st.session_state.agent = create_agent_chain()


for message in st.session_state.conversation_memory: #メッセージ履歴の表示
    if(message[0]=="ai"):
        with st.chat_message("assistant"):
            st.markdown(message[1])
    else:
        with st.chat_message("user"):
            st.markdown(message[1])

prompt = st.chat_input("質問はありますか?") #chat入力欄の表示

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"): #応答の生成と表示
        agent_result = st.session_state.agent.invoke({"question":prompt, "history":st.session_state.chain_memory.load_memory_variables({})["history"]})
        response = agent_result["output"]
        im_steps = agent_result["intermediate_steps"] #中間出力の取得
        st.markdown(response)
        if len(im_steps)!=0: #ツールを使用していた場合
            tool = "「"+im_steps[0][0].tool+"」"
            query =  "「"+im_steps[0][0].tool_input["query"]+"」"
            reff = "参照ツール:"+tool+"参照クエリ:"+query
            if return_source:
                document =  "「"+im_steps[0][1]+"」"
                reff = reff + "  \n参照文書:"+document
        else: #ツールを使用していなかった場合
            reff = "参照ツール:なし"
    with st.chat_message("assistant"):
        st.markdown(reff)
    st.session_state.conversation_memory.append(("user",prompt)) #会話履歴表示用memoryへの追加
    st.session_state.conversation_memory.append(("ai",response)) 
    st.session_state.conversation_memory.append(("ai",reff)) 
    st.session_state.chain_memory.save_context({"input":prompt}, {"output":response}) #agent用memoryへの追加
    
