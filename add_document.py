#docフォルダ内にある文書をベクトルデータベース(pinecone)に保存する関数

import logging
import os
import sys

import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import re

load_dotenv(override=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

#Pineconeをlangchain内のインスタンスとして作成する関数
def initialize_vectorstore():
    #initは内包されていらなくなったらしい
    index_name = os.environ["PINECONE_INDEX_NAME"]
    embeddings = OpenAIEmbeddings()
    return PineconeVectorStore.from_existing_index(index_name, embeddings)

#指定したテキストファイルの余計な部分を正規表現で除去する関数
def remove_ruby():
    with open("/workspace/doc/kagakuron_utf8.txt", 'r', encoding='utf_8') as input_file:
        content = input_file.read()

    pattern = r'《[^》]*》'
    tmp = re.sub(pattern, '', content)
    pattern = r'-{2,}.*?-{2,}'
    tmp = re.sub(pattern, '', tmp, flags=re.DOTALL)
    pattern = r'［＃.*\n'
    modified_content = re.sub(pattern, '', tmp)

    with open("/workspace/doc/kagakuron.txt", 'w', encoding='utf-8') as output_file:
        output_file.write(modified_content)

#指定したファイル内の文章を読み取りDBに保存する
if __name__ == "__main__":
    remove_ruby()
    loader = TextLoader("/workspace/doc/kagakuron.txt")
    documents = loader.load()
    logger.info("Loaded %d documents", len(documents))

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    logger.info("Split %d documents", len(docs))

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(docs)

