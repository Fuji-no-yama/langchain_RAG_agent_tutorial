import logging
import os
import sys

from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv

load_dotenv(override=True)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = os.environ["PINECONE_INDEX_NAME"]

if index_name in pc.list_indexes():
    pc.delete_index(index_name)

pc.create_index(name=index_name, metric="cosine", dimension=1536, spec=PodSpec(
        environment='gcp-starter', 
        pod_type='starter'
    ))