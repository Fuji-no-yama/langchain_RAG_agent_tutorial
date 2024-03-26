FROM python:3.10
USER root

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip && \
    apt-get install -y sudo wget vim curl gawk make gcc

# Install Basic Packages
RUN pip install ipykernel jupyter && \
    pip install scikit-learn && \
    pip install matplotlib && \
    pip install langchain openai && \
    pip install langchain_openai langchainhub langchain_pinecone && \
    pip install pydantic && \
    pip install GitPython && \
    pip install streamlit && \
    pip install python-dotenv && \
    pip install duckduckgo-search wikipedia && \
    pip install pinecone-client tiktoken && \
    pip install faiss-cpu

EXPOSE 8501