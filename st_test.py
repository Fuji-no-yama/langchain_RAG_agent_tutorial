import os
import streamlit as st
import zipfile
import re

st.title("file test")

uploaded_file = st.file_uploader("ZIPファイルを選択してください", type=['zip'])

def remove_ruby(filepath):
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

def format_file(zip_filename):
    try:
        with zipfile.ZipFile("/workspace/test_doc/"+zip_filename, 'r') as zip_ref:
            files = zip_ref.namelist()
            print(files)
            zip_ref.extractall("/workspace/test_doc")
        os.remove("/workspace/test_doc/"+zip_filename)
    except:
        st.error("ファイルの整形中にエラーが発生しました")
    if len(files) != 1:
        st.error("1ファイルのみを含む形式のzipファイルをアップロードしてください")
        return
    filename = files[0]
    filepath = "/workspace/test_doc/"+filename
    remove_ruby(filepath)


if uploaded_file is not None:
    # バイトとしてファイルを読み取る場合：
    bytes_data = uploaded_file.getvalue()
    with open("/workspace/test_doc/"+uploaded_file.name, "wb") as output_file:
        output_file.write(bytes_data)
    format_file(uploaded_file.name)