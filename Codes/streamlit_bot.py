# streamlit_app.py

import streamlit as st
from main import load_and_chunk_pdfs,pdf_paths,create_nodes,get_embeddings,create_metadata_query_engine,query_index
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from llama_index.core import VectorStoreIndex



st.title("ChatBot")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Get Your Desired Answer')
st.markdown('<style>h3{color: pink; text-align: center;}</style>', unsafe_allow_html=True)
    

# Streamlit code to load PDFs
pdf_document = load_and_chunk_pdfs(pdf_paths)  

nodes1 = create_nodes(pdf_document)


# Streamlit code to generate embeddings
# sentence_index = VectorStoreIndex(nodes1)
sentence_index = get_embeddings(nodes1)

metadata_query_engine = create_metadata_query_engine(sentence_index)




# Streamlit code to load QA chain
# qa_chain = load_qa_chain(db)

# Streamlit code for UI
## question = st.text_input("Question:")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything about"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Display chat history
reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        question = st.text_input("Question:", placeholder="Ask about the pdf", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and question:
        output = query_index(metadata_query_engine,question)
        st.session_state['past'].append(question)
        st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
    
    

