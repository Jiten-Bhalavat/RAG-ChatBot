import openai
import os
from dotenv import load_dotenv
from llama_index.core.text_splitter import SentenceSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from llama_index.core import SimpleDirectoryReader

from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import MetadataReplacementPostProcessor


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

pdf_paths=[r"C:\Users\ASUS\Desktop\Jiten\RIL-70_compressed-1-50-1-20.pdf"]

def load_and_chunk_pdfs(pdf_paths):
  documents = {}
  
  for pdf_path in pdf_paths:
    documents = SimpleDirectoryReader(input_files=[pdf_path])
    docs = documents.load_data()  

  return docs

def create_nodes(documents, chunk_size=1000, chunk_overlap=200):
  node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  nodes = node_parser.get_nodes_from_documents(documents)

  return nodes

  
# Create FAISS index
def get_embeddings(nodes1):
  llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
  ctx = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-base-en-v1.5"
  )
  sentence_index = VectorStoreIndex(nodes1, service_context=ctx)
  return sentence_index


def create_metadata_query_engine(sentence_index):
  metadata_query_engine = sentence_index.as_query_engine(
      similarity_top_k=5,
      # the target key defaults to `window` to match the node_parser's default
      node_postprocessors=[
          MetadataReplacementPostProcessor(target_metadata_key="window") 
      ]
  )
  
  return metadata_query_engine





# # Load QA model
# def load_qa_chain(faiss_index):
#   # llm_name="gpt-3.5-turbo"
#   # llm = ChatOpenAI(model_name=llm_name, temperature=0.3)

#   template = """Use the context to answer the question. If you don't know the answer, say "I don't know". Keep the answer concise and say "Thanks for asking" at the end.
#   {context}  
#   Question: {question}
#   Answer:"""
  
#   qa_prompt = PromptTemplate.from_template(template)

#   qa_chain = RetrievalQA.from_chain_type(llm,  
#                                          chain_type="stuff",
#                                          retriever=faiss_index.as_retriever(),
#                                          return_source_documents=True,
#                                          chain_type_kwargs={"prompt": qa_prompt})
  
#   return qa_chain

def query_index(metadata_query_engine, query_text):
  response = metadata_query_engine.query(query_text)
  return str(response)






# def query(question, qa_chain):
#   result = qa_chain({"query": question})
#   answer = result["result"]
  
#   sources = ""
#   for doc in result["source_documents"]:
#     sources += f"{doc.metadata['source']}\n"
  
#   full_response = f"{answer}\n\nSources:\n{sources}"
  
#   return full_response