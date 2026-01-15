from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Вставьте любой файл в разрешении dox
file_path = "C:\\Users\\User\\Downloads\\Potamon ДЗ1_1.docx"

try:
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    print("File loaded successfully!")
except FileNotFoundError:
    print(f"Error: Такого файла не существует: {file_path}")
    documents = []
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    documents = []


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
splits = text_splitter.split_documents(documents)


# Create Ollama embeddings
embeddings = OllamaEmbeddings(model="llama3.2")

# Create a Chroma vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

# Convert vector store to a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print('Создано векторное хранилище')

# Initialize the chat model
llm = ChatOllama(model="llama3.1")

# Create a prompt template
prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say you don't know.

Question: {question} 

Context: {context}

Answer:
""")

# Create the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example query
question = "What is the main topic of the document?"
print('Запущена цепочка')
response = rag_chain.invoke(question)
print(response)

