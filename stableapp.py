from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
#from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import io
import PyPDF2
from io import BytesIO
from huggingface_hub import hf_hub_download



system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.

Begin!
----------------
"""
#messages = [
#    SystemMessagePromptTemplate.from_template(system_template),
#    HumanMessagePromptTemplate.from_template("{question}"),
#]
#prompt = set_custom_prompt()#ChatPromptTemplate.from_messages(messages)
#chain_type_kwargs = {"prompt": prompt}


custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""



def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#prompt = set_custom_prompt()
#chain_type_kwargs = {"prompt" : prompt}




messages = [     
            SystemMessagePromptTemplate.from_template(system_template),     
            HumanMessagePromptTemplate.from_template(custom_prompt_template), 
] 
prompt = ChatPromptTemplate.from_messages(messages) 

chain_type_kwargs = {"prompt": prompt}





def downloadmodel(model_id,model_filename,model_path_cache):
    model_path = hf_hub_download(             
                                 repo_id=model_id,             
                                 filename=model_filename,             
                                 resume_download=True,             
                                 cache_dir=model_path_cache,         
                                 )
    return model_path

#Loading the model
def load_llm():

    #uncomment to allow the UI to download the model during first execution
    #mdpath = downloadmodel("TheBloke/Llama-2-7b-Chat-GGUF","llama-2-7b-chat.Q8_0.gguf","./models")

    # Load the locally downloaded model here
    llm = CTransformers(
        model = "models/llama-2-7b-chat.Q8_0.gguf"
        #model = mdpath,
        #model_file = "llama-2-7b-chat.Q8_0.gguf",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.2
    )
    return llm

def hugging_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    return embeddings


def retrieval_qa_chain(llm, prompt, db):     
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}                                        
                                           )     
    return qa_chain


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=280,
        ).send()

    file = files[0]
    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # Read the PDF file
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                   chunk_overlap=40)
    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = hugging_embedding()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    #chain = ConversationalRetrievalChain.from_llm(
    #    load_llm(),
    #    chain_type="stuff",
    #    retriever=docsearch.as_retriever(),
    #    memory=memory,
    #    return_source_documents=True,
    #)

    chain = retrieval_qa_chain(load_llm(), set_custom_prompt(), docsearch)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    #cb = cl.AsyncLangchainCallbackHandler()
    cb = cl.AsyncLangchainCallbackHandler(         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]     )     
    cb.answer_reached = True

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
