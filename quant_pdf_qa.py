# Usage: python quant_pdf_qa.py [path_to_pdf_folder]
# Built on top of the tutorial shown in https://github.com/samwit/langchain-tutorials

import sys
import torch
import transformers
import textwrap

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter



def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    assistant_prefix = "[assistant]:"
    print(assistant_prefix, wrap_text_preserve_newlines(llm_response['result']))
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


def main():
    if len(sys.argv) != 2:
        print("Invalid command. Usage: python quant_pdf_qa.py [path_to_pdf_folder]")
        sys.exit()

    path_pdf_folder = sys.argv[1]

    model_id = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    # Load and process the text files
    loader = DirectoryLoader(path_pdf_folder + '/', glob="./*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()

    #Split the documents into paragraph text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

    vectordb = Chroma.from_documents(documents=texts, embedding=instructor_embeddings)

    top_k_document_sources = 3
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k_document_sources})

    # Create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)

    # Main controller loop for the chat bot
    user_prefix = "\n[user]: "
    assistant_prefix = f"[assistant]:"

    print("\nEnter 'exit' or 'quit' to terminate this session.\n")

    while (True):
        query = input(user_prefix)

        if query.lower() == "exit" or query.lower() == "quit":
            break

        print("Question: ", query)
        llm_response = qa_chain.invoke(query)
        process_llm_response(llm_response)


if __name__ == "__main__":
    main()

