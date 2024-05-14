import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
    
def process_pdf_batch(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
        )
    pages = character_splitter.split_documents(pages)
    return pages

def generate_queries():
        
    # Multi Query: Different Perspectives
    template = """You are an AI language model assistant. Your task is to generate Four 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)


    generate_querie = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    return generate_querie 


def _get_docs(m):
    docs=[]
    docs.extend(m['original'])
    for i in m['new']:
        docs.extend(i)
    return docs



def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)

   
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings

def keyword_extractor():
    prompt="""
    You are an AI language model assistant. Your task is to help the user retrieve keywords from their query. 

    Please provide me with the keywords you would like to extract from your query. 

    Keywords: {keywords}
    """
    prompt_perspectives=ChatPromptTemplate.from_template(prompt)
    generate_querie = (
        prompt_perspectives 
        | ChatOpenAI(temperature=0) 
        | StrOutputParser() )
    return generate_querie

def main(Query):
    d="resume"
    chunks=process_pdf_batch(f"data\{d}.pdf")
    document_in_faiss=FAISS.load_local("faiss_index", OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    faiss_retriever=document_in_faiss.as_retriever(search_kwargs={'k': 10})

    original_question= faiss_retriever
    retrieval_chain =  generate_queries() | faiss_retriever.map()
    map_chain = RunnableParallel(original=original_question,new=retrieval_chain) | _get_docs

    Bm25_retriever = BM25Retriever.from_documents(chunks)
    Bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
    retrievers=[Bm25_retriever, map_chain], weights=[0.5, 0.5]
    )

    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=4)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    final_prompt="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:"""

    final_prompt_perspectives=ChatPromptTemplate.from_template(final_prompt)

    llm_chain2=({"context": itemgetter("query") | compression_retriever,
              "question":itemgetter("query")}
             | final_prompt_perspectives
             | ChatOpenAI(temperature=0) | StrOutputParser() )
    
    return llm_chain2.invoke({"query":Query})

if __name__=="__main__":
    d=main("What are all the education qualification of shyam?")
    print(d)