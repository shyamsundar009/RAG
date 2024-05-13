from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
import os


def _read_pdf(filename):
    reader = PdfReader(filename)
    
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts


def _chunk_texts(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(texts))

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts

def process_pdf_batch(pdf_files):
    batch_docs = []
    for pdf_file_path in pdf_files:
        temp_file = pdf_file_path.name
        with open(temp_file, "wb") as file:
            file.write(pdf_file_path.getvalue())
        loader = PyPDFLoader(temp_file)
        pages = loader.load_and_split()
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
             chunk_size=1000,
            chunk_overlap=0
            )
        pages = character_splitter.split_documents(pages)
        os.remove(temp_file)
        batch_docs.extend(pages)
    return batch_docs

def load_faiss(filename):
    texts = _read_pdf(filename)
    chunks = _chunk_texts(texts)

    if os.path.exists(f"my_index"):
        document_in_faiss.load_local(f"my_index")
    else:
        document_in_faiss = FAISS.from_documents(chunks, OpenAIEmbeddings())
        document_in_faiss.save_local(f"my_index")

    return document_in_faiss


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