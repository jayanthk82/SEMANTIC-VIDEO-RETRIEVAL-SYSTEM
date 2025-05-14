# 📦 Install required packages
!pip install PyMuPDF spacy sentence-transformers chromadb

# 🧠 Import libraries
import os
import numpy as np
import tensorflow as tf
import chromadb
import fitz  # PyMuPDF
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


class QUERY_DOCUMENT_RETRIEVAL:

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.data_paths = self.get_data_paths(root_folder)
        self.text = [self.extract_text_from_pdf(path) for path in self.data_paths]
        self.vector = self.vectorize(N=True)
        self.vector2 = self.vectorize(T=True)
        self.collection = self.chromaDB(
            location="/content/drive/MyDrive/MOVIE_SCRIPTS/chromadb",
            collection_name="movie_scripts"
        )

    def vectorize(self, N=False, T=False):
        if N:
            return [self.model.encode(text, convert_to_numpy=True) for text in self.text]
        elif T:
            return [self.model.encode(text, convert_to_tensor=True) for text in self.text]
        else:
            raise ValueError("Please specify either N or T for vectorization.")

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def get_data_paths(self, root_folder):
        data_paths = []
        for dirpath, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.lower().endswith(".pdf"):
                    data_paths.append(os.path.join(dirpath, filename))
        return data_paths

    def chromaDB(self, location: str, collection_name: str):
        client = chromadb.PersistentClient(path=location)
        collection = client.get_or_create_collection(name=collection_name)
        for i, embedding in enumerate(self.vector):
            collection.add(
                ids=[str(i)],
                embeddings=[embedding]
            )
        return collection

    def QUERY(self, QUERY_TEXT):
        VD_EMBEDDINGS = self.model.encode(QUERY_TEXT, convert_to_numpy=True)
        ST_EMBEDDINGS = self.model.encode(QUERY_TEXT, convert_to_tensor=True)

        ST = util.semantic_search(ST_EMBEDDINGS, self.vector2, top_k=1)

        VD = self.collection.query(query_embeddings=VD_EMBEDDINGS, n_results=1)

        print('🔍 Using chromaDB: Related document is at →', self.data_paths[int(VD['ids'][0][0])])
        print('🤖 Using Sentence Transformers: Related document is at →', self.data_paths[ST[0][0]['corpus_id']])


# 🗂️ Define your folder
root_folder = "/content/drive/MyDrive/MOVIE_SCRIPTS/"
OBJ = QUERY_DOCUMENT_RETRIEVAL(root_folder)
OBJ.QUERY("I am a hero")
