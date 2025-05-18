
import time 
import os 
from preprocessing import upload_your_dataset
from transformers import BlipProcessor, BlipForQuestionAnswering
from sentence_transformers import SentenceTransformer
def init_worker_VQA():
    global video_captioning_processor
    global video_captioning_model
    video_captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)
    video_captioning_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def init_worker_text_vectorization():
    global text_vectorization_model
    text_vectorization_model = SentenceTransformer("all-MiniLM-L6-v2")

def folder_walkthrough(root_folder):
    data_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            data_paths.append(os.path.join(dirpath, fname))
    return data_paths


# ------------------- QUERY FUNCTION -------------------
def QUERY(Query,chromadb_collection,video_address):
  global text_vectorization_model
  ChromaDB_Query_Embeddings = text_vectorization_model.encode(Query, convert_to_numpy=True)
  ChromaDB_Query_result = chromadb_collection.query(query_embeddings = ChromaDB_Query_Embeddings,
                 n_results=1)
  print('using chromaDB: Your query is related to the document is at ',video_address[int(ChromaDB_Query_result['ids'][0][0])])

# ------------------- Safe Main Runner -------------------

if __name__ == '__main__':
    tic = time.time()
    init_worker_VQA()
    init_worker_text_vectorization()
    video_address = folder_walkthrough('/home/jayanth/Documents/SMART-RETRIVEVAL /VIDEO_DATASET')
    chromadb_collection = upload_your_dataset(video_address)
    toc = time.time()
    print('Time taken to upload the dataset:', toc - tic)
    print('Dataset uploaded successfully!')
    while True:
        Query =  input('Enter your query: ')
        if Query == 'exit':
            break
        QUERY(Query,chromadb_collection,video_address)