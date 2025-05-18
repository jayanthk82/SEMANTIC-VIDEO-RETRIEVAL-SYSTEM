
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForQuestionAnswering
from multiprocessing import Pool
import math
import cv2
from PIL import Image
from database import chromadb_setup

def init_worker_VQA():
    global video_captioning_processor
    global video_captioning_model
    video_captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)
    video_captioning_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def init_worker_text_vectorization():
    global text_vectorization_model
    text_vectorization_model = SentenceTransformer("all-MiniLM-L6-v2")
def text_vectorization_model_encode(text):
    global text_vectorization_model
    return text_vectorization_model.encode(text, convert_to_numpy=True)

def upload_your_dataset(video_address):
    chromadb_setup_path = '/home/jayanth/Documents/SMART-RETRIVEVAL /CHROMADB'
    chroma_collection_name = 'VIDEOs'
    samples_count = len(video_address)

    with Pool(initializer=init_worker_VQA) as pool:
        summary_list = pool.map(summaries, video_address)
    print('Summaries:', len(summary_list))

    with Pool(initializer=init_worker_text_vectorization) as pool:
        vector = pool.map(text_vectorization_model_encode, summary_list)
    print('Vectorization:', len(vector))
    #EDIT
    return chromadb_setup(chromadb_setup_path, chroma_collection_name, vector, samples_count, summary_list)

def summaries(video_path):
    global video_captioning_processor
    global video_captioning_model

    vidcap = cv2.VideoCapture(video_path)
    story = ''
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    second = 0

    def caption(raw_image):
        question = "what's happening in the image"
        inputs = video_captioning_processor(raw_image, question, return_tensors="pt")
        out = video_captioning_model.generate(**inputs)
        return video_captioning_processor.decode(out[0], skip_special_tokens=True)

    while True:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, math.floor(second * fps))
        ret, frame = vidcap.read()
        if not ret:
            break
        story += caption(Image.fromarray(frame)) + ' '
        second += 1

    vidcap.release()
    return story
