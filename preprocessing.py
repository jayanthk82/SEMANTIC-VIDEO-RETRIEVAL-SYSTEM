from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForQuestionAnswering
from multiprocessing import Pool
import math
import cv2
from PIL import Image
from database import chromadb_setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
pipeline = logging.getLogger('pipeline')

    
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
init_worker_VQA()
init_worker_text_vectorization()
def upload_your_dataset(video_address):

    pipeline.info('Starting video summary...')
    summary_list = []
    vector = []
    for i in video_address:
        summary_list.append(summaries(i))
    #with Pool(initializer=init_worker_VQA) as pool:
     #   summary_list = pool.map(summaries, video_address)
    pipeline.info('Video summary complete.Total summaries: %s', len(summary_list))
    pipeline.info('Starting text vectorization...')
    for i in summary_list:
        vector.append(text_vectorization_model.encode(i, convert_to_numpy=True))
    #with Pool(initializer=init_worker_text_vectorization) as pool:
     #   vector = pool.map(text_vectorization_model_encode, summary_list)
    pipeline.info('Text vectorization complete.Total vectors: %s', len(vector))
    pipeline.info('Inserting data into ChromaDB...')

    return chromadb_setup(vector,summary_list,video_address)

def summaries(video_address):
    global video_captioning_processor
    global video_captioning_model
    vidcap = cv2.VideoCapture(video_address)
    story = ''
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    second = 0
    def caption(raw_image):
        question = "what's happening in the image?"
        inputs = video_captioning_processor(raw_image, question, return_tensors="pt")
        out = video_captioning_model.generate(**inputs)
        return video_captioning_processor.decode(out[0], skip_special_tokens=True)
    while True:
        if second >= duration:
            break
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, math.floor(second * fps))
        ret, frame = vidcap.read()
        if not ret:
            break
        story += caption(Image.fromarray(frame)) + ' '
        second += 1
    vidcap.release()
    return story
