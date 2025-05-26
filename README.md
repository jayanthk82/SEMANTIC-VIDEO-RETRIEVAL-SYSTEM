# 🎥 Smart Video Retrieval System

A **production-ready, multimodal video retrieval system** that matches videos to **natural language queries** using powerful vision-language models and LLMs. This system enables **semantic video search** by leveraging frame-level visual summaries and enriched audio transcripts, all indexed in a vector database for blazing-fast similarity search.

---

## ✨ Features

- 🔍 **Natural Language Video Search**: Find relevant videos using everyday language.
- 🧠 **Multimodal Understanding**: Combines visual and audio information for richer context.
- 🖼️ **Visual Summarization**: Extracts key frames (1 FPS) and generates meaningful summaries using a vision-language model.
- 🎙️ **Audio Transcript Enrichment**: Converts speech to text and integrates it with visual data.
- ⚡ **Fast Retrieval**: Stores embeddings and metadata in a vector database for efficient semantic search.
- 🧪 **Scalable & Production-Ready**: Clean architecture ready for deployment and scaling.

---

## 🧠 Tech Stack

| Component | Technology |
|----------|-------------|
| Language | Python |
| Visualsummary Model | BLIP |
| Vector DB |Chroma |
| Audio Processing | ffmpeg, pydub |
| Frame Extraction | OpenCV |
| Deployment | Streamlit |

---

## 🚀 Getting Started

RUN run.py using |streamlit run| command in CLI using seperate environmet to get streamlit UI as local webhost. 
