# YouTube Video Q&A with RAG

An AI-powered application that extracts transcripts from YouTube videos and enables intelligent question-answering using Retrieval-Augmented Generation (RAG). Built with LangChain, FAISS, and Google Gemini AI.


## Features

- **Transcript Extraction**: Automatically fetches English transcripts from YouTube videos
- **Intelligent Q&A**: Ask questions about video content and get accurate, context-aware answers
- **RAG Pipeline**: Uses Retrieval-Augmented Generation for accurate information retrieval
- **Vector Search**: FAISS-based similarity search for relevant context retrieval
- **Modern UI**: Clean, professional Gradio interface with real-time progress indicators
- **Customizable Settings**: Adjust chunk sizes, temperature, and retrieval parameters
- **Quick Actions**: One-click summarization, key points extraction, and topic identification

## Demo

Try the live demo: [https://huggingface.co/spaces/Ekanshu/youtube-rag]

## Architecture
```
- User Input (YouTube URL)
        ↓
- Transcript Extraction 
        ↓
- Text Chunking 
        ↓
- Embedding Generation 
        ↓
- Vector Store (FAISS)
        ↓
- Question → Retrieval → Context + Question → Prompt → LLM → Answer
```

## Technology Stack

- **LangChain**: RAG pipeline orchestration
- **FAISS**: Vector similarity search and storage
- **Google Gemini AI**: Language model for embeddings and generation
- **Gradio**: Web interface
- **YouTube Transcript API**: Video transcript extraction

### Prerequisites

- Python 3.8 or higher
- Google API key ([Get one here](https://makersuite.google.com/app/apikey))
