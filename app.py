import os
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs

# Global variables to store state
vector_store = None
transcript_text = None
video_id_global = None


def extract_video_id(url_or_id: str) -> str:
    """Extract the YouTube video ID from a URL or return the ID if it's already in the correct format."""
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        parsed_url = urlparse(url_or_id)
        # Handle standard watch URLs
        if parsed_url.hostname and "youtube" in parsed_url.hostname:
            query = parse_qs(parsed_url.query)
            if "v" in query:
                return query["v"][0]
        # Handle youtu.be short links
        if parsed_url.hostname and "youtu.be" in parsed_url.hostname:
            return parsed_url.path.lstrip("/")
    return url_or_id


def get_transcript(video_id):
    """Fetch transcript from YouTube
       Proxies: Unfortunately, YouTube has started blocking most IPs that are known
       belong to cloud providers, which means you will most likely run into
       RequestBlocked or IpBlocked exceptions when deploying your code to any cloud platforms.
       To prevent this proxy has been set up.
       Be aware that using a proxy doesn't guarantee that you won't be blocked,
       as YouTube can always block the IP of your proxy!
       Therefore, you should always choose a solution that rotates through a pool of proxy addresses,
       if you want to maximize reliability. Although it's not being used here, you have to purchase Residential Proxy package.
    """
    try:
        transcript_snippets = YouTubeTranscriptApi(
            proxy_config = WebshareProxyConfig(
                proxy_username = "zvvcdjca",
                proxy_password = "rtgzzncfso2s",
            )
        ).fetch(video_id, languages=['en'])
        text = " ".join(snippet.text for snippet in transcript_snippets)
        return text, None
    except TranscriptsDisabled:
        return None, "No captions available for this video"
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_video(video_input, api_key, chunk_size, chunk_overlap):
    """Process video and create vector store"""
    global vector_store, transcript_text, video_id_global
    
    if not api_key:
        return "Please enter your Google API key", "", ""
    
    if not video_input:
        return "Please enter a YouTube video URL or ID", "", ""
    
    try:
        # Extract video ID
        video_id = extract_video_id(video_input)
        video_id_global = video_id
        
        # Get transcript
        transcript, error = get_transcript(video_id)
        if error:
            return f"Error: {error}", "", ""
        
        transcript_text = transcript
        
        # Create vector store
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size), 
            chunk_overlap=int(chunk_overlap)
        )
        chunks = splitter.create_documents([transcript])
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model='models/text-embedding-004',
            google_api_key=api_key
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create stats
        stats = f"""Video processed successfully!
        
Stats:
- Characters: {len(transcript)}
- Words: {len(transcript.split())}
- Chunks: {len(chunks)}
"""
        
        video_embed = f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
        
        return stats, video_embed, transcript[:500] + "..." if len(transcript) > 500 else transcript
        
    except Exception as e:
        return f"Error: {str(e)}", "", ""

def format_docs(retrieved_docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def answer_question(question, api_key, temperature, top_k):
    """Answer question using RAG"""
    global vector_store
    
    if vector_store is None:
        return "Please process a video first"
    
    if not question:
        return "Please enter a question"
    
    try:
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type='similarity', 
            search_kwargs={'k': int(top_k)}
        )
        
        # Create prompt
        prompt = PromptTemplate(
            template="""You are a helpful assistant.
                        Answer ONLY from the provided transcript context.
                        If the context is insufficient, just say you don't know.
                        
                        Context: {context}
                        
                        Question: {question}
                        
                        Answer:""",
            input_variables=['context', 'question']
        )
        
        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',
            temperature=float(temperature),
            google_api_key=api_key
        )
        
        # Create chain
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        
        rag_chain = parallel_chain | prompt | llm | StrOutputParser()
        
        # Get answer
        answer = rag_chain.invoke(question)
        return answer
        
    except Exception as e:
        return f"Error: {str(e)}"


def quick_key_points(api_key, temperature, top_k):
    """Quick key points button"""
    return answer_question(
        "What are the main key points discussed in this video?",
        api_key,
        temperature,
        top_k
    )


# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.main-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    color: white;
}
.instructions-box {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
"""
# Create Gradio Interface

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
)

with gr.Blocks(title="YouTube Video Q&A with RAG", theme=theme, css=custom_css) as demo:
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">YouTube Video Q&A</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
            AI-powered video analysis using Retrieval-Augmented Generation
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            with gr.Group():
                gr.Markdown("### Input")
                api_key_input = gr.Textbox(
                    label="Google API Key",
                    type="password",
                    placeholder="Enter your Google API key (get one at https://makersuite.google.com/app/apikey)"
                )
                video_input = gr.Textbox(
                    label="YouTube Video URL or ID",
                    placeholder="e.g., Gfr50f6ZBvo or https://www.youtube.com/watch?v=Gfr50f6ZBvo"
                )
                
                with gr.Row():
                    chunk_size = gr.Slider(500, 2000, 1000, step=100, label="Chunk Size")
                    chunk_overlap = gr.Slider(0, 500, 200, step=50, label="Chunk Overlap")
                
                process_btn = gr.Button("Process Video", variant="primary", size="lg")
            
            # Results section
            with gr.Group():
                gr.Markdown("### Results")
                status_output = gr.Textbox(label="Status", lines=5)
                video_output = gr.HTML(label="Video")
                transcript_preview = gr.Textbox(label="Transcript Preview", lines=3)
        
        with gr.Column(scale=1):
            # Settings
            with gr.Group():
                gr.Markdown("### Settings")
                temperature = gr.Slider(0.0, 1.0, 0.2, step=0.1, label="Temperature")
                top_k = gr.Slider(1, 10, 4, step=1, label="Top K Documents")
            
            # Instructions
            with gr.Group():
                gr.Markdown("### How to use")
                gr.Markdown("""
                        1. Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
                        2. Enter your API key above
                        3. Paste a YouTube video URL or ID
                        4. Click 'Process Video'
                        5. Ask questions or use quick actions
                """)

            with gr.Group():
                gr.Markdown("### ℹ️ About")
                gr.Markdown("""
                        This tool uses:
                        - **LangChain** for RAG pipeline
                        - **FAISS** for vector search
                        - **Google Gemini** for AI responses
                        - **YouTube Transcript API** for captions
                        
                        All processing happens in real-time.
                """)
    
    gr.Markdown("---")
    
    # Q&A section
    with gr.Group():
        gr.Markdown("### Ask Questions")
        
        with gr.Row():
            key_points_btn = gr.Button("Key Points")
        
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What is discussed about nuclear fusion?",
            lines=2
        )
        ask_btn = gr.Button("Get Answer", variant="primary", size="lg")
        
        answer_output = gr.Textbox(label="Answer", lines=10)
    
    gr.Markdown("---")
    
    # Full transcript
    with gr.Accordion("View Full Transcript", open=False):
        full_transcript = gr.Textbox(label="Full Transcript", lines=20)
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("Built with Gradio, LangChain, and Google Gemini AI")
    
    # Event handlers
    process_btn.click(
        fn=process_video,
        inputs=[video_input, api_key_input, chunk_size, chunk_overlap],
        outputs=[status_output, video_output, transcript_preview]
    ).then(
        fn=lambda: transcript_text if transcript_text else "",
        outputs=[full_transcript]
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, api_key_input, temperature, top_k],
        outputs=[answer_output]
    )
    
    key_points_btn.click(
        fn=quick_key_points,
        inputs=[api_key_input, temperature, top_k],
        outputs=[answer_output]
    )
    

# Launch the interface
if __name__ == "__main__":
    demo.launch()
