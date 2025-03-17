import gradio as gr
import time
import os
import tempfile
import subprocess
import sys
import platform

# Set environment variables for better compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback for operations that support it
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # Avoid warnings

# Import torch first to check availability
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("Installing PyTorch...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch'])
    import torch

# Now import other dependencies
import whisper
from transformers import pipeline

# Check for audio processing tools
def check_audio_tools():
    # Try to find ffmpeg first (should be pre-installed on Spaces)
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("ffmpeg is installed! Using ffmpeg for audio processing.")
        return "ffmpeg"
    except FileNotFoundError:
        # Try other alternatives
        try:
            subprocess.run(['avconv', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("avconv (libav) is installed! Using libav for audio processing.")
            return "libav"
        except FileNotFoundError:
            try:
                subprocess.run(['sox', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("SoX is installed! Using SoX for audio processing.")
                return "sox"
            except FileNotFoundError:
                print("No audio processing tools found. Some functionality may be limited.")
                return "none"

# Check for required packages and install if missing
def check_and_install_packages():
    required_packages = ['whisper', 'transformers', 'sentencepiece', 'accelerate']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"{package} not found. Installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])

# Force CPU mode for Spaces to ensure compatibility
device = "cpu"  
print("Using CPU mode for all models to ensure compatibility")

# Try to use faster-whisper if available
try:
    from faster_whisper import WhisperModel
    print("Using faster-whisper for improved performance!")
    
    # Configure for best stability on Spaces
    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
    use_faster_whisper = True
    print("Loaded faster-whisper model with optimized CPU settings")
        
except (ImportError, ValueError) as e:
    if isinstance(e, ValueError):
        try:
            # If int8 failed, try with standard float32
            print(f"Could not use int8 precision: {str(e)}")
            print("Trying with standard float32...")
            model = WhisperModel("small", device="cpu", compute_type="float32")
            use_faster_whisper = True
            print("Loaded faster-whisper model with float32 precision")
        except Exception as e2:
            print(f"Failed to initialize faster-whisper: {str(e2)}")
            use_faster_whisper = False
    else:
        print("faster-whisper not found. Using standard whisper.")
        use_faster_whisper = False

# Load standard whisper as fallback
if not use_faster_whisper:
    try:
        # Use small model for better results but with reasonable resource usage
        print("Loading Whisper small model on CPU...")
        model = whisper.load_model("small", device="cpu")
        print("Whisper small model loaded successfully!")
    except Exception as e:
        print(f"Error loading standard whisper model: {str(e)}")
        print("Trying with tiny model as fallback...")
        # Last resort - use tiny model which has fewer parameters
        model = whisper.load_model("tiny", device="cpu")
        print("Whisper tiny model loaded with fallback configuration!")

# Load summarization model
print("Loading summarization model...")
try:
    # Use a relatively small model for Spaces
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    print("Summarization model loaded!")
except Exception as e:
    print(f"Error loading standard summarization model: {str(e)}")
    print("Trying with a smaller model...")
    try:
        # If the large model fails, try with a smaller one
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)
        print("Smaller summarization model loaded as fallback!")
    except Exception as e2:
        print(f"Error loading fallback summarization model: {str(e2)}")
        # Define a simple function that just returns the input as a last resort
        def dummy_summarize(text, **kwargs):
            words = text.split()
            if len(words) > 50:
                return [{'summary_text': ' '.join(words[:50]) + '...'}]
            return [{'summary_text': text}]
        
        summarizer = dummy_summarize
        print("Using basic text truncation as summarization fallback!")

# Detect the audio tool once at startup
audio_tool = check_audio_tools()

def transcribe_and_summarize(audio_file):
    start_time = time.time()
    
    # Step 1: Check if audio file was uploaded
    if audio_file is None:
        return "Please upload an audio file.", "", 0
    
    # Step 2: Convert audio if necessary based on the available tool
    processed_audio = audio_file
    if audio_tool != "none" and audio_tool != "ffmpeg":
        try:
            # Create temp file for processed audio
            temp_dir = tempfile.gettempdir()
            processed_audio = os.path.join(temp_dir, "processed_audio.wav")
            
            if audio_tool == "libav":
                # Use avconv (libav) to convert
                subprocess.run(['avconv', '-i', audio_file, '-ar', '16000', '-ac', '1', processed_audio], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elif audio_tool == "sox":
                # Use SoX to convert
                subprocess.run(['sox', audio_file, '-r', '16000', '-c', '1', processed_audio], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Audio processed with {audio_tool}")
        except Exception as e:
            print(f"Warning: Could not process audio with {audio_tool}: {e}")
            # Fall back to original file
            processed_audio = audio_file
    
    # Step 3: Transcribe the audio using Whisper with error handling
    transcribe_start = time.time()
    try:
        if use_faster_whisper:
            # Using faster-whisper
            segments, info = model.transcribe(processed_audio, beam_size=5)
            transcript = " ".join([segment.text for segment in segments])
        else:
            # Using standard whisper with additional error handling
            try:
                result = model.transcribe(processed_audio)
                transcript = result["text"]
            except RuntimeError as e:
                if "cuda" in str(e).lower() or "mps" in str(e).lower():
                    print("GPU error detected. Forcing CPU mode and retrying...")
                    # Force CPU mode if GPU fails - use a different variable name to avoid confusion
                    fallback_model = whisper.load_model("tiny", device="cpu")
                    result = fallback_model.transcribe(processed_audio)
                    transcript = result["text"]
                else:
                    raise
        
        if not transcript.strip():
            return "No speech detected in the audio file.", "", 0
            
        transcribe_time = time.time() - transcribe_start
        print(f"Transcription completed in {transcribe_time:.2f} seconds")
    except Exception as e:
        return f"Error transcribing audio: {str(e)}", "", 0
        
    # Clean up temp file if created
    if processed_audio != audio_file and os.path.exists(processed_audio):
        try:
            os.remove(processed_audio)
        except:
            pass
    
    # Step 4: Summarize the transcript
    summarize_start = time.time()
    
    # Function to chunk text for summarization
    def chunk_text(text, max_length=1000):
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    # Process chunks and summarize
    chunks = chunk_text(transcript)
    chunk_summaries = []
    
    for chunk in chunks:
        # Skip empty chunks
        if not chunk.strip():
            continue
            
        try:
            # Summarize the chunk
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Error summarizing chunk: {str(e)}")
            # If summarization fails, include original chunk
            chunk_summaries.append(chunk[:100] + "...")
    
    # Combine chunk summaries
    summary = " ".join(chunk_summaries)
    
    # If we have multiple chunks, summarize again for coherence
    if len(chunk_summaries) > 1:
        try:
            final_summary = summarizer(" ".join(chunk_summaries), max_length=100, min_length=30, do_sample=False)
            summary = final_summary[0]['summary_text']
        except Exception as e:
            print(f"Error in final summarization: {str(e)}")
            # Keep existing summary if final summarization fails
        
    summarize_time = time.time() - summarize_start
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    return transcript, summary, round(total_time, 2)

# Create Gradio interface
with gr.Blocks(title="Audio Transcription & Summarization") as demo:
    gr.Markdown("# 🎙️ Audio Transcription & Summarization Tool")
    gr.Markdown("Upload an audio file or record directly to transcribe and summarize its content.")
    
    # Add environment info
    with gr.Accordion("Environment Info", open=False):
        if use_faster_whisper:
            gr.Markdown(f"**Model**: faster-whisper (small)")
        else:
            gr.Markdown(f"**Model**: standard whisper ({model.model.dims.n_text_ctx}-ctx)")
        gr.Markdown(f"**Audio processing**: Using {audio_tool}")
    
    # Add error message display
    error_output = gr.Textbox(label="Status", visible=True)
    
    with gr.Row():
        with gr.Column():
            # Changed to allow both microphone recording and file upload
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Audio")
            submit_btn = gr.Button("Transcribe & Summarize", variant="primary")
            processing_time = gr.Number(label="Processing Time (seconds)", precision=2)
        
        with gr.Column():
            transcript_output = gr.Textbox(label="Transcript", lines=10)
            summary_output = gr.Textbox(label="Summary", lines=5)
    
    # Function to handle errors
    def process_with_error_handling(audio_file):
        try:
            transcript, summary, time_taken = transcribe_and_summarize(audio_file)
            return "", transcript, summary, time_taken
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if "ffmpeg" in str(e).lower() or "audio" in str(e).lower():
                error_msg += "\n\nAudio processing error. Please try a different audio format."
            return error_msg, "", "", 0
    
    submit_btn.click(
        fn=process_with_error_handling,
        inputs=audio_input,
        outputs=[error_output, transcript_output, summary_output, processing_time]
    )
    
    gr.Markdown("### How to Use")
    gr.Markdown("""
    1. Upload an audio file (mp3, wav, m4a, etc.) or record directly
    2. Click the 'Transcribe & Summarize' button
    3. Wait for processing (time depends on audio length)
    4. View the transcript and summary results
    
    This app uses Whisper for transcription and BART-CNN for summarization.
    """)

    gr.Markdown("### Limitations")
    gr.Markdown("""
    - For best results, use clear audio with minimal background noise
    - Processing long audio files may take more time
    - Maximum audio length is limited by system resources
    - Audio files with multiple speakers may have reduced accuracy
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
