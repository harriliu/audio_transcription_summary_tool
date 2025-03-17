# Audio Transcription & Summarization Tool

A powerful application that automatically transcribes speech from audio files and generates concise summaries of the content, deployed on Hugging Face Spaces.

![Hugging Face](https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg)

## Overview

This app allows you to:
- Upload or record audio files in various formats
- Automatically transcribe speech to text using Whisper AI
- Generate concise summaries of the transcribed content
- View processing time and results in a user-friendly interface

## How to Use

1. **Record or Upload Audio**:
   - Click the microphone icon to record audio directly
   - Or click "Upload" to select an audio file from your device
   - Supported formats: MP3, WAV, M4A, and more

2. **Process the Audio**:
   - Click the "Transcribe & Summarize" button
   - Wait for processing to complete (time varies based on audio length)

3. **View Results**:
   - The complete transcript appears in the "Transcript" box
   - A concise summary appears in the "Summary" box
   - Processing time is displayed for reference

## Technical Details

The application uses state-of-the-art models for speech recognition and text summarization:

- **Transcription**: [OpenAI's Whisper](https://github.com/openai/whisper) - a robust speech recognition model that works across multiple languages and audio conditions
- **Optimization**: [faster-whisper](https://github.com/guillaumekln/faster-whisper) - an optimized implementation that improves performance
- **Summarization**: [BART-CNN](https://huggingface.co/facebook/bart-large-cnn) - a model fine-tuned for text summarization

The app includes several fallback mechanisms to ensure reliability:
- Multiple audio processing tool support
- Model size adjustments based on resource availability
- Alternative summarization models if primary models fail

## Limitations

- **Audio Quality**: For best results, use clear audio with minimal background noise
- **Processing Time**: Longer audio files will take more time to process
- **File Size**: Very large audio files may exceed the space's resource limits
- **Multiple Speakers**: Accuracy may be reduced for audio with multiple speakers or overlapping speech

## Local Installation

If you want to run this application locally:

1. Clone the repository:
```bash
git clone https://huggingface.co/spaces/harriliu1129/meeting_minutes_summarizer
cd meeting_minutes_summarizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open the URL displayed in your terminal (typically http://127.0.0.1:7860)

## Apple Silicon (M1/M2) Users

When running locally on Apple Silicon Macs, the application includes optimizations for compatibility with these processors. All processing defaults to CPU mode to avoid issues with Metal Performance Shaders (MPS) and sparse tensor operations.

## License

[MIT License](LICENSE)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for optimized implementation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the summarization models
- [Gradio](https://github.com/gradio-app/gradio) for the web interface framework

---

Created with ❤️ by [Your Name]

[Fork this Space](https://huggingface.co/spaces/YOUR_USERNAME/audio-transcription-app/fork) | [Report an Issue](https://huggingface.co/spaces/YOUR_USERNAME/audio-transcription-app/discussions)
