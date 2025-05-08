#!/usr/bin/env python3
"""
Chinese Podcast Transcription and Translation Tool

This script converts audio files to text and translates Chinese speech to English.
Supports various audio formats and provides progress tracking.
"""
import wave
import json
import os
import subprocess
import argparse
import tempfile
import time
import logging
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("podcast_translation.log")
    ]
)
logger = logging.getLogger(__name__)

def setup_argparser() -> argparse.Namespace:
    """Set up command-line argument parser with helpful descriptions."""
    parser = argparse.ArgumentParser(
        description="Transcribe and translate podcast audio from Chinese to English.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_file", 
        help="Path to the input audio file (supports various formats: M4A, MP3, WAV, etc.)"
    )
    parser.add_argument(
        "--model_path", 
        default="model-cn", 
        help="Path to Vosk model directory for Chinese speech recognition"
    )
    parser.add_argument(
        "--transcript_file", 
        help="Output file for Chinese transcript (default: input_filename_transcript.txt)"
    )
    parser.add_argument(
        "--translated_file", 
        help="Output file for English translation (default: input_filename_translated.txt)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=10, 
        help="Number of paragraphs to translate in each batch"
    )
    parser.add_argument(
        "--skip_translation", 
        action="store_true", 
        help="Skip translation step (transcribe only)"
    )
    parser.add_argument(
        "--translation_model", 
        default="Helsinki-NLP/opus-mt-zh-en", 
        help="HuggingFace translation model to use"
    )
    parser.add_argument(
        "--keep_temp", 
        action="store_true", 
        help="Keep temporary WAV file after processing"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        # Check for FFmpeg
        subprocess.run(
            ["ffmpeg", "-version"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Try importing required Python packages
        import vosk
        from transformers import pipeline
        
        return True
    except (subprocess.SubprocessError, ImportError) as e:
        logger.error(f"Dependency check failed: {str(e)}")
        logger.error("Please ensure FFmpeg is installed and in your PATH")
        logger.error("Install required Python packages: pip install vosk transformers tqdm")
        return False

def convert_to_wav(input_file: str) -> str:
    """
    Convert input audio file to mono 16kHz WAV format using FFmpeg.
    
    Args:
        input_file: Path to the input audio file
    
    Returns:
        Path to the temporary WAV file
    
    Raises:
        subprocess.SubprocessError: If FFmpeg conversion fails
    """
    input_path = Path(input_file)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        output_file = tmpfile.name
    
    logger.info(f"Converting '{input_path.name}' to 16kHz mono WAV format...")
    
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-ar", "16000",  # Sample rate: 16kHz
        "-ac", "1",      # Channels: mono
        "-y",            # Overwrite output file if it exists
        output_file
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Conversion complete: {output_file}")
        return output_file
    except subprocess.SubprocessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)
        raise

def transcribe_audio(wav_file: str, model_path: str) -> str:
    """
    Transcribe the WAV audio file using Vosk.
    
    Args:
        wav_file: Path to the WAV audio file
        model_path: Path to the Vosk model directory
    
    Returns:
        Transcribed text with timestamps
    
    Raises:
        ValueError: If audio format is incorrect
        RuntimeError: If transcription fails
    """
    try:
        from vosk import Model, KaldiRecognizer
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        wf = wave.open(wav_file, "rb")
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            raise ValueError("Audio must be mono 16kHz")
        
        # Get total audio duration for progress calculation
        total_frames = wf.getnframes()
        total_duration = total_frames / wf.getframerate()
        
        logger.info(f"Loading speech recognition model from {model_path}...")
        model = Model(model_path)
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)  # Enable word timestamps
        
        # Results will contain text and timing information
        results = []
        processed_frames = 0
        chunk_size = 4000
        
        logger.info(f"Transcribing {total_duration:.2f} seconds of audio...")
        
        # Use tqdm for progress bar
        with tqdm(total=total_duration, unit="sec", desc="Transcribing") as pbar:
            last_update = 0
            
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                
                # Process audio chunk
                if rec.AcceptWaveform(data):
                    chunk_result = json.loads(rec.Result())
                    if chunk_result.get("text"):
                        results.append(chunk_result)
                
                # Update progress bar
                processed_frames += chunk_size
                current_time = processed_frames / wf.getframerate()
                pbar.update(current_time - last_update)
                last_update = current_time
            
            # Process final result
            final_result = json.loads(rec.FinalResult())
            if final_result.get("text"):
                results.append(final_result)
        
        wf.close()
        
        # Process results to include timestamps
        processed_text = []
        for r in results:
            if "result" in r and r["text"]:
                # Extract start and end times if available
                words = r.get("result", [])
                if words:
                    start_time = words[0].get("start", 0)
                    end_time = words[-1].get("end", 0)
                    timestamp = f"[{format_timestamp(start_time)} --> {format_timestamp(end_time)}]"
                    processed_text.append(f"{timestamp} {r['text']}")
                else:
                    processed_text.append(r["text"])
            elif r.get("text"):
                processed_text.append(r["text"])
        
        return "\n".join(processed_text)
    
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Failed to transcribe audio: {str(e)}")

def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.MS."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

def translate_text(chinese_text: str, model_name: str, batch_size: int = 10) -> str:
    """
    Translate Chinese text to English using a pre-trained model with batching.
    
    Args:
        chinese_text: Text in Chinese
        model_name: Name of the HuggingFace translation model
        batch_size: Number of paragraphs to translate in each batch
    
    Returns:
        Translated text in English
    """
    from transformers import pipeline
    
    logger.info(f"Loading translation model: {model_name}")
    translator = pipeline("translation", model=model_name)
    
    # Split text into paragraphs
    paragraphs = [p for p in chinese_text.split("\n") if p.strip()]
    
    if not paragraphs:
        logger.warning("No text to translate")
        return ""
    
    translated_paragraphs = []
    total_paragraphs = len(paragraphs)
    
    logger.info(f"Translating {total_paragraphs} paragraphs (batch size: {batch_size})...")
    
    # Process in batches to avoid memory issues and provide better progress tracking
    with tqdm(total=total_paragraphs, desc="Translating") as pbar:
        for i in range(0, total_paragraphs, batch_size):
            batch = paragraphs[i:i+batch_size]
            
            # Process the batch
            translated_batch = []
            for para in batch:
                # Extract timestamp if present
                timestamp_match = None
                content = para
                
                # Look for timestamp pattern [HH:MM:SS.MS --> HH:MM:SS.MS]
                if para.startswith("[") and "]" in para:
                    timestamp_end = para.find("]") + 1
                    timestamp = para[:timestamp_end]
                    content = para[timestamp_end:].strip()
                    
                # Only translate non-empty content
                if content.strip():
                    try:
                        translation = translator(content)
                        translated_text = translation[0]['translation_text']
                        
                        # Reattach timestamp if it was present
                        if timestamp_match:
                            translated_batch.append(f"{timestamp} {translated_text}")
                        else:
                            translated_batch.append(translated_text)
                    except Exception as e:
                        logger.error(f"Translation error: {str(e)}")
                        # Keep original text if translation fails
                        translated_batch.append(f"[TRANSLATION ERROR] {para}")
                else:
                    # Keep empty lines or just timestamps
                    translated_batch.append(para)
            
            # Add batch results
            translated_paragraphs.extend(translated_batch)
            pbar.update(len(batch))
    
    return "\n".join(translated_paragraphs)

def get_default_output_filenames(input_file: str) -> Tuple[str, str]:
    """Generate default output filenames based on input filename."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    transcript_file = f"{base_name}_transcript.txt"
    translated_file = f"{base_name}_translated.txt"
    return transcript_file, translated_file

def main():
    """Main function to run the transcription and translation pipeline."""
    start_time = time.time()
    
    # Parse command-line arguments
    args = setup_argparser()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Set default output filenames if not provided
    if not args.transcript_file or not args.translated_file:
        default_transcript, default_translated = get_default_output_filenames(args.input_file)
        args.transcript_file = args.transcript_file or default_transcript
        args.translated_file = args.translated_file or default_translated
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Error: Input file {args.input_file} does not exist.")
        return 1
    
    wav_file = None
    
    try:
        # Convert audio to WAV format
        wav_file = convert_to_wav(args.input_file)
        
        # Transcribe audio
        logger.info("Starting transcription...")
        chinese_text = transcribe_audio(wav_file, args.model_path)
        
        # Save transcript
        with open(args.transcript_file, "w", encoding="utf-8") as f:
            f.write(chinese_text)
        logger.info(f"Transcription saved to {args.transcript_file}")
        
        # Translate if not skipped
        if not args.skip_translation:
            logger.info("Starting translation...")
            english_text = translate_text(
                chinese_text, 
                args.translation_model, 
                args.batch_size
            )
            
            # Save translation
            with open(args.translated_file, "w", encoding="utf-8") as f:
                f.write(english_text)
            logger.info(f"Translation saved to {args.translated_file}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        return 0
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
        
    finally:
        # Clean up temporary file
        if wav_file and os.path.exists(wav_file) and not args.keep_temp:
            os.remove(wav_file)
            logger.debug(f"Removed temporary file: {wav_file}")

if __name__ == "__main__":
    sys.exit(main())