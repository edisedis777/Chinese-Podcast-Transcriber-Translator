# Chinese Podcast Transcriber & Translator

A command-line tool that automatically transcribes Chinese podcasts and translates them to English.

## Features

- **Audio Processing**: Converts various audio formats to optimized WAV
- **Speech Recognition**: Uses Vosk for accurate Chinese speech recognition
- **Translation**: Automatically translates Chinese text to English
- **Progress Tracking**: Shows real-time progress during long operations
- **Timestamps**: Preserves timing information in transcripts

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chinese-podcast-transcriber.git
cd chinese-podcast-transcriber

# Install dependencies
pip install vosk transformers tqdm

# Download Vosk Chinese model
wget https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip
unzip vosk-model-cn-0.22.zip
mv vosk-model-cn-0.22 model-cn
```

Make sure FFmpeg is installed on your system.

## Usage

Basic usage:

```bash
python translate-podcasts.py your-podcast.mp3
```

Advanced options:

```bash
python translate-podcasts.py your-podcast.mp3 \
    --model_path path/to/model \
    --transcript_file chinese_output.txt \
    --translated_file english_output.txt \
    --batch_size 15 \
    --verbose
```

## Requirements

- Python 3.7+
- FFmpeg
- Vosk
- Transformers
- tqdm

## License
- GNU Affero General Public License v3.0
