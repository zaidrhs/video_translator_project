# Video Translator: English to Arabic

A Python tool that automatically translates English videos into Arabic by generating subtitle files and optionally embedding them directly into the video.

## Features

- Extracts audio from video files
- Transcribes English audio to text using OpenAI's Whisper
- Translates English text to Arabic using Googletrans
- Generates SRT subtitle files with proper timestamps
- Provides instructions for embedding subtitles into videos

## Project Structure

```
video_translator_project/
├── main.py              # Main script
├── requirements.txt     # Dependencies
├── input_videos/        # Place original videos here
├── output_videos/       # Stores extracted audio and final videos
└── subtitles/           # Stores generated .srt files
```

## Requirements

- Python 3.8+
- FFmpeg (for audio extraction and subtitle embedding)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/video_translator_project.git
   cd video_translator_project
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure FFmpeg is installed on your system:
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `apt-get install ffmpeg`

## Usage

### Basic Usage

1. Place your English video file in the `input_videos` folder
2. Run the script:
   ```
   python main.py
   ```
3. The script will:
   - Extract audio from the video
   - Transcribe the audio to English text
   - Translate the text to Arabic
   - Generate an SRT subtitle file in the `subtitles` folder

### Advanced Usage

You can specify a specific video file:
```
python main.py --video path/to/your/video.mp4
```

### Embedding Subtitles

After generating the subtitle file, you can embed it into the video using FFmpeg:
```
ffmpeg -i input_videos/your_video.mp4 -vf subtitles=subtitles/your_video.srt output_videos/your_video_translated.mp4
```

## Used Libraries

- [MoviePy](https://zulko.github.io/moviepy/): Video editing with Python
- [OpenAI Whisper](https://github.com/openai/whisper): Robust speech recognition
- [Googletrans](https://py-googletrans.readthedocs.io/): Google Translate API (unofficial)
- [PySRT](https://github.com/byroot/pysrt): SubRip subtitle manipulation
- [FFmpeg-Python](https://github.com/kkroening/ffmpeg-python): Python bindings for FFmpeg
- [tqdm](https://github.com/tqdm/tqdm): Progress bar display

## Extension Points

The project is designed to be modular and extensible:

### Translation Module

The `translate_text()` method in `VideoTranslator` class can be easily replaced with other translation APIs:

```python
# Current implementation (Googletrans)
def translate_text(self, text: str, src_lang: str = 'en', dest_lang: str = 'ar') -> str:
    translated = self.translator.translate(text, src=src_lang, dest=dest_lang)
    return translated.text

# Example alternative (DeepL API)
def translate_text(self, text: str, src_lang: str = 'EN', dest_lang: str = 'AR') -> str:
    import deepl
    translator = deepl.Translator(your_api_key)
    result = translator.translate_text(text, source_lang=src_lang, target_lang=dest_lang)
    return result.text
```

### Transcription Module

The Whisper model size can be adjusted for speed/accuracy tradeoffs:

```python
# Current implementation (base model)
self.whisper_model = whisper.load_model("base")

# For higher accuracy
self.whisper_model = whisper.load_model("large")

# For faster processing
self.whisper_model = whisper.load_model("tiny")
```

## Future Improvements

- [ ] Add a graphical user interface (Tkinter, PyQt)
- [ ] Deploy as a web application (Flask, Django)
- [ ] Integrate professional translation APIs (DeepL, Azure Translator)
- [ ] Support additional languages beyond English and Arabic
- [ ] Implement faster-whisper for improved performance
- [ ] Add direct video upload via web interface
- [ ] Enable automatic YouTube uploads via API
- [ ] Implement batch processing for multiple videos
- [ ] Add voice-over capability in target language
- [ ] Improve subtitle formatting and styling options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License with the following condition: The code cannot be used for business purposes without explicit permission from the author. For further details, see the LICENSE file for details.