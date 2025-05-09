"""
Video Translator
---------------
A tool to automatically translate English videos into Arabic.

This script performs the following:
1. Extracts audio from a video file
2. Transcribes the audio to English text using OpenAI's Whisper
3. Translates the text to Arabic using Googletrans
4. Generates an SRT subtitle file
5. Optionally guides on how to embed subtitles into the video
"""

import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Set FFmpeg path - IMPORTANT: Change this to your FFmpeg installation path
# For example: r"C:\FFmpeg\bin\ffmpeg.exe" 
FFMPEG_PATH = r"C:\FFmpeg\bin\ffmpeg.exe"  # Change this to your actual FFmpeg path

def ensure_ffmpeg():
    """
    Helper function to ensure FFmpeg is available.
    Tries to download it if not found.
    """
    # Use a nonlocal reference to the global variable
    global FFMPEG_PATH
    
    if os.path.exists(FFMPEG_PATH):
        print(f"FFmpeg found at: {FFMPEG_PATH}")
        return True
        
    # Try to find ffmpeg in PATH
    import shutil
    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        print(f"FFmpeg found in PATH: {ffmpeg_in_path}")
        FFMPEG_PATH = ffmpeg_in_path
        return True
    
    # Ask user if they want to attempt automatic download
    print("\nFFmpeg not found. It's required for audio processing.")
    user_input = input("Do you want to attempt to automatically download FFmpeg? (y/n): ")
    
    if user_input.lower() != 'y':
        print("Please install FFmpeg manually:")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Extract to a folder (e.g., C:\\FFmpeg)")
        print("3. Update this script with the path to ffmpeg.exe")
        return False
    
    try:
        # Try to automatically download FFmpeg for Windows
        import urllib.request
        import zipfile
        import tempfile
        
        print("Downloading FFmpeg... (this may take a while)")
        ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "ffmpeg.zip")
        
        # Download
        urllib.request.urlretrieve(ffmpeg_url, zip_path)
        
        # Extract
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find ffmpeg.exe
        for root, dirs, files in os.walk(temp_dir):
            if "ffmpeg.exe" in files:
                ffmpeg_path = os.path.join(root, "ffmpeg.exe")
                print(f"FFmpeg downloaded to: {ffmpeg_path}")
                
                # Update path
                FFMPEG_PATH = ffmpeg_path
                
                # Add to PATH
                os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
                return True
        
        print("Downloaded FFmpeg, but couldn't locate ffmpeg.exe in the package.")
        return False
        
    except Exception as e:
        print(f"Error downloading FFmpeg: {e}")
        print("Please install FFmpeg manually.")
        return False

# Add FFmpeg to environment variable PATH if path is provided
if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

# For audio extraction
from moviepy.editor import VideoFileClip

# For transcription
import whisper

# For translation
from googletrans import Translator

# For subtitle generation
import pysrt
from pysrt import SubRipFile, SubRipItem

# For progress display
from tqdm import tqdm


class VideoTranslator:
    def __init__(self, input_path: str, output_dir: str, subtitle_dir: str):
        """
        Initialize the VideoTranslator with paths.
        
        Args:
            input_path: Path to the input video file
            output_dir: Directory to save output videos
            subtitle_dir: Directory to save subtitle files
            
        Note:
            The subtitle file will have the same name as the original video file but with .srt extension
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.subtitle_dir = subtitle_dir
        
        # Get original video filename without extension
        self.video_filename = os.path.basename(input_path)
        self.video_name = os.path.splitext(self.video_filename)[0]
        
        self.audio_path = os.path.join(output_dir, f"{self.video_name}.mp3")
        self.subtitle_path = os.path.join(subtitle_dir, f"{self.video_name}.srt")
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(subtitle_dir, exist_ok=True)
        
        # Initialize translation module
        self.translator = Translator()
        
        # Initialize whisper model (can be replaced with different size models)
        print("Loading Whisper model (this may take a moment)...")
        self.whisper_model = whisper.load_model("base")
        
    def extract_audio(self) -> str:
        """
        Extract audio from the input video file using FFmpeg directly.
        
        Returns:
            Path to the extracted audio file
        """
        print("Extracting audio from video...")
        try:
            print("Using FFmpeg for direct audio extraction...")
            import subprocess
            
            # Extract audio using ffmpeg directly
            temp_wav = self.audio_path.replace('.mp3', '.wav')
            ffmpeg_cmd = [
                FFMPEG_PATH,
                '-i', self.input_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # WAV format
                '-ar', '44100',  # Sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite output file
                temp_wav
            ]
            
            print("Running FFmpeg command:", ' '.join(ffmpeg_cmd))
            subprocess.run(ffmpeg_cmd, check=True)
            
            # Convert to MP3
            print("Converting to MP3...")
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(self.audio_path, format="mp3", bitrate="192k")
            
            # Clean up
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            
            print(f"Audio extracted successfully to {self.audio_path}")
            return self.audio_path
            
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Check if the video file is corrupted")
            print("2. Try converting the video to a different format (e.g., MP4 with H.264 codec)")
            print("3. Ensure FFmpeg is properly installed and accessible")
            print("4. Check if the video has a valid audio track")
            return None
        
    def transcribe_audio(self) -> dict:
        """
        Transcribe the audio file using Whisper.
        
        Returns:
            Dictionary containing the transcription results with timestamps
        """
        print("Transcribing audio with Whisper...")
        try:
            # Try direct transcription with FFmpeg path
            if os.path.exists(FFMPEG_PATH):
                print(f"Using FFmpeg from: {FFMPEG_PATH}")
                result = self.whisper_model.transcribe(self.audio_path, verbose=False)
            else:
                # Fallback to direct file transcription without specifying FFmpeg
                print("Attempting direct transcription...")
                result = self.whisper_model.transcribe(self.audio_path, verbose=False)
            
            print("Transcription complete!")
            return result
        except Exception as e:
            print(f"\nERROR during transcription: {str(e)}")
            print("Please make sure FFmpeg is installed and accessible.")
            print("You can download FFmpeg from: https://ffmpeg.org/download.html")
            print("After installation, set the FFMPEG_PATH variable in this script.\n")
            
            # Try with a simpler approach as last resort
            try:
                print("Attempting alternative transcription method...")
                # Use a different approach that might work without FFmpeg
                import torch
                import numpy as np
                from pathlib import Path
                
                # First try loading with absolute path
                file_path = Path(self.audio_path).absolute()
                print(f"Trying with absolute path: {file_path}")
                
                try:
                    # Load audio file with scipy if available
                    from scipy.io import wavfile
                    print("Loading audio with scipy...")
                    if self.audio_path.endswith('.mp3'):
                        print("Converting mp3 to wav first...")
                        temp_wav = self.audio_path.replace('.mp3', '.wav')
                        from moviepy.editor import AudioFileClip
                        AudioFileClip(self.audio_path).write_audiofile(temp_wav)
                        audio_path = temp_wav
                    else:
                        audio_path = self.audio_path
                        
                    sample_rate, audio_data = wavfile.read(audio_path)
                    
                    # Convert to mono if stereo
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Convert to float32 and normalize
                    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
                    
                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        print(f"Resampling from {sample_rate}Hz to 16000Hz...")
                        import librosa
                        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    
                    # Convert to tensor
                    audio_tensor = torch.from_numpy(audio_data)
                    
                    # Transcribe
                    print("Transcribing audio tensor...")
                    result = self.whisper_model.transcribe(audio_tensor, verbose=False)
                    print("Alternative transcription complete!")
                    return result
                    
                except Exception as e3:
                    print(f"Audio loading with scipy failed: {str(e3)}")
                    # Try one more approach
                    print("Trying with librosa as last resort...")
                    import librosa
                    
                    audio_data, sample_rate = librosa.load(self.audio_path, sr=16000, mono=True)
                    result = self.whisper_model.transcribe(audio_data, verbose=False)
                    print("Librosa transcription complete!")
                    return result
                
            except Exception as e2:
                print(f"All transcription methods failed: {str(e2)}")
                print("Please make sure FFmpeg is properly installed and try again.")
                sys.exit(1)
        
    def translate_text(self, text: str, src_lang: str = 'en', dest_lang: str = 'ar') -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            dest_lang: Destination language code
            
        Returns:
            Translated text
        """
        # This can be easily replaced with other translation APIs
        translated = self.translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
        
    def create_subtitles(self, transcription: dict) -> None:
        """
        Create subtitle file from transcription.
        
        Args:
            transcription: Whisper transcription result
        """
        print("Creating subtitles...")
        subs = SubRipFile()
        
        for i, segment in enumerate(tqdm(transcription['segments'])):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            
            # Translate each segment
            translated_text = self.translate_text(text)
            
            # Convert time to SubRip format (hours:minutes:seconds,milliseconds)
            start = self._seconds_to_srt_time(start_time)
            end = self._seconds_to_srt_time(end_time)
            
            # Create subtitle item
            item = SubRipItem(index=i+1, 
                             start=start, 
                             end=end, 
                             text=translated_text)
            subs.append(item)
        
        # Save subtitle file
        subs.save(self.subtitle_path, encoding='utf-8')
        print(f"Subtitle file created at {self.subtitle_path}")
        
    def _seconds_to_srt_time(self, seconds: float) -> pysrt.SubRipTime:
        """
        Convert seconds to SubRip time format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Time in SubRip format
        """
        hours = int(seconds / 3600)
        seconds %= 3600
        minutes = int(seconds / 60)
        seconds %= 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        
        return pysrt.SubRipTime(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        
    def process_video(self) -> None:
        """
        Process the video: extract audio, transcribe, translate, and create subtitles.
        """
        self.extract_audio()
        transcription = self.transcribe_audio()
        self.create_subtitles(transcription)
        
        # Display instructions for embedding subtitles
        print("\nTo embed subtitles into the video, run:")
        output_video = os.path.join(self.output_dir, f"{self.video_name}_translated.mp4")
        print(f"ffmpeg -i {self.input_path} -vf subtitles={self.subtitle_path} {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Translate English videos to Arabic")
    parser.add_argument("--video", type=str, help="Path to the input video file")
    args = parser.parse_args()
    
    # Ensure FFmpeg is available
    if not ensure_ffmpeg():
        print("FFmpeg is required but not available. Please install FFmpeg and try again.")
        return
    
    # Set default directories
    input_dir = "input_videos"
    output_dir = "output_videos"
    subtitle_dir = "subtitles"
    
    # If no video specified, look for videos in the input directory
    if args.video is None:
        videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not videos:
            print(f"No videos found in {input_dir}. Please add videos or specify a path with --video.")
            return
        
        video_path = os.path.join(input_dir, videos[0])
        print(f"Processing first video found: {video_path}")
    else:
        video_path = args.video
    
    # Process the video
    translator = VideoTranslator(
        input_path=video_path,
        output_dir=output_dir,
        subtitle_dir=subtitle_dir
    )
    
    translator.process_video()


if __name__ == "__main__":
    main() 
    