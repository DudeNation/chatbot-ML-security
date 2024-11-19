import os
import chainlit as cl
from moviepy.editor import VideoFileClip
import librosa
import logging

logger = logging.getLogger(__name__)

async def handle_media_upload(file: cl.File) -> str:
    if not file or not file.path:
        logger.error(f"Invalid file or empty path: {file}")
        return "Error: Invalid file or empty path"

    try:
        file_extension = os.path.splitext(file.name)[1].lower()

        if file_extension in ['.mp4', '.mov', '.avi']:
            return extract_video_content(file.path)
        elif file_extension in ['.mp3', '.wav', '.ogg']:
            return extract_audio_content(file.path)
        else:
            return f"Unsupported media file type: {file_extension}"

    except Exception as e:
        logger.error(f"Error processing media file {file.name}: {str(e)}")
        return f"Error processing media file: {str(e)}"

def extract_video_content(file_path: str) -> str:
    with VideoFileClip(file_path) as video:
        duration = video.duration
        fps = video.fps
        size = video.size
        audio_present = video.audio is not None

    return f"Video analysis: Duration: {duration:.2f} seconds, FPS: {fps}, Resolution: {size[0]}x{size[1]}, Audio: {'Present' if audio_present else 'Absent'}"

def extract_audio_content(file_path: str) -> str:
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return f"Audio analysis: Duration: {duration:.2f} seconds, Sample rate: {sr} Hz, Estimated tempo: {tempo:.2f} BPM"
