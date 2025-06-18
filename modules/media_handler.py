import os
import chainlit as cl
import logging
import tempfile
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import json
import re
import magic
import subprocess
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

# Enhanced media format support - comprehensive list
SUPPORTED_VIDEO_FORMATS = {
    # Common video formats
    '.mp4': 'MP4 Video', '.avi': 'AVI Video', '.mov': 'QuickTime Video',
    '.mkv': 'Matroska Video', '.wmv': 'Windows Media Video', '.flv': 'Flash Video',
    '.webm': 'WebM Video', '.m4v': 'iTunes Video', '.3gp': '3GPP Video',
    '.3g2': '3GPP2 Video', '.mpg': 'MPEG Video', '.mpeg': 'MPEG Video',
    '.m2v': 'MPEG-2 Video', '.ts': 'Transport Stream', '.mts': 'AVCHD Video',
    '.m2ts': 'Blu-ray Video', '.vob': 'DVD Video', '.f4v': 'Flash Video F4V',
    '.swf': 'Shockwave Flash', '.rm': 'RealMedia', '.rmvb': 'RealMedia VBR',
    '.asf': 'Advanced Systems Format', '.ogv': 'Ogg Video', '.dv': 'Digital Video',
    '.y4m': 'YUV4MPEG2', '.mxf': 'Material eXchange Format'
}

SUPPORTED_AUDIO_FORMATS = {
    # Common audio formats
    '.mp3': 'MP3 Audio', '.wav': 'WAV Audio', '.flac': 'FLAC Audio',
    '.aac': 'AAC Audio', '.ogg': 'Ogg Vorbis', '.m4a': 'MPEG-4 Audio',
    '.wma': 'Windows Media Audio', '.aiff': 'AIFF Audio', '.au': 'AU Audio',
    '.ra': 'RealAudio', '.ac3': 'Dolby Digital', '.dts': 'DTS Audio',
    '.ape': "Monkey's Audio", '.tak': 'TAK Audio', '.opus': 'Opus Audio',
    '.3ga': '3GPP Audio', '.amr': 'AMR Audio', '.awb': 'AMR-WB Audio',
    '.gsm': 'GSM Audio', '.voc': 'Creative Voice', '.vox': 'Dialogic VOX',
    '.snd': 'SND Audio', '.8svx': '8-Bit Sampled Voice'
}

MEDIA_ANALYSIS_CACHE = {}

class EnhancedMediaAnalyzer:
    """Comprehensive media analyzer with extensive format support."""
    
    def __init__(self):
        self.ffprobe_timeout = 30
        
    def detect_media_format(self, file_path: str) -> Tuple[str, str, Dict[str, Any]]:
        """Enhanced format detection."""
        try:
            file_ext = Path(file_path).suffix.lower()
            mime_type = magic.from_file(file_path, mime=True)
            
            if file_ext in SUPPORTED_VIDEO_FORMATS:
                return "video", "success", {
                    "format_name": SUPPORTED_VIDEO_FORMATS[file_ext],
                    "extension": file_ext, "mime_type": mime_type
                }
            elif file_ext in SUPPORTED_AUDIO_FORMATS:
                return "audio", "success", {
                    "format_name": SUPPORTED_AUDIO_FORMATS[file_ext],
                    "extension": file_ext, "mime_type": mime_type
                }
            elif mime_type.startswith('video/'):
                return "video", "success", {
                    "format_name": f"Video ({mime_type})",
                    "extension": file_ext, "mime_type": mime_type
                }
            elif mime_type.startswith('audio/'):
                return "audio", "success", {
                    "format_name": f"Audio ({mime_type})",
                    "extension": file_ext, "mime_type": mime_type
                }
            else:
                return None, f"Unsupported format: {file_ext}", {}
        except Exception as e:
            return None, f"Detection failed: {e}", {}
    
    async def analyze_with_ffprobe(self, file_path: str) -> Dict[str, Any]:
        """FFprobe analysis with comprehensive metadata extraction."""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                   '-show_format', '-show_streams', file_path]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.ffprobe_timeout
            )
            
            if process.returncode == 0:
                return self._process_ffprobe_data(json.loads(stdout.decode()))
            else:
                return {"error": f"FFprobe failed: {stderr.decode() if stderr else 'Unknown error'}"}
                
        except asyncio.TimeoutError:
            return {"error": "Analysis timed out"}
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def _process_ffprobe_data(self, probe_data: Dict) -> Dict[str, Any]:
        """Process FFprobe data into structured format."""
        format_info = probe_data.get('format', {})
        streams = probe_data.get('streams', [])
        
        analysis = {
            "format": {
                "duration": float(format_info.get('duration', 0)),
                "size": int(format_info.get('size', 0)),
                "bit_rate": int(format_info.get('bit_rate', 0)),
                "format_name": format_info.get('format_name', ''),
                "nb_streams": int(format_info.get('nb_streams', 0)),
                "tags": format_info.get('tags', {})
            },
            "video_streams": [],
            "audio_streams": [],
            "subtitle_streams": []
        }
        
        for stream in streams:
            codec_type = stream.get('codec_type')
            if codec_type == 'video':
                analysis["video_streams"].append({
                    "codec_name": stream.get('codec_name'),
                    "width": stream.get('width'),
                    "height": stream.get('height'),
                    "r_frame_rate": stream.get('r_frame_rate'),
                    "bit_rate": stream.get('bit_rate'),
                    "pix_fmt": stream.get('pix_fmt')
                })
            elif codec_type == 'audio':
                analysis["audio_streams"].append({
                    "codec_name": stream.get('codec_name'),
                    "sample_rate": stream.get('sample_rate'),
                    "channels": stream.get('channels'),
                    "bit_rate": stream.get('bit_rate')
                })
            elif codec_type == 'subtitle':
                analysis["subtitle_streams"].append({
                    "codec_name": stream.get('codec_name')
                })
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], format_info: Dict[str, Any], 
                       file_name: str) -> str:
        """Generate comprehensive analysis report."""
        report = f"""# 🎬 **Enhanced Media Analysis**

## 📄 **File Information**
- **Name:** {file_name}
- **Format:** {format_info.get('format_name', 'Unknown')}
- **Type:** {format_info.get('extension', 'Unknown')}
- **MIME:** {format_info.get('mime_type', 'Unknown')}

## 📊 **Container Analysis**"""
        
        format_data = analysis.get("format", {})
        duration = format_data.get("duration", 0)
        size = format_data.get("size", 0)
        bitrate = format_data.get("bit_rate", 0)
        
        report += f"""
- **Duration:** {self._format_duration(duration)}
- **File Size:** {self._format_file_size(size)}
- **Bitrate:** {self._format_bitrate(bitrate)}
- **Streams:** {format_data.get('nb_streams', 0)}"""
        
        # Video analysis
        video_streams = analysis.get("video_streams", [])
        if video_streams:
            report += f"\n\n## 📹 **Video Streams** ({len(video_streams)})"
            for i, stream in enumerate(video_streams):
                fps = self._parse_frame_rate(stream.get('r_frame_rate', '0/1'))
                report += f"""
### Stream {i+1}
- **Codec:** {stream.get('codec_name', 'Unknown')}
- **Resolution:** {stream.get('width', 0)}x{stream.get('height', 0)}
- **Frame Rate:** {fps:.2f} fps
- **Pixel Format:** {stream.get('pix_fmt', 'Unknown')}
- **Bitrate:** {self._format_bitrate(stream.get('bit_rate', 0))}"""
        
        # Audio analysis  
        audio_streams = analysis.get("audio_streams", [])
        if audio_streams:
            report += f"\n\n## 🎵 **Audio Streams** ({len(audio_streams)})"
            for i, stream in enumerate(audio_streams):
                report += f"""
### Stream {i+1}
- **Codec:** {stream.get('codec_name', 'Unknown')}
- **Sample Rate:** {stream.get('sample_rate', 0):,} Hz
- **Channels:** {stream.get('channels', 0)}
- **Bitrate:** {self._format_bitrate(stream.get('bit_rate', 0))}"""
        
        # Subtitle streams
        subtitle_streams = analysis.get("subtitle_streams", [])
        if subtitle_streams:
            report += f"\n\n## 📝 **Subtitle Streams** ({len(subtitle_streams)})"
            for i, stream in enumerate(subtitle_streams):
                report += f"\n- Stream {i+1}: {stream.get('codec_name', 'Unknown')}"
        
        # Metadata
        metadata = format_data.get("tags", {})
        if metadata:
            report += f"\n\n## 🏷️ **Metadata**"
            for key, value in list(metadata.items())[:10]:  # Show first 10
                report += f"\n- **{key}:** {value}"
        
        # Technical summary
        complexity = self._calculate_complexity(analysis)
        report += f"""

## ⚙️ **Technical Summary**
- **Analysis:** Complete ✅
- **Complexity:** {complexity}/10
- **Compatible:** Yes
- **Streams Total:** {len(video_streams) + len(audio_streams) + len(subtitle_streams)}"""
        
        return report
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration readably."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m {seconds%60:.0f}s"
        else:
            return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"
    
    def _format_file_size(self, bytes_size: int) -> str:
        """Format file size readably."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"
    
    def _format_bitrate(self, bitrate: int) -> str:
        """Format bitrate readably."""
        if bitrate == 0:
            return "Unknown"
        elif bitrate < 1000:
            return f"{bitrate} bps"
        elif bitrate < 1000000:
            return f"{bitrate/1000:.1f} kbps"
        else:
            return f"{bitrate/1000000:.1f} Mbps"
    
    def _parse_frame_rate(self, fps_str: str) -> float:
        """Parse frame rate from string."""
        try:
            if '/' in fps_str:
                num, denom = fps_str.split('/')
                return float(num) / max(float(denom), 1)
            return float(fps_str)
        except:
            return 0.0
    
    def _calculate_complexity(self, analysis: Dict) -> int:
        """Calculate media complexity score."""
        score = 1
        
        # Stream count
        total_streams = (len(analysis.get("video_streams", [])) + 
                        len(analysis.get("audio_streams", [])) +
                        len(analysis.get("subtitle_streams", [])))
        score += min(total_streams, 4)
        
        # Video complexity
        for stream in analysis.get("video_streams", []):
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            if width * height > 1920 * 1080:
                score += 2
        
        # Audio complexity
        for stream in analysis.get("audio_streams", []):
            if stream.get("channels", 0) > 2:
                score += 1
        
        return min(score, 10)

    def generate_contextual_report(self, analysis: Dict[str, Any], format_info: Dict[str, Any], 
                       file_name: str) -> str:
        """Generate natural, conversational analysis report."""
        format_data = analysis.get("format", {})
        duration = format_data.get("duration", 0)
        video_streams = analysis.get("video_streams", [])
        audio_streams = analysis.get("audio_streams", [])
        
        # Determine content type and purpose
        content_type = self._analyze_content_type(file_name, duration, video_streams, audio_streams)
        quality_assessment = self._assess_quality(video_streams, audio_streams, duration)
        
        # Generate contextual insights
        insights = self._generate_insights(content_type, quality_assessment, duration)
        
        # Build natural response
        response_parts = []
        
        # Start with what this is
        response_parts.append(content_type['description'])
        
        # Add quality assessment naturally
        response_parts.append(f"Quality assessment: {quality_assessment['summary']}")
        
        # Add key details in a natural way
        key_details = [
            f"Duration: {self._format_duration(duration)}",
            f"Purpose: {content_type['purpose']}",
            f"Best suited for: {content_type['use_case']}"
        ]
        response_parts.append("Key details: " + "; ".join(key_details))
        
        # Add insights naturally
        if insights:
            response_parts.append(f"Technical notes: {insights}")
        
        # Add recommendations naturally
        if quality_assessment['recommendations']:
            response_parts.append(f"Recommendations: {quality_assessment['recommendations']}")
        
        return "\n\n".join(response_parts)
    
    def _analyze_content_type(self, filename: str, duration: float, video_streams: List, audio_streams: List) -> Dict[str, str]:
        """Analyze what type of content this appears to be."""
        filename_lower = filename.lower()
        
        # Determine content type based on characteristics
        if video_streams and audio_streams:
            if any(keyword in filename_lower for keyword in ['screen', 'recording', 'capture', 'demo']):
                return {
                    "description": "This appears to be a **screen recording** or software demonstration",
                    "purpose": "Tutorial, demonstration, or documentation",
                    "use_case": "Training, tutorials, bug reports, software documentation"
                }
            elif any(keyword in filename_lower for keyword in ['poc', 'proof', 'concept', 'exploit', 'demo']):
                return {
                    "description": "This appears to be a **proof of concept (PoC)** or demonstration video",
                    "purpose": "Security demonstration or technical proof",
                    "use_case": "Security research, vulnerability demonstration, technical validation"
                }
            elif duration < 30:
                return {
                    "description": "This is a **short video clip** or quick demonstration",
                    "purpose": "Brief demonstration or highlight",
                    "use_case": "Quick sharing, social media, brief explanations"
                }
            else:
                return {
                    "description": "This is a **standard video file** with both video and audio",
                    "purpose": "General multimedia content",
                    "use_case": "Presentation, training, entertainment, documentation"
                }
        elif video_streams and not audio_streams:
            return {
                "description": "This is a **silent video** (video only, no audio)",
                "purpose": "Visual demonstration or screen capture",
                "use_case": "GIFs alternative, silent tutorials, visual documentation"
            }
        elif audio_streams and not video_streams:
            if duration < 60:
                return {
                    "description": "This is a **short audio clip** or voice note",
                    "purpose": "Quick audio message or sound sample",
                    "use_case": "Voice notes, audio samples, sound effects"
                }
            else:
                return {
                    "description": "This is an **audio file** (podcast, music, or voice recording)",
                    "purpose": "Audio content for listening",
                    "use_case": "Podcasts, music, voice recordings, audio documentation"
                }
        else:
            return {
                "description": "This appears to be a **media container** with unknown content",
                "purpose": "Unknown media type",
                "use_case": "Requires further investigation"
            }
    
    def _assess_quality(self, video_streams: List, audio_streams: List, duration: float) -> Dict[str, str]:
        """Assess the quality and provide user-friendly recommendations."""
        quality_score = 0
        issues = []
        strengths = []
        
        # Video quality assessment
        if video_streams:
            for stream in video_streams:
                width = stream.get("width", 0)
                height = stream.get("height", 0)
                fps = self._parse_frame_rate(stream.get('r_frame_rate', '0/1'))
                
                # Resolution assessment
                if width >= 1920 and height >= 1080:
                    strengths.append("High resolution (Full HD or higher)")
                    quality_score += 3
                elif width >= 1280 and height >= 720:
                    strengths.append("Good resolution (HD)")
                    quality_score += 2
                else:
                    issues.append("Low resolution (below HD)")
                
                # Frame rate assessment
                if fps >= 30:
                    strengths.append("Smooth frame rate")
                    quality_score += 2
                elif fps >= 24:
                    quality_score += 1
                else:
                    issues.append("Low frame rate (choppy playback)")
        
        # Audio quality assessment
        if audio_streams:
            for stream in audio_streams:
                sample_rate = stream.get("sample_rate", 0)
                channels = stream.get("channels", 0)
                
                if sample_rate >= 44100:
                    strengths.append("Good audio quality")
                    quality_score += 2
                elif sample_rate >= 22050:
                    quality_score += 1
                else:
                    issues.append("Low audio quality")
                
                if channels >= 2:
                    strengths.append("Stereo audio")
                    quality_score += 1
        
        # Duration assessment
        if duration > 0:
            if duration < 5:
                issues.append("Very short duration")
            elif duration > 3600:
                strengths.append("Long-form content")
        
        # Generate summary
        if quality_score >= 6:
            summary = "**Excellent quality** - Professional grade content"
        elif quality_score >= 4:
            summary = "**Good quality** - Suitable for most purposes"
        elif quality_score >= 2:
            summary = "**Acceptable quality** - Basic viewing/listening"
        else:
            summary = "**Low quality** - May need improvement"
        
        # Generate recommendations
        recommendations = []
        if issues:
            recommendations.append("**Issues to consider:** " + "; ".join(issues))
        if strengths:
            recommendations.append("**Strengths:** " + "; ".join(strengths))
        
        if not video_streams and audio_streams:
            recommendations.append("• Perfect for podcasts or voice content")
        elif video_streams and duration < 60:
            recommendations.append("• Great for quick demos or tutorials")
        elif quality_score >= 4:
            recommendations.append("• Suitable for professional presentations")
        
        return {
            "summary": summary,
            "recommendations": "\n".join(recommendations) if recommendations else "No specific recommendations"
        }
    
    def _generate_insights(self, content_type: Dict, quality: Dict, duration: float) -> str:
        """Generate technical insights in user-friendly language."""
        insights = []
        
        if "screen recording" in content_type["description"]:
            insights.append("Optimized for screen capture and software demonstrations")
        if "proof of concept" in content_type["description"].lower():
            insights.append("Contains technical demonstration or security research content")
        if duration < 30:
            insights.append("Short format - ideal for quick sharing and embedding")
        elif duration > 600:
            insights.append("Long format - suitable for detailed tutorials or presentations")
        
        return "; ".join(insights) if insights else "Standard media file with typical characteristics"

# Global analyzer
media_analyzer = EnhancedMediaAnalyzer()

async def handle_media_upload(uploaded_file) -> str:
    """Enhanced media handler supporting 50+ formats."""
    try:
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Handle different file object types from Chainlit
            if hasattr(uploaded_file, 'content') and uploaded_file.content:
                # File has content attribute
                content = uploaded_file.content
            elif hasattr(uploaded_file, 'path') and uploaded_file.path:
                # File has path attribute
                with open(uploaded_file.path, 'rb') as f:
                    content = f.read()
            else:
                # Try to read directly if it's a file-like object
                if hasattr(uploaded_file, 'read'):
                    try:
                        content = uploaded_file.read()
                    except Exception as e:
                        return f"❌ **Unsupported File Object**\n\n**File:** {uploaded_file.name}\n**Issue:** Cannot read file content"
                else:
                    return f"❌ **Unsupported File Object**\n\n**File:** {uploaded_file.name}\n**Issue:** Cannot read file content"
            
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Cache check
            file_hash = hashlib.md5(content).hexdigest()
            if file_hash in MEDIA_ANALYSIS_CACHE:
                return MEDIA_ANALYSIS_CACHE[file_hash]
            
            # Format detection
            media_type, status, format_info = media_analyzer.detect_media_format(temp_file_path)
            
            if not media_type:
                return f"I couldn't analyze this media file '{uploaded_file.name}': {status}. This system supports {len(SUPPORTED_VIDEO_FORMATS) + len(SUPPORTED_AUDIO_FORMATS)} different video and audio formats. Please check if your file is a supported media format."
            
            # Analysis
            analysis = await media_analyzer.analyze_with_ffprobe(temp_file_path)
            
            if "error" in analysis:
                return f"Analysis failed for {uploaded_file.name}: {analysis['error']}"
            
            # Generate contextual report instead of technical metadata
            report = media_analyzer.generate_contextual_report(analysis, format_info, uploaded_file.name)
            
            # Cache result
            MEDIA_ANALYSIS_CACHE[file_hash] = report
            if len(MEDIA_ANALYSIS_CACHE) > 50:
                MEDIA_ANALYSIS_CACHE.pop(next(iter(MEDIA_ANALYSIS_CACHE)))
            
            return report
        finally:
            os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Media processing error: {e}")
        return f"❌ **Processing Error**\n\n**File:** {uploaded_file.name}\n**Error:** {str(e)}"

def get_supported_formats() -> Dict[str, int]:
    """Get supported format counts."""
    return {
        "video_formats": len(SUPPORTED_VIDEO_FORMATS),
        "audio_formats": len(SUPPORTED_AUDIO_FORMATS),
        "total_formats": len(SUPPORTED_VIDEO_FORMATS) + len(SUPPORTED_AUDIO_FORMATS)
    }
