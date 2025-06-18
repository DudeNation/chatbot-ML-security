import os
from openai import OpenAI
import logging
import base64
import re
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Generation cache and rate limiting
IMAGE_GENERATION_CACHE = {}
LAST_GENERATION_TIME = {}

def sanitize_prompt(prompt: str) -> str:
    """Sanitize and improve image generation prompts."""
    # Remove potentially harmful content
    harmful_patterns = [
        r'\b(nude|naked|explicit|nsfw)\b',
        r'\b(violence|gore|blood)\b',
        r'\b(illegal|harmful|dangerous)\b'
    ]
    
    for pattern in harmful_patterns:
        prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE)
    
    # Clean up the prompt
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    
    return prompt

def enhance_cybersecurity_prompt(prompt: str) -> str:
    """Enhance prompts for cybersecurity-related image generation."""
    cybersecurity_keywords = [
        'hacking', 'security', 'cyber', 'network', 'firewall', 'malware',
        'vulnerability', 'penetration', 'forensics', 'encryption', 'authentication'
    ]
    
    # Check if prompt is cybersecurity-related
    is_cybersec = any(keyword in prompt.lower() for keyword in cybersecurity_keywords)
    
    if is_cybersec:
        # Add professional, educational context
        enhanced_prompt = f"""Professional cybersecurity illustration: {prompt}. 
        Style: Clean, modern, technical diagram style. Educational and professional appearance. 
        Include elements like: computer screens, network diagrams, security icons, locks, shields.
        Avoid: Any inappropriate or harmful content. Focus on legitimate security education."""
        return enhanced_prompt
    
    return prompt

def get_style_enhancement(style: str = "professional") -> str:
    """Get style enhancement based on context."""
    styles = {
        "professional": "Clean, professional, business-appropriate, high-quality digital art",
        "technical": "Technical diagram style, clean lines, informative, schematic-like",
        "educational": "Educational illustration, clear and informative, textbook style",
        "creative": "Creative and artistic, visually appealing, modern design",
        "minimalist": "Minimalist design, clean and simple, focused composition"
    }
    
    return styles.get(style, styles["professional"])

async def generate_image_advanced(prompt: str, size: str = "1024x1024", quality: str = "standard", style: str = "vivid") -> Dict[str, Any]:
    """Generate an image with advanced options using DALL-E 3."""
    try:
        logger.info(f"Generating image with prompt: {prompt[:50]}...")
        
        # Validate parameters
        valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        valid_qualities = ["standard", "hd"]
        valid_styles = ["vivid", "natural"]
        
        if size not in valid_sizes:
            size = "1024x1024"
        if quality not in valid_qualities:
            quality = "standard"
        if style not in valid_styles:
            style = "vivid"
        
        # Generate image
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1
        )
        
        # Extract image URL
        image_url = response.data[0].url
        
        # Return simple result with just the image URL
        return {
            "image_url": image_url
        }
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return {"error": str(e)}

async def generate_image(prompt: str) -> Dict[str, Any]:
    """Generate an image using DALL-E 3 with default settings."""
    return await generate_image_advanced(prompt)

async def generate_cybersecurity_diagram(prompt: str) -> Dict[str, Any]:
    """Generate a cybersecurity diagram or network architecture visualization."""
    enhanced_prompt = f"Create a professional, clear cybersecurity diagram showing {prompt}. Use a clean, technical style with labeled components and clear connections. Make it suitable for a technical presentation or documentation."
    return await generate_image_advanced(enhanced_prompt, size="1024x1024", quality="hd", style="natural")

async def generate_security_themed_image(prompt: str) -> Dict[str, Any]:
    """Generate a security-themed image with professional styling."""
    enhanced_prompt = f"Create a professional cybersecurity-themed image representing {prompt}. Use modern, clean visual style with appropriate security iconography. Make it suitable for a professional security presentation or blog."
    return await generate_image_advanced(enhanced_prompt)

async def generate_image_variation(image_path: str, size: str = "1024x1024", n: int = 1) -> str:
    """Enhanced image variation generation."""
    try:
        logger.info(f"Generating image variation from: {image_path}")
        
        # Validate size
        valid_sizes = ["256x256", "512x512", "1024x1024"]
        if size not in valid_sizes:
            logger.warning(f"Invalid size {size} for variations, defaulting to 1024x1024")
            size = "1024x1024"
        
        # Check file exists and size
        if not os.path.exists(image_path):
            return "Error: Image file not found"
        
        file_size = os.path.getsize(image_path)
        if file_size > 4 * 1024 * 1024:  # 4MB limit
            return "Error: Image file too large (max 4MB)"
        
        with open(image_path, "rb") as image_file:
            response = client.images.create_variation(
                image=image_file,
                size=size,
                n=n,
            )

        image_url = response.data[0].url
        logger.info("Image variation generated successfully")
        return image_url
        
    except Exception as e:
        logger.error(f"Error generating image variation: {str(e)}", exc_info=True)
        return f"Error generating image variation: {str(e)}"

def get_generation_stats() -> Dict[str, Any]:
    """Get image generation statistics."""
    try:
        total_generations = len(IMAGE_GENERATION_CACHE)
        recent_generations = sum(1 for result in IMAGE_GENERATION_CACHE.values() 
                               if (datetime.now() - datetime.fromisoformat(result["timestamp"])).seconds < 3600)
        
        return {
            "total_cached": total_generations,
            "recent_hour": recent_generations,
            "cache_size_mb": len(str(IMAGE_GENERATION_CACHE)) / (1024 * 1024),
            "most_recent": max(IMAGE_GENERATION_CACHE.values(), 
                             key=lambda x: x["timestamp"])["timestamp"] if IMAGE_GENERATION_CACHE else None
        }
    except Exception as e:
        logger.error(f"Error getting generation stats: {str(e)}")
        return {"error": str(e)}
