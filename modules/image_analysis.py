import io
import base64
from PIL import Image
import chainlit as cl
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def analyze_image(image_element: cl.Image) -> str:
    try:
        logger.info("Starting image analysis")
        if not image_element.content and not image_element.path:
            logger.error("No image content or path provided")
            return None

        if image_element.content:
            logger.info("Using image content")
            image_data = image_element.content
        elif image_element.path:
            logger.info(f"Reading image from path: {image_element.path}")
            with open(image_element.path, "rb") as image_file:
                image_data = image_file.read()
        else:
            logger.error("No valid image data found")
            return None

        logger.info("Opening image with PIL")
        image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"Image mode: {image.mode}")
        if image.mode != 'RGB':
            logger.info("Converting image to RGB")
            image = image.convert('RGB')
        
        logger.info("Saving image to byte stream")
        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_stream.seek(0)
        
        logger.info("Analyzing image with OpenAI's vision model")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image and provide a detailed description, focusing on any cybersecurity-related elements if present."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(byte_stream.getvalue()).decode('utf-8')}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        
        logger.info("Image analysis completed successfully")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
        return None