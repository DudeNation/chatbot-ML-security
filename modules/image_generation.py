import os
from openai import OpenAI
import logging
import base64

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_image(prompt: str, size: str = "1024x1024", quality: str = "standard") -> str:
    try:
        logger.info(f"Generating image with prompt: {prompt}")
        # Ensure the size is in the correct format
        if size not in ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]:
            logger.warning(f"Invalid size {size}, defaulting to 1024x1024")
            size = "1024x1024"
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )

        image_url = response.data[0].url
        logger.info("Image generated successfully")
        return image_url
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}", exc_info=True)
        return f"Error generating image: {str(e)}"

async def generate_image_variation(image_path: str, size: str = "1024x1024", n: int = 1) -> str:
    try:
        logger.info(f"Generating image variation from: {image_path}")
        # Ensure the size is in the correct format
        if size not in ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]:
            logger.warning(f"Invalid size {size}, defaulting to 1024x1024")
            size = "1024x1024"
        
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
