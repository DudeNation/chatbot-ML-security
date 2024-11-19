import chainlit as cl
from chatbot import setup_vector_indices, setup_query_engine
from llama_index.agent.openai import OpenAIAgent
import logging
import glob
from modules.index_manager import update_indices_if_needed
from modules.auth import oauth_callback
from chainlit.types import ThreadDict
from llama_index.core.memory import ChatMemoryBuffer
from chainlit.input_widget import Select, Switch, Slider
from llama_index.llms.openai import OpenAI as LlamaOpenAI
import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64
from modules.image_analysis import analyze_image
from modules.image_generation import generate_image, generate_image_variation
from modules.file_handler import handle_file_upload, handle_url
from modules.media_handler import handle_media_upload
from modules.discord_bot import run_discord_bot
import threading

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store indices and query engine
global_index_set = None
global_query_engine = None
global_tools = None

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# Default settings
DEFAULT_SETTINGS = {
    "Model": "gpt-4o",
    "Streaming": True,
    "Temperature": 1.0,
    "SAI_Steps": 30,
    "SAI_Cfg_Scale": 7,
    "SAI_Width": 512,
    "SAI_Height": 512
}

async def initialize_agent(blog_files, settings=None):
    global global_index_set, global_query_engine, global_tools
    
    if settings is None:
        settings = DEFAULT_SETTINGS
    
    if global_index_set is None:
        try:
            await update_indices_if_needed(blog_files)
            global_index_set = await cl.make_async(setup_vector_indices)(blog_files)
        except ValueError as e:
            logger.error(f"Error setting up vector indices: {str(e)}")
            await cl.Message(content=f"Error: {str(e)}. The assistant may not have access to all information.").send()
            global_index_set = {}  # Initialize with an empty dict to allow the rest of the setup to continue
        
        if global_index_set:
            global_query_engine, global_tools = await cl.make_async(setup_query_engine)(global_index_set, blog_files)
        else:
            logger.warning("No indices were created. Initializing agent without query engine.")
            global_query_engine, global_tools = None, []

    memory = ChatMemoryBuffer.from_defaults(token_limit=10000)
    llm = LlamaOpenAI(
        model=settings.get("Model", DEFAULT_SETTINGS["Model"]),
        temperature=settings.get("Temperature", DEFAULT_SETTINGS["Temperature"]),
        streaming=settings.get("Streaming", DEFAULT_SETTINGS["Streaming"]),
        api_key=openai_api_key
    )
    agent = OpenAIAgent.from_tools(
        global_tools,
        verbose=True,
        streaming=settings.get("Streaming", DEFAULT_SETTINGS["Streaming"]),
        memory=memory,
        llm=llm
    )
    return agent

@cl.on_chat_start
async def start():
    logger.info("Starting new chat session")
    cl.user_session.set("history", [])
    blog_files = glob.glob("./data/*.html")
    
    try:
        settings = await cl.ChatSettings(
            [
                Select(
                    id="Model",
                    label="OpenAI - Model",
                    values=["gpt-4o", "gpt-4o-mini"],
                    initial_index=0,
                ),
                Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
                Slider(
                    id="Temperature",
                    label="OpenAI - Temperature",
                    initial=1,
                    min=0,
                    max=2,
                    step=0.1,
                ),
                Slider(
                    id="SAI_Steps",
                    label="Stability AI - Steps",
                    initial=30,
                    min=10,
                    max=150,
                    step=1,
                    description="Amount of inference steps performed on image generation.",
                ),
                Slider(
                    id="SAI_Cfg_Scale",
                    label="Stability AI - Cfg_Scale",
                    initial=7,
                    min=1,
                    max=35,
                    step=0.1,
                    description="Influences how strongly your generation is guided to match your prompt.",
                ),
                Slider(
                    id="SAI_Width",
                    label="Stability AI - Image Width",
                    initial=512,
                    min=256,
                    max=2048,
                    step=64,
                    tooltip="Measured in pixels",
                ),
                Slider(
                    id="SAI_Height",
                    label="Stability AI - Image Height",
                    initial=512,
                    min=256,
                    max=2048,
                    step=64,
                    tooltip="Measured in pixels",
                ),
                Slider(
                    id="Image_Size",
                    label="Image Generation - Size",
                    initial=1024,
                    min=256,
                    max=1024,
                    step=256,
                    tooltip="Size of the generated image (square)",
                ),
                Select(
                    id="Image_Quality",
                    label="Image Generation - Quality",
                    values=["standard", "hd"],
                    initial_index=0,
                ),
            ]
        ).send()
        cl.user_session.set("settings", settings)
        
        agent = await initialize_agent(blog_files, settings)
        cl.user_session.set("agent", agent)
        logger.info("Agent setup complete!")
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}", exc_info=True)
        await cl.Message(content=f"I'm having trouble setting up. Error: {str(e)}. Please try again later or contact support if the issue persists.").send()
        return

@cl.on_settings_update
async def setup_agent(settings):
    logger.info(f"Updating settings: {settings}")
    blog_files = glob.glob("./data/*.html")
    agent = await initialize_agent(blog_files, settings)
    cl.user_session.set("agent", agent)
    cl.user_session.set("settings", settings)
    
    # Ensure the settings are immediately available
    await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-4o", "gpt-4o-mini"],
                initial_index=0 if settings.get("Model") == "gpt-4o" else 1,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=settings.get("Streaming", DEFAULT_SETTINGS["Streaming"])),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=settings.get("Temperature", DEFAULT_SETTINGS["Temperature"]),
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="SAI_Steps",
                label="Stability AI - Steps",
                initial=settings.get("SAI_Steps", DEFAULT_SETTINGS["SAI_Steps"]),
                min=10,
                max=150,
                step=1,
                description="Amount of inference steps performed on image generation.",
            ),
            Slider(
                id="SAI_Cfg_Scale",
                label="Stability AI - Cfg_Scale",
                initial=settings.get("SAI_Cfg_Scale", DEFAULT_SETTINGS["SAI_Cfg_Scale"]),
                min=1,
                max=35,
                step=0.1,
                description="Influences how strongly your generation is guided to match your prompt.",
            ),
            Slider(
                id="SAI_Width",
                label="Stability AI - Image Width",
                initial=settings.get("SAI_Width", DEFAULT_SETTINGS["SAI_Width"]),
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
            Slider(
                id="SAI_Height",
                label="Stability AI - Image Height",
                initial=settings.get("SAI_Height", DEFAULT_SETTINGS["SAI_Height"]),
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
            Slider(
                id="Image_Size",
                label="Image Generation - Size",
                initial=1024,
                min=256,
                max=1024,
                step=256,
                tooltip="Size of the generated image (square)",
            ),
            Select(
                id="Image_Quality",
                label="Image Generation - Quality",
                values=["standard", "hd"],
                initial_index=0,
            ),
        ]
    ).send()

@cl.on_message
async def main(message: cl.Message):
    logger.info(f"Received message: {message.content}")
    agent = cl.user_session.get("agent")
    settings = cl.user_session.get("settings")
    
    if not agent:
        logger.error("Agent not found in user session")
        await cl.Message(content="I'm sorry, but I'm not properly set up. Please try restarting the conversation or contact support.").send()
        return

    history = cl.user_session.get("history", [])
    
    # Handle file uploads
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                file_content = await handle_file_upload(element)
                if file_content.startswith("Error:"):
                    await cl.Message(content=file_content).send()
                    return
                message.content += f"\n\nFile content: {file_content}"
            elif isinstance(element, (cl.Audio, cl.Video)):
                media_content = await handle_media_upload(element)
                if media_content.startswith("Error:"):
                    await cl.Message(content=media_content).send()
                    return
                message.content += f"\n\nMedia content: {media_content}"

    # Check if the message is an image generation request
    if message.content.lower().startswith("generate image:"):
        prompt = message.content[15:].strip()
        size = f"{settings.get('Image_Size', 1024)}x{settings.get('Image_Size', 1024)}"
        quality = settings.get("Image_Quality", "standard")
        
        image_url = await generate_image(prompt, size, quality)
        
        if image_url.startswith("Error"):
            await cl.Message(content=image_url).send()
        else:
            await cl.Message(content="", elements=[cl.Image(url=image_url)]).send()
        
        history.append({"user": message.content, "bot": f"Generated image: {image_url}"})
        cl.user_session.set("history", history)
        return

    # Handle URL processing
    if message.content.lower().startswith("process url:"):
        url = message.content[12:].strip()
        url_content = await handle_url(url)
        message.content += f"\n\nURL content: {url_content}"

    # Check if an image is attached to the message
    image_analysis = ""
    if message.elements:
        logger.info(f"Message contains {len(message.elements)} elements")
        for element in message.elements:
            if isinstance(element, cl.Image):
                logger.info(f"Image element found: {element}")
                try:    
                    # Analyze the image
                    logger.info("Analyzing image...")
                    image_analysis = await analyze_image(element)
                    logger.info(f"Image analysis result: {image_analysis}")
                    
                    # Add image analysis to the message content
                    message.content += f"\n\nImage analysis: {image_analysis}"
                    
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}", exc_info=True)
                    await cl.Message(content=f"I'm sorry, but I encountered an error while processing the image: {str(e)}. Could you please try uploading it again?").send()
            else:
                logger.info(f"Non-image element found: {type(element)}")
    else:
        logger.info("No elements found in the message")

    if not image_analysis:
        logger.info("No image analysis performed")

    context = "\n".join([f"User: {item['user']}\nAssistant: {item['bot']}" for item in history[-3:]])
    full_query = f"Given the conversation history:\n{context}\n\nUser query: {message.content}\n\nPlease provide a relevant and accurate response to the user's query, using the information from the cybersecurity blogs and any image or media analysis provided."

    try:
        response_message = cl.Message(content="")
        await response_message.send()

        full_response = ""
        if settings.get("Streaming", DEFAULT_SETTINGS["Streaming"]):
            streaming_response = await cl.make_async(agent.stream_chat)(full_query)
            for token in streaming_response.response_gen:
                full_response += token
                await response_message.stream_token(token)
        else:
            response = await cl.make_async(agent.chat)(full_query)
            full_response = response.response
            await response_message.update(content=full_response)

        # Update the final content
        response_message.content = full_response
        await response_message.update()
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await cl.Message(content="I apologize, but I encountered an error while processing your query. Could you please rephrase or ask another question?").send()
        return

    history.append({"user": message.content, "bot": full_response})
    cl.user_session.set("history", history)
    logger.info("Response sent and history updated")

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    logger.info(f"Resuming chat session for thread: {thread['id']}")
    
    blog_files = glob.glob("./data/*.html")
    settings = cl.user_session.get("settings")
    
    if settings is None:
        logger.warning("Settings not found in user session. Using default settings.")
        settings = DEFAULT_SETTINGS
    
    # Restore chat settings
    await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-4o", "gpt-4o-mini"],
                initial_index=0 if settings.get("Model") == "gpt-4o" else 1,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=settings.get("Streaming", DEFAULT_SETTINGS["Streaming"])),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=settings.get("Temperature", DEFAULT_SETTINGS["Temperature"]),
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="SAI_Steps",
                label="Stability AI - Steps",
                initial=settings.get("SAI_Steps", DEFAULT_SETTINGS["SAI_Steps"]),
                min=10,
                max=150,
                step=1,
                description="Amount of inference steps performed on image generation.",
            ),
            Slider(
                id="SAI_Cfg_Scale",
                label="Stability AI - Cfg_Scale",
                initial=settings.get("SAI_Cfg_Scale", DEFAULT_SETTINGS["SAI_Cfg_Scale"]),
                min=1,
                max=35,
                step=0.1,
                description="Influences how strongly your generation is guided to match your prompt.",
            ),
            Slider(
                id="SAI_Width",
                label="Stability AI - Image Width",
                initial=settings.get("SAI_Width", DEFAULT_SETTINGS["SAI_Width"]),
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
            Slider(
                id="SAI_Height",
                label="Stability AI - Image Height",
                initial=settings.get("SAI_Height", DEFAULT_SETTINGS["SAI_Height"]),
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
            Slider(
                id="Image_Size",
                label="Image Generation - Size",
                initial=1024,
                min=256,
                max=1024,
                step=256,
                tooltip="Size of the generated image (square)",
            ),
            Select(
                id="Image_Quality",
                label="Image Generation - Quality",
                values=["standard", "hd"],
                initial_index=0,
            ),
        ]
    ).send()
    
    agent = await initialize_agent(blog_files, settings)
    cl.user_session.set("agent", agent)
    cl.user_session.set("settings", settings)
    
    # Repopulate the memory with the conversation history
    if 'messages' in thread:
        history = []
        for message in thread['messages']:
            if message['role'] == 'human':
                history.append({"user": message['content'], "bot": ""})
            elif message['role'] == 'assistant':
                if history:
                    history[-1]["bot"] = message['content']
                else:
                    history.append({"user": "", "bot": message['content']})
        cl.user_session.set("history", history)
        logger.info(f"Restored {len(history)} interactions to conversation history")
    else:
        logger.warning("No messages found in thread. Starting with empty history.")
    
    logger.info("Chat session resumed with updated memory and restored settings.")

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Red Team Basics",
            message="Can you explain the basics of a Red Team in cybersecurity? What are their main objectives and how do they differ from Blue Teams?",
            icon="/public/red-team.svg",
        ),
        cl.Starter(
            label="Bug Bounty Programs",
            message="What are bug bounty programs and how do they work? Can you provide some examples of well-known bug bounty platforms?",
            icon="/public/bug-bounty.svg",
        ),
        cl.Starter(
            label="Common Vulnerabilities",
            message="What are some of the most common vulnerabilities that Red Teams and bug bounty hunters look for? Can you explain one in detail?",
            icon="/public/vulnerability.svg",
        ),
        cl.Starter(
            label="Getting Started in Bug Bounties",
            message="I'm interested in getting started with bug bounty hunting. What skills should I develop and what resources would you recommend for beginners?",
            icon="/public/getting-started.svg",
        )
    ]

if __name__ == "__main__":
    logger.info("Starting Chainlit app and Discord bot")
    discord_thread = threading.Thread(target=run_discord_bot)
    discord_thread.start()
    cl.run()
