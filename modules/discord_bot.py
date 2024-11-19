import discord
from discord.ext import commands
import asyncio
import chainlit as cl
from chainlit.types import AskFileResponse
import os
from dotenv import load_dotenv
from modules.image_analysis import analyze_image
from modules.image_generation import generate_image
import io
from collections import deque
import json
from discord.ext.commands import Bot, Context

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_CHANNEL_IDS = list(map(int, os.getenv("ALLOWED_CHANNEL_IDS", "")))

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize conversation history
user_history = {}
MAX_HISTORY_LENGTH = 5

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    if bot.user in message.mentions:
        await process_message(message)

async def process_message(message: discord.Message):
    if message.channel.id not in ALLOWED_CHANNEL_IDS:
        await message.channel.send("I'm not allowed to respond in this channel.")
        return

    user_id = str(message.author.id)
    if user_id not in user_history:
        user_history[user_id] = deque(maxlen=MAX_HISTORY_LENGTH)

    # Remove the bot mention from the message content
    content = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()

    async with message.channel.typing():
        try:
            # Prepare the context with conversation history
            context = "\n".join([f"User: {item['user']}\nAssistant: {item['bot']}" for item in user_history[user_id]])
            full_query = f"Given the conversation history:\n{context}\n\nUser query: {content}\n\nPlease provide a relevant and accurate response to the user's query."

            result = await cl.AskUserMessage(content=full_query).send()
            if isinstance(result, cl.AskFileResponse):
                response_content = result.content
                await message.channel.send(response_content)
                
                # Update conversation history
                user_history[user_id].append({"user": content, "bot": response_content})
            else:
                await message.channel.send("An error occurred while processing your request.")
        except Exception as e:
            await message.channel.send(f"An error occurred: {str(e)}")

    # Check for image attachments
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_analysis = await analyze_image_attachment(attachment)
                await message.channel.send(f"Image analysis: {image_analysis}")

async def analyze_image_attachment(attachment: discord.Attachment) -> str:
    image_data = await attachment.read()
    image = cl.Image(content=image_data, name=attachment.filename)
    return await analyze_image(image)

@bot.command()
async def history(ctx):
    if ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        await ctx.send("This command is not allowed in this channel.")
        return

    user_id = str(ctx.author.id)
    if user_id not in user_history or not user_history[user_id]:
        await ctx.send("You don't have any conversation history.")
        return

    history_text = "Your conversation history:\n\n"
    for item in user_history[user_id]:
        history_text += f"User: {item['user']}\nAssistant: {item['bot']}\n\n"

    await ctx.send(history_text)

@bot.command()
async def clear_history(ctx):
    if ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        await ctx.send("This command is not allowed in this channel.")
        return

    user_id = str(ctx.author.id)
    if user_id in user_history:
        user_history[user_id].clear()
        await ctx.send("Your conversation history has been cleared.")
    else:
        await ctx.send("You don't have any conversation history to clear.")

def run_discord_bot():
    bot.run(DISCORD_BOT_TOKEN)

if __name__ == "__main__":
    run_discord_bot()
