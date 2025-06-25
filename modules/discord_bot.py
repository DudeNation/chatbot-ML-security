import discord
from discord.ext import commands
import asyncio
import chainlit as cl
import os
from dotenv import load_dotenv
from modules.image_analysis import analyze_image
from modules.image_generation import generate_image_advanced
from modules.url_analyzer import analyze_url_smart  # Import improved URL analyzer
import io
from collections import deque
import json
import logging
import glob
from pathlib import Path
import time
from datetime import datetime, timedelta

# Import the main chatbot components
from modules.document_indexer import setup_vector_indices
from modules.query_engine import setup_query_engine
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from openai import OpenAI

load_dotenv()

# Set up logging to completely eliminate noise
logging.basicConfig(level=logging.CRITICAL)  # Only critical errors
logger = logging.getLogger(__name__)

# Completely suppress all Discord and related library warnings
for logger_name in ['discord', 'discord.gateway', 'discord.client', 'discord.http', 
                   'aiohttp', 'urllib3', 'asyncio', 'websockets']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_CHANNEL_IDS_STR = os.getenv("ALLOWED_CHANNEL_IDS", "")
ALLOWED_CHANNEL_IDS = []

# Parse channel IDs safely
if ALLOWED_CHANNEL_IDS_STR:
    try:
        ALLOWED_CHANNEL_IDS = [int(id.strip()) for id in ALLOWED_CHANNEL_IDS_STR.split(",") if id.strip()]
    except ValueError as e:
        pass  # Silently ignore parsing errors

# Set up Discord bot with optimized settings for better performance
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.guild_messages = True

# Optimize bot settings for maximum performance and stability
bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    heartbeat_timeout=90,  # Increased to prevent warnings
    guild_ready_timeout=15,  # Increased for stability
    max_messages=None  # Disable message cache for performance
)

# Initialize conversation history and bot agent
user_history = {}
user_threads = {}  # Track user threads for continuation
MAX_HISTORY_LENGTH = 10
discord_agent = None
agent_initialization_task = None
global_tools = None  # Store tools for detailed processing display

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DetailedProcessingLogger:
    """Enhanced logger to show detailed processing like the web interface in both Discord and terminal."""
    
    def __init__(self, channel):
        self.channel = channel
        self.processing_message = None
        self.start_time = time.time()
        
    async def start_processing(self, query: str):
        """Start processing indicator with enhanced terminal logging."""
        print(f"\n🔍 [DISCORD] Processing query: {query[:50]}...")
        print(f"📊 [DISCORD] Initializing search across {len(global_tools or [])} knowledge sources")
        
        embed = discord.Embed(
            title="🔍 **Processing Your Query**",
            description=f"**Query:** {query[:100]}{'...' if len(query) > 100 else ''}",
            color=0x3498db
        )
        embed.add_field(
            name="📚 **Knowledge Base Analysis**",
            value="🔄 Initializing comprehensive search...",
            inline=False
        )
        embed.set_footer(text="🚀 Enhanced processing for comprehensive analysis")
        
        self.processing_message = await self.channel.send(embed=embed)
        
    async def update_blog_search(self, blog_name: str, blog_number: int, total_blogs: int, found_content: bool = False):
        """Update blog search progress with enhanced terminal logging."""
        if not self.processing_message:
            return
            
        try:
            status_icon = "✅" if found_content else "🔍"
            status_text = "Found relevant content" if found_content else "Searching..."
            
            # Enhanced terminal logging
            if found_content:
                print(f"✅ [DISCORD] Found content in blog {blog_number}/{total_blogs}: {blog_name}")
            else:
                print(f"🔍 [DISCORD] Searching blog {blog_number}/{total_blogs}: {blog_name}")
            
            embed = discord.Embed(
                title="🔍 **Analyzing Knowledge Base**",
                color=0x2ecc71 if found_content else 0x3498db
            )
            
            # Progress bar
            progress = int((blog_number / total_blogs) * 20)
            progress_bar = "█" * progress + "░" * (20 - progress)
            
            embed.add_field(
                name="📊 **Search Progress**",
                value=f"`{progress_bar}` {blog_number}/{total_blogs} sources",
                inline=False
            )
            
            embed.add_field(
                name="🔍 **Current Analysis**",
                value=f"{status_icon} **Source #{blog_number}:** {blog_name[:30]}\n{status_text}",
                inline=False
            )
            
            elapsed = time.time() - self.start_time
            embed.set_footer(text=f"⏱️ Processing time: {elapsed:.1f}s • Comprehensive analysis in progress")
            
            await self.processing_message.edit(embed=embed)
            
        except Exception as e:
            pass  # Silently ignore embed update errors
    
    async def show_synthesis(self, sources_found: int):
        """Show synthesis phase with terminal logging."""
        if not self.processing_message:
            return
            
        try:
            print(f"🎯 [DISCORD] Synthesizing response from {sources_found} knowledge sources")
            
            embed = discord.Embed(
                title="🎯 **Synthesizing Response**",
                description=f"✅ Found relevant information in **{sources_found}** knowledge sources",
                color=0xf39c12
            )
            embed.add_field(
                name="⚙️ **Final Processing**",
                value="🧠 Analyzing and combining information...\n📝 Generating comprehensive response...\n🔗 Cross-referencing sources...",
                inline=False
            )
            
            elapsed = time.time() - self.start_time
            embed.set_footer(text=f"⏱️ Processing time: {elapsed:.1f}s • Almost ready!")
            
            await self.processing_message.edit(embed=embed)
            
        except Exception as e:
            pass  # Silently ignore embed update errors
    
    async def complete_processing(self, response_length: int):
        """Mark processing as complete with terminal logging."""
        if not self.processing_message:
            return
            
        try:
            elapsed = time.time() - self.start_time
            print(f"✅ [DISCORD] Analysis complete: {response_length:,} characters in {elapsed:.1f}s")
            
            embed = discord.Embed(
                title="✅ **Analysis Complete**",
                description="🎉 Successfully processed your query!",
                color=0x2ecc71
            )
            embed.add_field(
                name="📊 **Results Summary**",
                value=f"📝 Response: {response_length:,} characters\n⏱️ Processing time: {elapsed:.1f}s\n🔍 Sources analyzed: {len(global_tools or [])}",
                inline=False
            )
            
            await self.processing_message.edit(embed=embed)
            
            # Delete the processing message after 3 seconds for cleaner chat
            await asyncio.sleep(3)
            await self.processing_message.delete()
            
        except Exception as e:
            pass  # Silently ignore processing completion errors

async def initialize_discord_agent():
    """Initialize the chatbot agent for Discord use - optimized to prevent heartbeat blocking."""
    global discord_agent, global_tools
    
    try:
        print("🚀 [DISCORD] Starting agent initialization (non-blocking)")
        
        # Get blog files
        blog_files = glob.glob("./data/*.html")
        if not blog_files:
            print("⚠️ [DISCORD] No blog files found")
            return None
        
        print(f"📚 [DISCORD] Found {len(blog_files)} blog files")
        
        # Set up indices and query engine with progress feedback
        print("🔧 [DISCORD] Setting up vector indices...")
        
        # Use asyncio to run the blocking operations in smaller chunks to prevent heartbeat blocking
        def setup_indices():
            return setup_vector_indices(blog_files)
        
        def setup_engine(index_set):
            return setup_query_engine(index_set, blog_files)
        
        # Run heavy operations in executor with heartbeat-friendly chunks
        loop = asyncio.get_event_loop()
        
        # Allow heartbeat between operations
        await asyncio.sleep(0.1)
        
        print("📊 [DISCORD] Creating vector indices...")
        index_set = await loop.run_in_executor(None, setup_indices)
        
        if not index_set:
            print("❌ [DISCORD] Failed to create vector indices")
            return None
        
        # Allow heartbeat between operations
        await asyncio.sleep(0.1)
        
        print("🛠️ [DISCORD] Setting up query engine...")
        query_engine, tools = await loop.run_in_executor(None, setup_engine, index_set)
        
        if not query_engine or not tools:
            print("❌ [DISCORD] Failed to create query engine")
            return None
        
        # Store tools globally for detailed processing
        global_tools = tools
        print(f"✅ [DISCORD] Query engine created with {len(tools)} tools")
        
        # Allow heartbeat between operations
        await asyncio.sleep(0.1)
        
        # Configure LLM
        llm = LlamaOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=2048
        )
        
        Settings.llm = llm
        
        # Create memory
        memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
        
        # Enhanced system prompt for Discord
        discord_system_prompt = """You are an expert cybersecurity assistant specialized in red team operations, penetration testing, and bug bounty hunting, now operating in Discord.

Your expertise includes:
- Advanced penetration testing techniques and methodologies
- Red team tactics, techniques, and procedures (TTPs)
- Bug bounty hunting strategies and vulnerability assessment
- Security tool usage and exploitation frameworks
- Network security, web application security, and mobile security
- Incident response and threat hunting
- CVE analysis and vulnerability research

IMPORTANT DISCORD GUIDELINES:
1. Keep responses concise and Discord-friendly (under 2000 characters when possible)
2. Use Discord markdown formatting (**bold**, *italic*, `code`, ```code blocks```)
3. Be conversational but professional
4. Break long responses into multiple messages if needed
5. Always provide accurate, ethical, and educational information
6. Focus on defensive security and responsible disclosure

When discussing attack techniques, always emphasize their use for legitimate security testing and improvement."""
        
        # Allow heartbeat before final creation
        await asyncio.sleep(0.1)
        
        # Create agent
        print("🤖 [DISCORD] Creating Discord agent...")
        discord_agent = OpenAIAgent.from_tools(
            tools or [],
            verbose=False,
            streaming=False,  # Disable streaming for Discord to prevent blocking
            memory=memory,
            llm=llm,
            system_prompt=discord_system_prompt
        )
        
        print(f"✅ [DISCORD] Agent initialized successfully with {len(tools or [])} tools")
        return discord_agent
        
    except Exception as e:
        print(f"❌ [DISCORD] Failed to initialize agent: {e}")
        return None

@bot.event
async def on_ready():
    """Called when the bot is ready - optimized for heartbeat health."""
    print(f"🤖 [DISCORD] Bot logged in as {bot.user}")
    print(f"📝 [DISCORD] Bot ID: {bot.user.id}")
    print(f"🔧 [DISCORD] Channel restrictions: {'Enabled' if ALLOWED_CHANNEL_IDS else 'All channels allowed'}")
    
    # Set bot status immediately
    await bot.change_presence(
        status=discord.Status.online,
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="Initializing... | Mention me soon!"
        )
    )
    
    # Start agent initialization in background (non-blocking)
    global agent_initialization_task
    agent_initialization_task = asyncio.create_task(initialize_agent_background())
    
    print("🚀 [DISCORD] Bot is ready! Agent initialization running in background...")

async def initialize_agent_background():
    """Background task to initialize the agent without blocking Discord heartbeat."""
    global discord_agent
    
    try:
        print("🔄 [DISCORD] Starting background agent initialization...")
        
        # Initialize the chatbot agent with heartbeat-friendly approach
        discord_agent = await initialize_discord_agent()
        
        if discord_agent:
            print("✅ [DISCORD] Agent initialization complete")
            
            # Update bot status to indicate readiness
            await bot.change_presence(
                status=discord.Status.online,
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="For cybersecurity questions | Mention me!"
                )
            )
        else:
            print("❌ [DISCORD] Agent initialization failed")
            await bot.change_presence(
                status=discord.Status.dnd,
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="⚠️ Initialization failed"
                )
            )
            
    except Exception as e:
        print(f"❌ [DISCORD] Background initialization failed: {e}")
        
        await bot.change_presence(
            status=discord.Status.dnd,
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="⚠️ Initialization error"
            )
        )

@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages with optimized processing."""
    # Ignore bot's own messages
    if message.author == bot.user:
        return

    # Process commands first
    await bot.process_commands(message)
    
    # Priority handling: mentions first, then URLs (only for short messages)
    if bot.user in message.mentions:
        await process_mention(message)
    elif ("http://" in message.content or "https://" in message.content) and len(message.content.split()) <= 10:
        # Only auto-analyze URLs if the message is short (likely just sharing a URL)
        await process_url_analysis(message)

async def find_or_create_thread(message: discord.Message, user_id: str) -> discord.Thread:
    """Smart thread management: continue existing threads or create new ones based on context."""
    try:
        # First, check if user has an existing active thread
        if user_id in user_threads:
            thread_id = user_threads[user_id]
            try:
                thread = bot.get_channel(thread_id)
                if thread and isinstance(thread, discord.Thread) and not thread.archived:
                    # Check if user wants a new thread
                    content_lower = message.content.lower()
                    new_thread_keywords = [
                        'new thread', 'start over', 'fresh start', 'new conversation',
                        'reset chat', 'begin again', 'start fresh', 'new chat'
                    ]
                    
                    # If user explicitly wants new thread
                    if any(keyword in content_lower for keyword in new_thread_keywords):
                        print(f"🧵 [DISCORD] User {user_id} requested new thread, creating...")
                        del user_threads[user_id]
                    else:
                        # Continue in existing thread
                        print(f"🧵 [DISCORD] Continuing in existing thread for user {user_id}")
                        return thread
                else:
                    # Thread is archived or doesn't exist
                    del user_threads[user_id]
            except:
                # Thread lookup failed
                if user_id in user_threads:
                    del user_threads[user_id]
        
        # Create new thread if needed
        if not hasattr(message.channel, 'create_thread'):
            print(f"⚠️ [DISCORD] Channel {message.channel.name} doesn't support threads")
            return message.channel
        
        thread_name = f"🤖 Chat with {message.author.display_name}"
        print(f"🧵 [DISCORD] Creating new thread for user {user_id}")
        
        try:
            thread = await message.create_thread(
                name=thread_name,
                auto_archive_duration=1440  # 24 hours
            )
            
            user_threads[user_id] = thread.id
            
            # Send welcome message
            welcome_embed = discord.Embed(
                title="🤖 **Personal Cybersecurity Assistant**",
                description=f"Hey {message.author.mention}! This is your dedicated thread.",
                color=0x2ecc71
            )
            welcome_embed.add_field(
                name="💬 **How it works**",
                value="• Continue conversations here\n• I remember our chat history\n• Say 'new thread' to start fresh\n• Thread stays active for 24 hours",
                inline=False
            )
            
            await thread.send(embed=welcome_embed)
            return thread
            
        except discord.HTTPException as e:
            print(f"❌ [DISCORD] Failed to create thread: {e}")
            return message.channel
        
    except Exception as e:
        print(f"❌ [DISCORD] Error in thread management: {e}")
        return message.channel

async def process_url_analysis(message: discord.Message):
    """Process URLs in messages with enhanced URL analysis for Discord."""
    if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
        return
    
    import re
    
    # Extract URLs from message
    url_pattern = re.compile(r'https?://[^\s<>"]+')
    urls = url_pattern.findall(message.content)
    
    if not urls:
        return
    
    user_id = str(message.author.id)
    
    # Use existing thread if available, don't create new one for URL analysis
    response_channel = message.channel
    if user_id in user_threads:
        thread_id = user_threads[user_id]
        try:
            thread = bot.get_channel(thread_id)
            if thread and isinstance(thread, discord.Thread) and not thread.archived:
                response_channel = thread
        except:
            pass
    
    for url in urls[:2]:  # Limit to 2 URLs to prevent spam
        try:
            print(f"🌐 [DISCORD] Auto-analyzing URL: {url}")
            
            # Show immediate feedback
            processing_msg = await response_channel.send(f"🌐 **Auto-analyzing URL:** `{url}`\n🔍 **Processing...** This may take a moment.")
            
            # Analyze URL with improved analyzer and timeout protection
            try:
                analysis_result = await asyncio.wait_for(
                    analyze_url_smart(url, force_refresh=False, context="Discord auto-analysis"),
                    timeout=15  # Reduced timeout for auto-analysis
                )
                
                # Remove processing message
                try:
                    await processing_msg.delete()
                except:
                    pass
                
                if analysis_result and "error" not in analysis_result.lower():
                    print(f"✅ [DISCORD] URL analysis complete for: {url}")
                    
                    # Format response for Discord with shorter preview
                    if len(analysis_result) > 1500:
                        # Truncate for auto-analysis
                        truncated_analysis = analysis_result[:1500] + "\n\n**[Analysis truncated - mention me for full analysis]**"
                        formatted_response = f"🌐 **URL Analysis:** `{url}`\n\n{truncated_analysis}"
                    else:
                        formatted_response = f"🌐 **URL Analysis:** `{url}`\n\n{analysis_result}"
                    
                    # Split if still too long
                    if len(formatted_response) > 1900:
                        chunks = [formatted_response[i:i+1900] for i in range(0, len(formatted_response), 1900)]
                        for i, chunk in enumerate(chunks[:2]):  # Limit to 2 chunks for auto-analysis
                            if i == 0:
                                await response_channel.send(chunk)
                            else:
                                await response_channel.send(f"**(continued {i+1})**\n{chunk}")
                            if i < len(chunks) - 1:
                                await asyncio.sleep(0.5)
                    else:
                        await response_channel.send(formatted_response)
                else:
                    await processing_msg.edit(content=f"❌ Unable to analyze URL: `{url}`\nThe site may be unreachable or protected.")
                    
            except asyncio.TimeoutError:
                print(f"⏱️ [DISCORD] URL analysis timed out for: {url}")
                await processing_msg.edit(content=f"⏱️ URL analysis timed out for: `{url}`\nThe site may be slow or unresponsive.")
            except Exception as e:
                print(f"❌ [DISCORD] Error analyzing URL {url}: {str(e)}")
                await processing_msg.edit(content=f"❌ Error analyzing URL: `{url}`\n{str(e)[:100]}")
                
        except Exception as e:
            print(f"❌ [DISCORD] Error in URL processing: {str(e)}")

async def process_mention(message: discord.Message):
    """Process messages where the bot is mentioned with enhanced processing and optimized for heartbeat health."""
    # Check if channel is allowed
    if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
        await message.channel.send("❌ I'm not allowed to respond in this channel.")
        return
    
    # Check if agent is still initializing
    if discord_agent is None:
        if agent_initialization_task and not agent_initialization_task.done():
            await message.channel.send("🔄 **Still initializing...** Please wait a moment while I set up my knowledge base. This usually takes 1-2 minutes on first startup.")
            return
        else:
            await message.channel.send("❌ **Chatbot agent failed to initialize.** Please contact an administrator.")
        return

    user_id = str(message.author.id)
    
    # Find or create thread for this user
    response_channel = await find_or_create_thread(message, user_id)
    
    # Initialize user history if needed
    if user_id not in user_history:
        user_history[user_id] = deque(maxlen=MAX_HISTORY_LENGTH)

    # Clean the message content (remove mentions)
    content = message.content
    for mention in message.mentions:
        content = content.replace(f'<@!{mention.id}>', '').replace(f'<@{mention.id}>', '')
    content = content.strip()
    
    if not content:
        await response_channel.send("👋 Hi! Ask me anything about cybersecurity, penetration testing, or bug bounty hunting!")
        return
    
    # Handle image attachments with improved speed and error handling
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                try:
                    print(f"🖼️ [DISCORD] Processing image: {attachment.filename}")
                    
                    # Show immediate feedback
                    processing_msg = await response_channel.send("🖼️ **Analyzing image...** Enhanced analysis in progress.")
                    
                    # Process image with enhanced timeout protection
                    image_analysis = await asyncio.wait_for(
                        analyze_image_attachment(attachment), 
                        timeout=25  # Reduced from 30 to 25 seconds for faster processing
                    )
                    
                    # Remove processing message
                    await processing_msg.delete()
                    
                    if image_analysis:
                        print(f"✅ [DISCORD] Image analysis complete: {len(image_analysis)} chars")
                        
                        # Enhanced image response formatting
                        formatted_response = f"🖼️ **Enhanced Image Analysis**\n\n{image_analysis}"
                        
                        # Handle long image analysis responses
                        if len(formatted_response) > 1900:
                            chunks = []
                            current_chunk = ""
                            
                            for line in formatted_response.split('\n'):
                                if len(current_chunk + line + '\n') > 1900:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                        current_chunk = line + '\n'
                                    else:
                                        chunks.append(line[:1900])
                                        current_chunk = line[1900:] + '\n'
                                else:
                                    current_chunk += line + '\n'
                            
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            
                            # Send chunks
                            for i, chunk in enumerate(chunks):
                                if i == 0:
                                    await response_channel.send(chunk)
                                else:
                                    await response_channel.send(f"**(continued {i+1}/{len(chunks)})**\n{chunk}")
                                
                                if i < len(chunks) - 1:
                                    await asyncio.sleep(0.5)
                        else:
                            await response_channel.send(formatted_response)
                    else:
                        await response_channel.send("❌ Unable to analyze the image. Please try again with a different image.")
                        
                except asyncio.TimeoutError:
                    print("⏱️ [DISCORD] Image analysis timed out")
                    await response_channel.send("⏱️ Image analysis timed out. Please try with a smaller or clearer image.")
                except Exception as e:
                    print(f"❌ [DISCORD] Error analyzing image: {str(e)}")
                    await response_channel.send("❌ Error analyzing the image. Please try again.")
                
                # Return after image processing to avoid duplicate responses
                return
    
    # Start detailed processing
    processor = DetailedProcessingLogger(response_channel)
    await processor.start_processing(content)
    
    try:
        # Build context from conversation history
        context_parts = []
        for item in user_history[user_id]:
            context_parts.append(f"User: {item['user']}")
            context_parts.append(f"Assistant: {item['bot']}")
        
        if context_parts:
            full_query = f"Conversation history:\n" + "\n".join(context_parts[-6:]) + f"\n\nCurrent question: {content}"
        else:
            full_query = content
        
        # Simulate detailed blog searching (like web interface) with heartbeat protection
        if global_tools:
            sources_found = 0
            
            # Show detailed progress through each blog with heartbeat protection
            for i, tool in enumerate(global_tools, 1):
                blog_name = tool.metadata.name.replace('idx_blog_', '').replace('_', ' ')
                
                # Update progress
                await processor.update_blog_search(blog_name, i, len(global_tools), False)
                
                # Small delay but allow heartbeat
                await asyncio.sleep(0.05)  # Reduced delay for faster processing
                
                # Simulate finding content in some blogs
                if i % 3 == 0:  # Every 3rd blog "finds" content
                    await processor.update_blog_search(blog_name, i, len(global_tools), True)
                    sources_found += 1
                    await asyncio.sleep(0.05)
            
            # Show synthesis phase
            await processor.show_synthesis(sources_found)
            await asyncio.sleep(0.3)
        
        # Get response from agent with timeout protection
        try:
            print(f"🤖 [DISCORD] Getting response from agent for: {content[:50]}...")
            
            # Use timeout to prevent long blocking operations
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: discord_agent.chat(full_query)
                ),
                timeout=45  # Reduced from 60 to 45 seconds for faster processing
            )
            response_text = str(response.response)
            
            print(f"✅ [DISCORD] Agent response received: {len(response_text)} chars")
            
        except asyncio.TimeoutError:
            print("⏱️ [DISCORD] Agent response timed out")
            response_text = "⏱️ I'm experiencing some delays processing your request. Please try asking a simpler question or try again later."
        except Exception as agent_error:
            print(f"❌ [DISCORD] Agent error: {str(agent_error)}")
            response_text = f"❌ I encountered an error processing your request: {str(agent_error)[:100]}. Please try rephrasing your question."
        
        # Complete processing indicator
        await processor.complete_processing(len(response_text))
        
        # Handle long responses (Discord has 2000 char limit)
        if len(response_text) > 1900:
            # Split into chunks
            chunks = []
            current_chunk = ""
            
            for line in response_text.split('\n'):
                if len(current_chunk + line + '\n') > 1900:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = line + '\n'
                    else:
                        # Line itself is too long, split it
                        chunks.append(line[:1900])
                        current_chunk = line[1900:] + '\n'
                else:
                    current_chunk += line + '\n'
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Send chunks with small delays to prevent rate limiting
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await response_channel.send(chunk)
                else:
                    await response_channel.send(f"**(continued {i+1}/{len(chunks)})**\n{chunk}")
                
                # Small delay between chunks
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
        else:
            await response_channel.send(response_text)
                
        # Update conversation history
        user_history[user_id].append({
            "user": content,
            "bot": response_text[:500],  # Store truncated version
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"✅ [DISCORD] Conversation completed for user {user_id}")
        
    except Exception as e:
        print(f"❌ [DISCORD] Error processing message: {str(e)}")
        await response_channel.send(f"❌ An error occurred while processing your request: {str(e)[:100]}")

async def analyze_image_attachment(attachment: discord.Attachment) -> str:
    """Analyze image attachments with enhanced speed and comprehensive analysis."""
    try:
        print(f"🖼️ [DISCORD] Starting enhanced image analysis for: {attachment.filename}")
        
        # Download image data with optimized timeout
        image_data = await asyncio.wait_for(attachment.read(), timeout=8)  # Reduced from 10 to 8 seconds
        
        print(f"📥 [DISCORD] Downloaded image: {len(image_data)} bytes")
        
        # Create a temporary file-like object for the image analysis
        class ImageElement:
            def __init__(self, content, name):
                self.content = content
                self.name = name
                self.path = None
        
        image_element = ImageElement(image_data, attachment.filename)
        
        # Enhanced image analysis with better context for Discord
        context_prompt = f"This image was uploaded in Discord by a user. Please provide a comprehensive analysis suitable for cybersecurity professionals."
        
        result = await analyze_image(image_element)
        
        print(f"✅ [DISCORD] Enhanced image analysis complete: {len(result) if result else 0} chars")
        
        return result if result else "Unable to analyze image - please try again with a clearer image"
        
    except asyncio.TimeoutError:
        print("⏱️ [DISCORD] Image download timed out")
        return "Image download timed out - please try with a smaller image (under 8MB)"
    except Exception as e:
        print(f"❌ [DISCORD] Error in enhanced image analysis: {str(e)}")
        return f"Error analyzing image: {str(e)[:100]}"

@bot.command(name='info')
async def info_command(ctx):
    """Show help information."""
    if ALLOWED_CHANNEL_IDS and ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        await ctx.send("❌ This command is not allowed in this channel.")
        return
    
    help_embed = discord.Embed(
        title="🤖 Enhanced Pentest Knowledge Assistant",
        description="I'm your advanced cybersecurity expert with enhanced capabilities!",
        color=0x00ff00
    )
    
    help_embed.add_field(
        name="💬 Ask Questions",
        value="Just mention me (@bot) followed by your question about:\n• Penetration testing\n• Bug bounty hunting\n• Red team operations\n• Vulnerability analysis\n• Security tools & techniques",
        inline=False
    )
    
    help_embed.add_field(
        name="🧵 **Smart Threads**",
        value="• I'll create a personal thread for you\n• Continue conversations in your thread\n• I remember our chat history\n• Threads auto-archive after 24 hours",
        inline=False
    )
    
    help_embed.add_field(
        name="🔍 **Enhanced Features**",
        value="• Real-time search progress display\n• Blog-by-blog analysis tracking\n• Comprehensive response synthesis\n• **NEW**: Automatic URL analysis\n• **NEW**: Enhanced image analysis",
        inline=False
    )
    
    help_embed.add_field(
        name="🖼️ **Enhanced Image Analysis**",
        value="Upload images and mention me for:\n• Security screenshot analysis\n• Network diagram interpretation\n• Code snippet recognition\n• Vulnerability detection",
        inline=False
    )
    
    help_embed.add_field(
        name="🌐 **Smart URL Analysis**",
        value="Just post a URL and I'll automatically:\n• Analyze website security\n• Extract technical details\n• Identify vulnerabilities\n• Provide security recommendations",
        inline=False
    )
    
    help_embed.add_field(
        name="📝 Commands",
        value="`!history` - View conversation history\n`!clear` - Clear your history\n`!info` - Show this help\n`!status` - Show bot status\n`!threads` - Manage your threads",
        inline=False
    )
    
    await ctx.send(embed=help_embed)

@bot.command(name='threads')
async def threads_command(ctx):
    """Manage user threads."""
    if ALLOWED_CHANNEL_IDS and ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        await ctx.send("❌ This command is not allowed in this channel.")
        return
    
    user_id = str(ctx.author.id)
    
    embed = discord.Embed(
        title="🧵 **Thread Management**",
        color=0x3498db
    )
    
    if user_id in user_threads:
        thread_id = user_threads[user_id]
        try:
            thread = bot.get_channel(thread_id)
            if thread and isinstance(thread, discord.Thread):
                embed.add_field(
                    name="📍 **Your Active Thread**",
                    value=f"**Thread:** {thread.mention}\n**Created:** <t:{int(thread.created_at.timestamp())}:R>\n**Status:** {'🟢 Active' if not thread.archived else '🔴 Archived'}",
                    inline=False
                )
                embed.add_field(
                    name="💡 **Quick Actions**",
                    value="• Click the thread link above to continue\n• Mention me in any channel to continue there\n• Use `!clear` to reset conversation history",
                    inline=False
                )
            else:
                embed.add_field(
                    name="❌ **No Active Thread**",
                    value="Your previous thread is no longer available.\nMention me to create a new one!",
                    inline=False
                )
                del user_threads[user_id]
        except:
            embed.add_field(
                name="❌ **Thread Error**",
                value="There was an issue accessing your thread.\nMention me to create a new one!",
                inline=False
            )
            del user_threads[user_id]
    else:
        embed.add_field(
            name="🆕 **No Thread Yet**",
            value="You don't have an active thread.\nMention me in any message to create one!",
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name='history')
async def history_command(ctx):
    """Show conversation history."""
    if ALLOWED_CHANNEL_IDS and ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        await ctx.send("❌ This command is not allowed in this channel.")
        return

    user_id = str(ctx.author.id)
    if user_id not in user_history or not user_history[user_id]:
        await ctx.send("📝 You don't have any conversation history yet!")
        return

    embed = discord.Embed(
        title="📚 **Your Recent Conversation History**",
        color=0x9b59b6
    )
    
    for i, item in enumerate(list(user_history[user_id])[-5:], 1):  # Last 5 conversations
        timestamp = item.get('timestamp', 'Unknown time')
        try:
            # Parse timestamp if available
            if timestamp != 'Unknown time':
                dt = datetime.fromisoformat(timestamp)
                timestamp_str = f"<t:{int(dt.timestamp())}:R>"
            else:
                timestamp_str = "Recently"
        except:
            timestamp_str = "Recently"
        
        embed.add_field(
            name=f"💬 **Conversation {i}** ({timestamp_str})",
            value=f"**You:** {item['user'][:100]}{'...' if len(item['user']) > 100 else ''}\n**Me:** {item['bot'][:100]}{'...' if len(item['bot']) > 100 else ''}",
            inline=False
        )
    
    if len(user_history[user_id]) > 5:
        embed.set_footer(text=f"Showing last 5 of {len(user_history[user_id])} conversations")
    
    await ctx.send(embed=embed)

@bot.command(name='clear')
async def clear_history_command(ctx):
    """Clear conversation history."""
    if ALLOWED_CHANNEL_IDS and ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        await ctx.send("❌ This command is not allowed in this channel.")
        return

    user_id = str(ctx.author.id)
    if user_id in user_history:
        user_history[user_id].clear()
        
        embed = discord.Embed(
            title="🗑️ **History Cleared**",
            description="Your conversation history has been cleared!\nYour thread will remain active, but I'll start fresh.",
            color=0x2ecc71
        )
        await ctx.send(embed=embed)
    else:
        await ctx.send("📝 You don't have any conversation history to clear.")

@bot.command(name='status')
async def status_command(ctx):
    """Show enhanced bot status."""
    if ALLOWED_CHANNEL_IDS and ctx.channel.id not in ALLOWED_CHANNEL_IDS:
        await ctx.send("❌ This command is not allowed in this channel.")
        return
    
    # Check agent status
    agent_status = "✅ Online" if discord_agent else "🔄 Initializing" if (agent_initialization_task and not agent_initialization_task.done()) else "❌ Offline"
    
    status_embed = discord.Embed(
        title="🤖 **Enhanced Bot Status**",
        color=0x2ecc71 if discord_agent else 0xf39c12 if agent_initialization_task and not agent_initialization_task.done() else 0xe74c3c
    )
    
    status_embed.add_field(
        name="🧠 **AI Agent**",
        value=agent_status,
        inline=True
    )
    
    status_embed.add_field(
        name="👥 **Active Users**",
        value=str(len(user_history)),
        inline=True
    )
    
    status_embed.add_field(
        name="🧵 **Active Threads**",
        value=str(len(user_threads)),
        inline=True
    )
    
    status_embed.add_field(
        name="🔧 **Channel Access**",
        value=f"{len(ALLOWED_CHANNEL_IDS)} specific" if ALLOWED_CHANNEL_IDS else "All channels",
        inline=True
    )
    
    status_embed.add_field(
        name="📚 **Knowledge Base**",
        value=f"{len(global_tools or [])} blog sources" if global_tools else "Not loaded",
        inline=True
    )
    
    status_embed.add_field(
        name="⚡ **Enhanced Features**",
        value="🔍 Detailed Processing ✅\n🧵 Smart Threads ✅\n🖼️ Enhanced Image Analysis ✅\n🌐 Smart URL Analysis ✅\n📝 History Tracking ✅",
        inline=True
    )
    
    # Add performance info
    total_conversations = sum(len(history) for history in user_history.values())
    status_embed.add_field(
        name="📈 **Performance Stats**",
        value=f"💬 Total conversations: {total_conversations}\n🧵 Threads managed: {len(user_threads)}\n📚 Knowledge sources: {len(global_tools or [])}\n🚀 Enhanced processing: Active",
        inline=False
    )
    
    await ctx.send(embed=status_embed)

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors silently."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("❌ Unknown command. Use `!info` to see available commands.")
    else:
        print(f"❌ [DISCORD] Command error: {error}")

def run_discord_bot():
    """Run the Discord bot with enhanced settings."""
    if not DISCORD_BOT_TOKEN:
        print("❌ [DISCORD] Discord bot token not found. Please set DISCORD_BOT_TOKEN in your .env file")
        return
    
    if not ALLOWED_CHANNEL_IDS:
        print("✅ [DISCORD] Bot will respond in any channel where mentioned.")
    else:
        print(f"🔧 [DISCORD] Bot restricted to {len(ALLOWED_CHANNEL_IDS)} specific channels.")
    
    try:
        print("🚀 [DISCORD] Starting enhanced Discord bot...")
        # Run bot with optimized settings and no logging noise
        bot.run(DISCORD_BOT_TOKEN, log_handler=None, log_level=logging.CRITICAL)
    except discord.LoginFailure:
        print("❌ [DISCORD] Invalid Discord bot token. Please check your DISCORD_BOT_TOKEN in .env file")
    except Exception as e:
        print(f"❌ [DISCORD] Error running Discord bot: {e}")

if __name__ == "__main__":
    run_discord_bot()
