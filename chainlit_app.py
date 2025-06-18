import chainlit as cl
from chatbot import setup_vector_indices, setup_query_engine
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
import logging
import glob
import os
import threading
from dotenv import load_dotenv
from chainlit.types import ThreadDict
from chainlit.input_widget import Select, Switch, Slider
import re
import asyncio
from urllib.parse import urlparse
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import colorama
from colorama import Fore, Back, Style
from pathlib import Path

# Import modules
from modules.index_manager import update_indices_if_needed
from modules.auth import oauth_callback
from modules.image_analysis import analyze_image
from modules.image_generation import (
    generate_image, generate_cybersecurity_diagram, 
    generate_security_themed_image, generate_image_advanced
)
from modules.file_handler import handle_file_upload, handle_url
from modules.media_handler import handle_media_upload
from modules.discord_bot import run_discord_bot
from modules.batch_processor import batch_processor, BatchProcessor
from modules.url_analyzer import analyze_url_smart, get_cache_stats
from modules.security_performance import (
    SecurityScanner, ContextAnalyzer, get_performance_report,
    monitor_performance, security_scan, rate_limit,
    enhance_response_with_context
)
from modules.query_engine import (
    display_query_statistics, show_available_blog_functions,
    log_function_execution, terminal_logger
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store indices and query engine
global_index_set = None
global_query_engine = None
global_tools = None

# Cache and performance settings (optimized for better performance)
URL_CACHE_SIZE = 100  # Reduced from 200 for better memory usage
KNOWLEDGE_BASE_CACHE_SIZE = 500  # Reduced from 1000 for better memory usage
MAX_CONVERSATION_HISTORY = 20  # Reduced from 50 for better memory usage
MAX_CONTENT_LENGTH = 3 * 1024 * 1024  # 3MB limit for content processing

# URL Cache system
URL_CACHE = {}
CACHE_DURATION = timedelta(hours=6)  # Cache URLs for 6 hours

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# Default settings optimized for GPT-4o with better performance
DEFAULT_SETTINGS = {
    "Model": "gpt-4o",
    "Streaming": True,
    "Temperature": 0.7,
    "Max_Tokens": 2048,  # Optimized for faster responses while maintaining quality
    "Top_P": 0.9,
    "SAI_Steps": 30,
    "SAI_Cfg_Scale": 7,
    "SAI_Width": 512,
    "SAI_Height": 512,
    "Image_Size": 1024,
    "Image_Quality": "standard"  # Changed from "hd" for faster generation
}

# Initialize global components
batch_processor = BatchProcessor()
security_scanner = SecurityScanner()
context_analyzer = ContextAnalyzer()

# Store conversation history for context analysis
conversation_history = []

# Enhanced conversation storage with persistent memory 
CONVERSATION_STORAGE = {}

def add_to_conversation_storage(user_id: str, role: str, content: str):
    """Add conversation entry to persistent storage."""
    if user_id not in CONVERSATION_STORAGE:
        CONVERSATION_STORAGE[user_id] = []
    
    CONVERSATION_STORAGE[user_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 20 entries for memory management
    if len(CONVERSATION_STORAGE[user_id]) > 20:
        CONVERSATION_STORAGE[user_id] = CONVERSATION_STORAGE[user_id][-20:]

def get_conversation_context(user_id: str = "default", last_n: int = 5) -> str:
    """Get conversation context for the current user."""
    if user_id not in CONVERSATION_STORAGE:
        return ""
    
    history = CONVERSATION_STORAGE[user_id][-last_n:]  # Get last N conversations
    context_parts = []
    
    for i, entry in enumerate(history):
        if entry.get("role") == "user":
            context_parts.append(f"Previous question ({i+1}): {entry['content'][:200]}")
        elif entry.get("role") == "assistant":
            context_parts.append(f"Previous answer ({i+1}): {entry['content'][:200]}")
    
    return "\n".join(context_parts)

def enhance_query_with_context(query: str, user_id: str = "default") -> str:
    """Enhance user query with conversation context."""
    context = get_conversation_context(user_id)
    
    if context:
        enhanced_query = f"""Previous conversation context:
{context}

Current question: {query}

Please consider the previous context when answering the current question. If the user is referring to something from our previous conversation, please reference it appropriately."""
        return enhanced_query
    
    return query

# Terminal formatting utilities
class TerminalFormatter:
    """Enhanced terminal output formatting for better readability."""
    
    def __init__(self):
        colorama.init(autoreset=True)
        self.width = 80
        
    def header(self, text: str, char: str = "=") -> str:
        """Create a formatted header."""
        padding = (self.width - len(text) - 2) // 2
        return f"\n{Fore.CYAN}{char * padding} {text} {char * padding}{Style.RESET_ALL}"
    
    def subheader(self, text: str) -> str:
        """Create a formatted subheader."""
        return f"\n{Fore.YELLOW}{'─' * 20} {text} {'─' * (self.width - len(text) - 22)}{Style.RESET_ALL}"
    
    def success(self, text: str) -> str:
        """Format success message."""
        return f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}"
    
    def error(self, text: str) -> str:
        """Format error message."""
        return f"{Fore.RED}❌ {text}{Style.RESET_ALL}"
    
    def warning(self, text: str) -> str:
        """Format warning message."""
        return f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}"
    
    def info(self, text: str) -> str:
        """Format info message."""
        return f"{Fore.BLUE}ℹ️  {text}{Style.RESET_ALL}"
    
    def processing(self, text: str) -> str:
        """Format processing message."""
        return f"{Fore.MAGENTA}⚙️  {text}{Style.RESET_ALL}"
    
    def step(self, step_num: int, total_steps: int, text: str) -> str:
        """Format step message."""
        return f"{Fore.CYAN}[{step_num}/{total_steps}] {text}{Style.RESET_ALL}"
    
    def metric(self, label: str, value: str, unit: str = "") -> str:
        """Format metric display."""
        return f"{Fore.WHITE}{label:<20} {Fore.GREEN}{value} {unit}{Style.RESET_ALL}"
    
    def separator(self, char: str = "─") -> str:
        """Create a separator line."""
        return f"{Fore.CYAN}{char * self.width}{Style.RESET_ALL}"

# Global formatter instance
terminal = TerminalFormatter()

def add_to_conversation_history(user_message: str, assistant_response: str):
    """Add messages to conversation history for context analysis."""
    conversation_history.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_message,
        "assistant": assistant_response
    })
    
    # Keep only last 10 exchanges
    if len(conversation_history) > 10:
        conversation_history.pop(0)

def get_url_cache_key(url: str) -> str:
    """Generate a cache key for a URL."""
    return hashlib.md5(url.encode()).hexdigest()

def is_url_cached(url: str) -> bool:
    """Check if URL is cached and not expired."""
    cache_key = get_url_cache_key(url)
    if cache_key in URL_CACHE:
        cached_time = URL_CACHE[cache_key]['timestamp']
        if datetime.now() - cached_time < CACHE_DURATION:
            return True
        else:
            # Remove expired cache
            del URL_CACHE[cache_key]
    return False

def get_cached_url_content(url: str) -> str:
    """Get cached URL content."""
    cache_key = get_url_cache_key(url)
    if cache_key in URL_CACHE:
        return URL_CACHE[cache_key]['content']
    return ""

def cache_url_content(url: str, content: str):
    """Cache URL content."""
    cache_key = get_url_cache_key(url)
    URL_CACHE[cache_key] = {
        'content': content,
        'timestamp': datetime.now(),
        'url': url
    }
    
    # Keep cache size reasonable (max 100 URLs)
    if len(URL_CACHE) > 100:
        # Remove oldest entries
        oldest_key = min(URL_CACHE.keys(), key=lambda k: URL_CACHE[k]['timestamp'])
        del URL_CACHE[oldest_key]

async def check_existing_knowledge(urls: list, agent) -> str:
    """Check if we have existing knowledge about the URLs in our blog storage."""
    if not urls or not agent:
        return ""
    
    try:
        # Extract domain names and potential topics from URLs
        search_terms = []
        for url_info in urls:
            # Extract the actual URL string from the dictionary
            if isinstance(url_info, dict) and 'url' in url_info:
                url = url_info['url']
            else:
                # For backward compatibility, handle if url_info is already a string
                url = url_info
                
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            path = parsed_url.path.lower()
            
            # Add domain name variations
            search_terms.append(domain)
            if domain.startswith('www.'):
                search_terms.append(domain[4:])  # Remove www.
            
            # Extract potential keywords from path
            path_parts = [part for part in path.split('/') if part and len(part) > 2]
            search_terms.extend(path_parts[:2])  # Only first 2 path parts
        
        # Remove duplicates
        search_terms = list(set(search_terms))
        
        if not search_terms:
            return ""
        
        # Search our existing knowledge base
        search_query = f"Find information about: {' OR '.join(search_terms[:3])}"  # Limit to 3 terms
        
        logger.info(f"Searching existing knowledge for: {search_terms[:3]}")
        
        # Try to get response from agent's knowledge base
        response = await cl.make_async(agent.chat)(search_query)
        
        if response and response.response:
            # Check if the response contains meaningful information
            response_text = response.response.strip()
            generic_responses = [
                "i don't have", "i can't find", "no information", 
                "not available", "unable to", "i'm not sure",
                "i don't know", "not found", "no specific",
                "i couldn't find", "no relevant", "doesn't contain"
            ]
            
            # Only return knowledge base info if it's substantial and relevant
            if len(response_text) > 150 and not any(generic in response_text.lower() for generic in generic_responses):
                return f"Based on our existing knowledge base:\n{response_text}"
        
        return ""
        
    except Exception as e:
        logger.error(f"Error checking existing knowledge: {str(e)}")
        return ""

def is_command_output(content: str) -> bool:
    """Detect if content appears to be command output that might contain incidental URLs."""
    command_indicators = [
        "└─$", "nmap scan", "Starting Nmap", "PORT", "STATE", "SERVICE", 
        "Host is up", "Nmap done:", "root@", "user@", "bash-", "sh-",
        "HTTP/1.1", "GET /", "POST /", "curl", "wget", "ping",
        "traceroute", "dig", "nslookup"
    ]
    return any(indicator in content for indicator in command_indicators)

def should_auto_process_urls(content: str, urls_found: list) -> bool:
    """Determine if URLs should be automatically processed based on context."""
    # More aggressive detection for URL analysis requests
    explicit_requests = [
        "analyze url", "process url", "check url", "crawl", "website analysis",
        "what's on this site", "extract from url", "scan website", "url content",
        "analyze this link", "check this link", "what's at", "visit this",
        "brief about this", "explain me", "tell me about", "about this blog",
        "from this blog", "this website", "this site", "this page", "content of",
        "information from", "details from", "summary of", "overview of"
    ]
    
    # Check if user is explicitly asking about URLs
    if any(phrase in content.lower() for phrase in explicit_requests):
        return True
    
    # If user mentions "blog", "website", "site", "page" near a URL, process it
    url_context_words = ["blog", "website", "site", "page", "article", "post", "link"]
    if any(word in content.lower() for word in url_context_words) and urls_found:
        return True
    
    # Standalone URL (likely intentional)
    if len(urls_found) == 1 and len(content.strip().split()) <= 5:
        return True
    
    # If user is asking questions about content and URLs are present
    question_words = ["what", "how", "why", "explain", "tell", "describe", "show"]
    if any(word in content.lower() for word in question_words) and urls_found:
        return True
    
    # Don't auto-process URLs in command output
    if is_command_output(content):
        return False
    
    # Don't auto-process if there are many URLs (likely copy-paste content)
    if len(urls_found) > 3:
        return False
    
    return False

def is_likely_internal_url(url: str) -> bool:
    """Check if URL is likely internal/private and shouldn't be crawled."""
    internal_indicators = [
        '.htb', '.local', '.internal', '.corp', '.lab', '.test', '.dev',
        'localhost', '127.0.0.1', '10.', '192.168.', '172.16.', '172.17.',
        '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.',
        '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.',
        '172.30.', '172.31.', 'DC01.', 'server.', 'internal.', 'vpn.',
        'intranet.', 'admin.', ':8080', ':3000', ':8000', ':9000', ':5000'
    ]
    return any(indicator in url.lower() for indicator in internal_indicators)

def is_detailed_explanation_request(content: str) -> bool:
    """Enhanced function to detect when user wants detailed explanations."""
    content_lower = content.lower()
    
    # Detailed explanation patterns
    detailed_patterns = [
        # Direct requests for more detail
        r'explain\s+(?:more|in\s+detail|thoroughly|comprehensively)',
        r'tell\s+me\s+more\s+about',
        r'give\s+me\s+(?:more\s+)?(?:details|information)',
        r'provide\s+(?:more\s+)?(?:details|information|analysis)',
        r'elaborate\s+on',
        r'expand\s+on',
        r'go\s+(?:into\s+)?(?:more\s+)?detail',
        
        # Educational requests
        r'how\s+does\s+(?:this\s+)?(?:work|function|operate)',
        r'what\s+exactly\s+(?:is|does|happens)',
        r'step\s+by\s+step',
        r'walk\s+me\s+through',
        r'break\s+(?:this\s+)?down',
        r'in\s+(?:simple\s+)?terms',
        
        # Context-specific requests
        r'(?:in\s+)?(?:the\s+)?context\s+of',
        r'depending\s+on',
        r'based\s+on\s+(?:my|the|user)',
        r'for\s+(?:my\s+)?(?:specific\s+)?(?:use\s+case|situation|needs)',
        r'considering\s+(?:my|the)',
        
        # Question words indicating need for explanation
        r'why\s+(?:is\s+)?(?:this\s+)?(?:important|significant|relevant)',
        r'how\s+(?:can\s+)?(?:this\s+)?(?:help|be\s+used|apply)',
        r'what\s+(?:are\s+the\s+)?(?:implications|consequences|effects)',
        r'what\s+should\s+(?:i|we)\s+(?:know|do|consider)',
        
        # Improvement/learning requests
        r'improve\s+(?:more|further)',
        r'better\s+understanding',
        r'learn\s+more\s+about',
        r'understand\s+(?:better|fully|completely)',
        r'comprehensive\s+(?:analysis|explanation|guide)',
        
        # Request for examples and practical info
        r'(?:give\s+me\s+)?examples?\s+of',
        r'show\s+me\s+how',
        r'practical\s+(?:application|use|implementation)',
        r'real[-\s]world\s+(?:example|scenario|application)',
    ]
    
    # Check for any detailed explanation pattern
    for pattern in detailed_patterns:
        if re.search(pattern, content_lower):
            return True
    
    # Check for follow-up indicators
    follow_up_indicators = [
        'also', 'additionally', 'furthermore', 'moreover', 'besides',
        'what else', 'anything else', 'other', 'more specifically'
    ]
    
    if any(indicator in content_lower for indicator in follow_up_indicators):
        return True
    
    return False

def extract_user_context_clues(content: str) -> Dict[str, Any]:
    """Extract context clues from user message to understand their needs better."""
    content_lower = content.lower()
    
    context = {
        'skill_level': 'intermediate',  # Default
        'specific_interest': None,
        'urgency': 'normal',
        'purpose': 'learning',
        'domain_focus': None
    }
    
    # Skill level indicators
    beginner_indicators = ['new to', 'beginner', 'just started', 'basic', 'simple', 'easy']
    advanced_indicators = ['advanced', 'expert', 'professional', 'complex', 'deep dive', 'technical']
    
    if any(indicator in content_lower for indicator in beginner_indicators):
        context['skill_level'] = 'beginner'
    elif any(indicator in content_lower for indicator in advanced_indicators):
        context['skill_level'] = 'advanced'
    
    # Purpose indicators
    if any(word in content_lower for word in ['pentest', 'penetration testing', 'red team']):
        context['purpose'] = 'penetration_testing'
        context['domain_focus'] = 'offensive_security'
    elif any(word in content_lower for word in ['bug bounty', 'bounty hunting', 'vulnerability research']):
        context['purpose'] = 'bug_bounty'
        context['domain_focus'] = 'vulnerability_research'
    elif any(word in content_lower for word in ['study', 'learn', 'understand', 'exam', 'certification']):
        context['purpose'] = 'learning'
    elif any(word in content_lower for word in ['work', 'job', 'project', 'implement']):
        context['purpose'] = 'professional'
    
    # Urgency indicators
    urgent_indicators = ['urgent', 'asap', 'quickly', 'fast', 'immediate', 'right now']
    if any(indicator in content_lower for indicator in urgent_indicators):
        context['urgency'] = 'high'
    
    return context

async def enhance_response_with_context(base_response: str, user_context: Dict[str, Any], original_query: str) -> str:
    """Enhance the base response with contextual information based on user needs."""
    
    # Check if response needs enhancement
    if len(base_response) > 1000 and 'comprehensive' in base_response.lower():
        return base_response  # Already comprehensive
    
    enhanced_sections = []
    
    # Add skill-level appropriate explanations
    if user_context.get('skill_level') == 'beginner':
        enhanced_sections.append("""
## 🎓 **Beginner-Friendly Explanation**

Let me break this down in simple terms:
- **What this means**: This is fundamental cybersecurity knowledge that forms the foundation for more advanced concepts
- **Why it matters**: Understanding these basics will help you progress in cybersecurity
- **Next steps**: Consider exploring related concepts to build your knowledge systematically
""")
    elif user_context.get('skill_level') == 'advanced':
        enhanced_sections.append("""
## 🔬 **Advanced Technical Details**

For advanced practitioners:
- **Technical implementation**: Consider the underlying mechanisms and implementation details
- **Advanced applications**: Look for ways to apply this in complex scenarios
- **Research opportunities**: This knowledge can be extended for cutting-edge research
""")
    
    # Add purpose-specific guidance
    if user_context.get('purpose') == 'penetration_testing':
        enhanced_sections.append("""
## 🎯 **Penetration Testing Perspective**

**For Red Team Operations:**
- **Attack vectors**: Consider how this knowledge applies to penetration testing scenarios
- **Tools and techniques**: Relevant tools and methodologies for practical implementation
- **Defensive considerations**: Understanding defensive measures helps improve attack strategies
- **Reporting**: How to document and present findings professionally
""")
    elif user_context.get('purpose') == 'bug_bounty':
        enhanced_sections.append("""
## 💰 **Bug Bounty Applications**

**For Bug Bounty Hunters:**
- **Target identification**: How to identify potential targets using this knowledge
- **Methodology**: Systematic approaches for vulnerability discovery
- **Scope considerations**: Understanding program scope and restrictions
- **Impact assessment**: Evaluating the severity and business impact of findings
""")
    
    # Add domain-specific insights
    if user_context.get('domain_focus') == 'offensive_security':
        enhanced_sections.append("""
## ⚔️ **Offensive Security Insights**

**Offensive Security Applications:**
- **Attack methodologies**: How this fits into broader attack chains
- **Evasion techniques**: Methods to bypass common defensive measures
- **Post-exploitation**: Leveraging this knowledge after initial compromise
- **Ethical considerations**: Responsible disclosure and legal boundaries
""")
    
    # Check if user wants more practical examples
    if is_detailed_explanation_request(original_query):
        enhanced_sections.append("""
## 📋 **Practical Examples & Applications**

**Real-World Scenarios:**
- **Case studies**: Examples from actual security incidents or research
- **Step-by-step walkthrough**: Detailed implementation guidance
- **Common pitfalls**: Mistakes to avoid and lessons learned
- **Best practices**: Industry-standard approaches and recommendations

**Further Learning:**
- **Related concepts**: Connected topics that build upon this knowledge
- **Advanced topics**: Next-level concepts to explore
- **Resources**: Books, papers, tools, and training materials for deeper study
""")
    
    # Combine base response with enhancements
    if enhanced_sections:
        enhanced_response = base_response + "\n\n" + "\n".join(enhanced_sections)
        
        # Add a personalized conclusion
        enhanced_response += f"""

## 🎯 **Key Takeaways for Your Context**

Based on your {user_context.get('skill_level', 'intermediate')} level and focus on {user_context.get('purpose', 'learning')}, the most important points are:

1. **Immediate action**: Start with the fundamentals and build systematically
2. **Practical application**: Look for opportunities to apply this knowledge in controlled environments
3. **Continuous learning**: Stay updated with latest developments and best practices
4. **Community engagement**: Connect with other professionals in the cybersecurity community

💡 **Need more specific guidance?** Feel free to ask follow-up questions about any aspect that interests you most!
"""
        return enhanced_response
    
    return base_response

async def initialize_agent(blog_files, settings=None):
    """Initialize the OpenAI agent with proper error handling and GPT-4o configuration."""
    global global_index_set, global_query_engine, global_tools
    
    if settings is None:
        settings = DEFAULT_SETTINGS.copy()
    
    try:
        print(terminal.subheader("AGENT INITIALIZATION"))
        
        if global_index_set is None:
            print(terminal.step(1, 4, "Setting up vector indices"))
            await update_indices_if_needed(blog_files)
            global_index_set = await cl.make_async(setup_vector_indices)(blog_files)
        
        if global_index_set:
            print(terminal.step(2, 4, "Setting up query engine"))
            global_query_engine, global_tools = await cl.make_async(setup_query_engine)(global_index_set, blog_files)
        else:
            print(terminal.warning("No indices were created. Initializing agent without query engine."))
            global_query_engine, global_tools = None, []

        # Create memory with appropriate token limit for GPT-4o
        print(terminal.step(3, 4, "Configuring memory and LLM"))
        memory = ChatMemoryBuffer.from_defaults(token_limit=16000)
        
        # Configure LLM with explicit settings
        llm = LlamaOpenAI(
            model=settings.get("Model", "gpt-4o"),
            temperature=settings.get("Temperature", 0.7),
            max_tokens=settings.get("Max_Tokens", 4096),
            top_p=settings.get("Top_P", 0.9),
            streaming=settings.get("Streaming", True),
            api_key=openai_api_key
        )
        
        # Set global LLM for consistency
        Settings.llm = llm
        
        # Enhanced system prompt for better contextual responses
        enhanced_system_prompt = """You are an expert cybersecurity assistant specialized in red team operations, penetration testing, and bug bounty hunting. 

Your expertise includes:
- Advanced penetration testing techniques and methodologies
- Red team tactics, techniques, and procedures (TTPs)
- Bug bounty hunting strategies and vulnerability assessment
- Security tool usage and exploitation frameworks
- Network security, web application security, and mobile security
- Incident response and threat hunting
- CVE analysis and vulnerability research
- Security blog analysis and educational content

IMPORTANT RESPONSE GUIDELINES:
1. **Always provide comprehensive, educational explanations** - Don't just give basic information
2. **Include practical context** - Explain how information applies to real-world scenarios
3. **Provide step-by-step guidance** when appropriate
4. **Include examples and case studies** to illustrate concepts
5. **Consider the user's skill level** and adjust explanations accordingly
6. **Connect related concepts** to help users build comprehensive understanding
7. **Include actionable recommendations** and next steps
8. **Explain security implications** and real-world impact
9. **Provide multiple perspectives** (offensive, defensive, research)
10. **Always emphasize ethical considerations** and responsible disclosure

When analyzing URLs or security research:
- Provide detailed technical analysis
- Explain vulnerability mechanisms thoroughly
- Include impact assessment and risk analysis
- Discuss mitigation strategies and defensive measures
- Connect to broader security concepts and frameworks
- Provide educational context for learning

When users ask for explanations about cybersecurity topics:
- Start with clear definitions and fundamentals
- Build up to more complex concepts systematically
- Include practical examples and real-world applications
- Provide context about why the topic is important
- Suggest related learning resources and next steps
- Address common misconceptions or pitfalls

Always provide accurate, ethical, and educational information. Focus on defensive security and responsible disclosure. When discussing attack techniques, always emphasize their use for legitimate security testing and improvement."""
        
        # Create agent with enhanced configuration
        print(terminal.step(4, 4, "Creating OpenAI agent"))
        agent = OpenAIAgent.from_tools(
            global_tools or [],
            verbose=True,
            streaming=settings.get("Streaming", True),
            memory=memory,
            llm=llm,
            system_prompt=enhanced_system_prompt
        )
        
        print(terminal.metric("Model", settings.get("Model", "gpt-4o")))
        print(terminal.metric("Tools Available", str(len(global_tools or []))))
        print(terminal.metric("Memory Limit", "16000 tokens"))
        print(terminal.success("Agent initialized successfully"))
        
        # Show available blog functions if we have tools
        if global_tools and global_query_engine:
            show_available_blog_functions(global_tools)
            
        return agent
        
    except Exception as e:
        print(terminal.error(f"Error initializing agent: {str(e)[:100]}"))
        await cl.Message(content=f"Error initializing assistant: {str(e)}. Please try refreshing or contact support.").send()
        return None

@cl.on_chat_start
async def start():
    """Initialize chat session with enhanced settings and error handling."""
    print(terminal.header("CHAT SESSION INITIALIZATION"))
    
    cl.user_session.set("history", [])
    
    # Get available blog files
    blog_files = glob.glob("./data/*.html")
    print(terminal.metric("Blog Files Found", str(len(blog_files))))
    
    if not blog_files:
        print(terminal.error("No data files found"))
        await cl.Message(content="⚠️ No data files found. Please ensure cybersecurity blog data is available in the ./data/ directory.").send()
        return
    
    try:
        print(terminal.step(1, 3, "Creating chat settings"))
        # Create enhanced chat settings
        settings = await cl.ChatSettings(
            [
                Select(
                    id="Model",
                    label="🤖 OpenAI Model",
                    values=["gpt-4o", "gpt-4o-mini"],
                    initial_index=0,
                ),
                Switch(
                    id="Streaming", 
                    label="🌊 Stream Response", 
                    initial=True
                ),
                Slider(
                    id="Temperature",
                    label="🌡️ Creativity (Temperature)",
                    initial=0.7,
                    min=0,
                    max=1,
                    step=0.1,
                    description="Higher values make responses more creative but less focused"
                ),
                Slider(
                    id="Max_Tokens",
                    label="📝 Max Response Length",
                    initial=4096,
                    min=512,
                    max=8192,
                    step=512,
                    description="Maximum length of AI responses"
                ),
                Slider(
                    id="Top_P",
                    label="🎯 Focus (Top-P)",
                    initial=0.9,
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    description="Controls response diversity"
                ),
                Slider(
                    id="Image_Size",
                    label="🖼️ Image Generation Size",
                    initial=1024,
                    min=256,
                    max=1024,
                    step=256,
                    tooltip="Size of generated images (square)"
                ),
                Select(
                    id="Image_Quality",
                    label="✨ Image Quality",
                    values=["standard", "hd"],
                    initial_index=0,
                ),
            ]
        ).send()
        
        cl.user_session.set("settings", settings)
        
        print(terminal.step(2, 3, "Initializing agent"))
        # Initialize agent
        agent = await initialize_agent(blog_files, settings)
        if agent:
            cl.user_session.set("agent", agent)
            print(terminal.step(3, 3, "Chat session ready"))
            print(terminal.success("Chat session initialized successfully"))
        else:
            print(terminal.error("Failed to initialize assistant"))
            await cl.Message(content="❌ Failed to initialize assistant. Please refresh and try again.").send()
            
    except Exception as e:
        print(terminal.error(f"Error during chat setup: {str(e)[:100]}"))
        await cl.Message(content=f"⚠️ Setup error: {str(e)}. Please refresh and try again.").send()

@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates with improved error handling."""
    print(terminal.subheader("SETTINGS UPDATE"))
    print(terminal.info(f"Model: {settings.get('Model', 'unknown')}, Streaming: {settings.get('Streaming', 'unknown')}"))
    try:
        blog_files = glob.glob("./data/*.html")
        agent = await initialize_agent(blog_files, settings)
        if agent:
            cl.user_session.set("agent", agent)
            cl.user_session.set("settings", settings)
            print(terminal.success("Settings updated successfully"))
        else:
            print(terminal.error("Failed to update agent with new settings"))
    except Exception as e:
        print(terminal.error(f"Error updating settings: {str(e)[:100]}"))

@cl.on_message
@monitor_performance
@security_scan
@rate_limit("request")
async def handle_message(message: cl.Message):
    """Enhanced message handler with streaming responses and better logging."""
    try:
        print(terminal.header("CHATBOT-ML REQUEST HANDLER"))
        
        # Initialize security scanner
        security_scanner = SecurityScanner()
        
        # Enhanced context analysis
        conversation_history = []
        if hasattr(cl.user_session, 'get') and cl.user_session.get("conversation_history"):
            conversation_history = cl.user_session.get("conversation_history", [])
        
        context_analyzer = ContextAnalyzer()
        context = context_analyzer.analyze_context(message.content, conversation_history)
        
        # Get history from session
        history = cl.user_session.get("history", [])
    
        # Check for image generation requests FIRST (before other processing)
        if is_image_generation_request(message.content):
            print(terminal.info("Image generation request detected"))
            await handle_image_generation(message.content)
            return

        # Check for image uploads
        image_analysis = ""
        if message.elements:
            logger.info(f"Message contains {len(message.elements)} elements")
            for element in message.elements:
                if isinstance(element, cl.Image):
                    print(terminal.info("Image element found"))
                    logger.info(f"Image element found: {element}")
                    try:    
                        # Analyze the image
                        print(terminal.step(1, 5, "Starting image analysis"))
                        logger.info("Analyzing image...")
                        
                        # Log detailed steps
                        if hasattr(element, 'path') and element.path:
                            print(terminal.step(2, 5, f"Reading image from path: {element.path}"))
                            logger.info(f"Image path: {element.path}")
                        elif hasattr(element, 'content') and element.content:
                            print(terminal.step(2, 5, "Using image content from memory"))
                            logger.info(f"Image content length: {len(element.content)} bytes")
                        
                        print(terminal.step(3, 5, "Processing with OpenAI Vision API"))
                        # Perform image analysis with detailed logging
                        image_analysis = await analyze_image(element)
                        
                        print(terminal.step(4, 5, "Analysis completed"))
                        logger.info(f"Image analysis result length: {len(image_analysis) if image_analysis else 0}")
                        
                        # Add image analysis to the message content
                        print(terminal.step(5, 5, "Adding analysis to message"))
                        message.content += f"\n\nImage analysis: {image_analysis}"
                        print(terminal.success("Image processing complete"))
                        
                        # Print the result for debugging
                        print(terminal.info(f"Image analysis result: {image_analysis[:100]}..."))
                        
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}", exc_info=True)
                        print(terminal.error(f"Image analysis failed: {str(e)}"))
                        await cl.Message(content=f"I'm sorry, but I encountered an error while processing the image: {str(e)}. Could you please try uploading it again?").send()
                else:
                    logger.info(f"Non-image element found: {type(element)}")
        else:
            logger.info("No elements found in the message")

        if not image_analysis:
            logger.info("No image analysis performed")

        # Check for file uploads
        for element in message.elements:
            if isinstance(element, cl.File):
                print(terminal.step(1, 3, f"Processing file upload: {element.name}"))
                file_ext = Path(element.name).suffix.lower()
                
                # Handle media files
                if file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg', '.m2v', '.ts', '.mts', '.m2ts', '.vob', '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.aiff', '.au', '.ra', '.ac3', '.dts', '.ape', '.tak', '.opus']:
                    try:
                        print(terminal.step(1, 3, f"Starting media analysis: {element.name}"))
                        print(terminal.info("Media upload detected - processing with comprehensive analysis"))
                        
                        media_result = await handle_media_upload(element)
                        await cl.Message(content=media_result).send()
                        print(terminal.success(f"Media analysis completed: {element.name}"))
                        return
                    except Exception as media_error:
                        print(terminal.error(f"Media processing failed: {str(media_error)[:100]}"))
                        await cl.Message(
                            content=f"❌ **Media Processing Error**\n\n**File:** {element.name}\n**Error:** {str(media_error)}\n\n**Supported Formats:** 50+ video and audio formats including MP4, MOV, AVI, MKV, MP3, WAV, FLAC, AAC, and many more."
                        ).send()
                        return
                
                # Handle document files
                elif file_ext in ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.csv', '.xls', '.xlsx', '.ppt', '.pptx']:
                    try:
                        print(terminal.step(1, 3, f"Starting document analysis: {element.name}"))
                        print(terminal.info("Document upload detected - processing with contextual analysis"))
                        
                        document_result = await handle_file_upload(element)
                        await cl.Message(content=document_result).send()
                        print(terminal.success(f"Document analysis completed: {element.name}"))
                        return
                    except Exception as doc_error:
                        print(terminal.error(f"Document processing failed: {str(doc_error)[:100]}"))
                        await cl.Message(
                            content=f"❌ **Document Processing Error**\n\n**File:** {element.name}\n**Error:** {str(doc_error)}\n\n**Supported Formats:** PDF, DOCX, TXT, CSV, XLSX, and more."
                        ).send()
                        return
                
                # Handle code files
                elif file_ext in ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.ts', '.sh', '.bat', '.ps1', '.sql', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.conf']:
                    try:
                        print(terminal.step(1, 3, f"Starting code analysis: {element.name}"))
                        print(terminal.info("Code file upload detected - processing with security analysis"))
                        
                        code_result = await handle_file_upload(element)
                        await cl.Message(content=code_result).send()
                        print(terminal.success(f"Code analysis completed: {element.name}"))
                        return
                    except Exception as code_error:
                        print(terminal.error(f"Code processing failed: {str(code_error)[:100]}"))
                        await cl.Message(
                            content=f"❌ **Code Processing Error**\n\n**File:** {element.name}\n**Error:** {str(code_error)}\n\n**Supported Formats:** Python, JavaScript, HTML, CSS, Java, C++, and many more programming languages."
                        ).send()
                        return
                
                else:
                    print(terminal.warning(f"Unsupported file format: {file_ext}"))
                    await cl.Message(
                        content=f"⚠️ **Unsupported File Format**\n\n**File:** {element.name}\n**Format:** {file_ext}\n\n**Supported Formats:**\n- Media: MP4, MOV, AVI, MP3, WAV, etc.\n- Documents: PDF, DOCX, TXT, CSV, etc.\n- Code: PY, JS, HTML, CSS, etc."
                    ).send()
                    return
        
        # URL detection and analysis (skip if we just processed an image)
        urls_found = extract_urls_from_message(message.content)
        should_crawl_url = should_auto_process_urls(message.content, urls_found)
        
        # Skip URL processing if we just analyzed an image (to avoid crawling URLs found in images)
        if image_analysis:
            logger.info("Skipping URL processing - image was just analyzed")
            should_crawl_url = False
        
        if should_crawl_url and urls_found:
            try:
                # Process the first URL found (optimized)
                url_info = urls_found[0]
                main_url = url_info['url']  # Extract the actual URL string from the dictionary
                print(terminal.processing(f"Processing URL: {main_url}"))
                
                # Create a progress message
                progress_msg = cl.Message(content=f"🔍 **Analyzing URL**: {main_url}\n\nRetrieving and analyzing content, please wait...")
                await progress_msg.send()
                
                # Analyze the URL
                url_context = message.content if len(message.content) < 500 else message.content[:500]
                url_analysis = await analyze_url_smart(main_url, force_refresh=False, context=url_context)
                
                # Update the progress message with the analysis
                await progress_msg.remove()
                await cl.Message(content=url_analysis).send()
                
                print(terminal.success(f"URL analysis completed: {main_url}"))
                return
            except Exception as url_error:
                print(terminal.error(f"URL processing failed: {str(url_error)[:100]}"))
                await cl.Message(
                    content=f"""❌ **URL Analysis Failed**

**URL:** {urls_found[0]['url']}
**Error:** {str(url_error)}

Please check if the URL is accessible and try again.
"""
                ).send()
                return
        
        # Enhanced query processing with streaming and detailed logging
        print(terminal.subheader("QUERY PROCESSING"))
        try:
            agent = cl.user_session.get("agent")
            
            if not agent:
                print(terminal.warning("No agent found, reinitializing..."))
                blog_files = glob.glob("./data/*.html")
                if not blog_files:
                    await cl.Message(content="⚠️ No cybersecurity blog data found. Please ensure data files are available in the ./data/ directory.").send()
                    return
                
                settings = cl.user_session.get("settings", DEFAULT_SETTINGS.copy())
                agent = await initialize_agent(blog_files, settings)
                
                if not agent:
                    await cl.Message(content="❌ Failed to initialize assistant. Please refresh and try again.").send()
                    return
                
                cl.user_session.set("agent", agent)
            
            # Create a response message
            response_message = cl.Message(content="")
            await response_message.send()

            print(terminal.metric("Query Length", f"{len(message.content)} chars"))
            print(terminal.metric("Knowledge Base", f"{len(glob.glob('./data/*.html'))} files"))
            
            try:
                logger.info("🚀 Starting query processing...")
                full_response = ""
                
                # Enhanced blog query detection
                is_blog_query, potential_topics = is_blog_related_query(message.content)
                
                if is_blog_query:
                    print(terminal.step(1, 3, "Blog query detected - searching knowledge base"))
                    logger.info("📚 Blog query detected, using enhanced knowledge retrieval")
                    
                    # Use enhanced blog retrieval
                    blog_response = await enhance_blog_knowledge_retrieval(message.content, agent, potential_topics)
                    
                    if blog_response:
                        print(terminal.step(2, 3, "Processing blog-related query"))
                        full_response = blog_response
                        await response_message.stream_token(full_response)
                        
                        print(terminal.step(3, 3, "Sending response"))
                        # Add to conversation history and return early to prevent duplicate responses
                        add_to_conversation_history(message.content, full_response)
                        logger.info(f"✅ Query processed successfully: {len(full_response)} chars")
                        return
                    else:
                        # Fallback to normal query
                        print(terminal.warning("No blog content matched, using standard processing"))
                
                settings = cl.user_session.get("settings", DEFAULT_SETTINGS.copy())
                
                try:
                    logger.info("🎯 Initiating agent query...")
                    logger.info(f"🤖 Using model: {settings.get('Model', 'gpt-4o')}")
                    logger.info(f"📚 Available tools: {len(global_tools or [])} knowledge sources")
                    
                    # Create context from conversation history
                    context = "\n".join([f"User: {item['user']}\nAssistant: {item['bot']}" for item in history[-3:]])
                    full_query = f"Given the conversation history:\n{context}\n\nUser query: {message.content}\n\nPlease provide a relevant and accurate response to the user's query, using the information from the cybersecurity blogs and any image or media analysis provided."
                    
                    # Stream the response if enabled
                    if settings.get("Streaming", DEFAULT_SETTINGS["Streaming"]):
                        print(terminal.step(2, 3, "Processing with streaming enabled"))
                        streaming_response = await cl.make_async(agent.stream_chat)(full_query)
                        
                        # Handle different streaming response types
                        if hasattr(streaming_response, 'response_gen'):
                            for token in streaming_response.response_gen:
                                full_response += token
                                await response_message.stream_token(token)
                            
                            logger.info(f"✅ Streaming response completed ({len(full_response)} chars)")
                        else:
                            print(terminal.step(2, 3, "Analyzing user context"))
                            user_context = extract_user_context_clues(message.content)
                            
                            enhanced_query = message.content
                            if is_detailed_explanation_request(message.content):
                                print(terminal.info("📝 Detailed explanation request detected"))
                                enhanced_query = f"{message.content}\n\nPlease provide a comprehensive, detailed explanation with examples, step-by-step guidance, and practical applications. Consider that the user wants in-depth understanding of this topic."
                            
                            # Use fallback method for streaming
                            print(terminal.step(2, 3, "Using fallback streaming method"))
                            response = await cl.make_async(agent.chat)(full_query)
                            full_response = response.response
                            
                            # Enhance response with context if needed
                            if user_context.get('skill_level') or user_context.get('purpose'):
                                print(terminal.info("🧠 Enhancing response with user context"))
                                enhanced_response = await enhance_response_with_context(
                                    full_response, user_context, message.content
                                )
                                full_response = enhanced_response
                            
                            # Stream the response token by token (optimized for speed)
                            for i in range(0, len(full_response), 5):  # Larger chunks for faster streaming
                                chunk = full_response[i:i+5]
                                await response_message.stream_token(chunk)
                                await asyncio.sleep(0.005)  # Reduced delay for faster streaming
                            
                            logger.info(f"✅ Fallback streaming completed ({len(full_response)} chars)")
                    else:
                            print(terminal.step(2, 3, "Processing with standard response"))
                            response = await cl.make_async(agent.chat)(full_query)
                            full_response = response.response
                            await response_message.update(content=full_response)
                            logger.info(f"✅ Standard response completed ({len(full_response)} chars)")
                        
                    # Update the final content
                    response_message.content = full_response
                    await response_message.update()
                
                    # Blog query enhancement
                    if is_blog_query:
                        print(terminal.step(3, 3, "Enhancing with blog knowledge"))
                        blog_response = await enhance_blog_knowledge_retrieval(message.content, agent, potential_topics)
                        
                        if blog_response:
                            full_response = blog_response
                            await response_message.stream_token(full_response)
                            
                            # Add to conversation history and return early to prevent duplicate responses
                            add_to_conversation_history(message.content, full_response)
                            logger.info(f"✅ Query processed successfully: {len(full_response)} chars")
                            return
                        else:
                            # Fallback to normal query
                            print(terminal.warning("No blog content matched, using standard processing"))
                    
                    logger.info(f"✅ Final response completed ({len(full_response)} total chars)")
                
                except Exception as streaming_error:
                    logger.warning(f"Streaming failed: {streaming_error}, trying sync approach")
                    print(terminal.warning(f"Streaming failed, using fallback method"))
                    
                    try:
                        print(terminal.step(2, 3, "Using synchronous fallback"))
                        response = await cl.make_async(agent.chat)(full_query)
                        full_response = response.response
                        await response_message.update(content=full_response)
                        
                        logger.info(f"✅ Fallback response completed ({len(full_response)} chars)")
                    except Exception as sync_error:
                        logger.error(f"Both streaming and sync failed: {sync_error}")
                        print(terminal.error(f"All response methods failed: {str(sync_error)[:100]}"))
                        await response_message.update(content="I apologize, but I encountered an error while processing your query. Could you please rephrase or try again?")
                        
                        logger.error(f"❌ Both methods failed, showing error message")
                
                if is_blog_query and full_response:
                    print(terminal.step(3, 3, "Processing blog-related query"))
                    print(terminal.success("Blog query processed successfully"))
                
                print(terminal.success(f"Query processed successfully ({len(full_response)} characters)"))
                
                conversation_history.append({
                    "role": "user", 
                    "content": message.content,
                    "timestamp": datetime.now().isoformat()
                })
                conversation_history.append({
                    "role": "assistant", 
                    "content": full_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Store in conversation history
                cl.user_session.set("conversation_history", conversation_history)
                
                # Store in history for session
                history = cl.user_session.get("history", [])
                history.append({"user": message.content, "bot": full_response})
                cl.user_session.set("history", history)
                logger.info("Response sent and history updated")
                
                if global_query_engine and hasattr(global_query_engine, 'get_stats'):
                    try:
                        stats = global_query_engine.get_stats()
                        logger.info(f"Query engine stats: {stats}")
                    except Exception as stats_error:
                        logger.warning(f"Could not display query statistics: {stats_error}")
                        
            except Exception as inner_query_error:
                print(terminal.error(f"Inner query processing error: {str(inner_query_error)[:100]}"))
                await cl.Message(
                    content=f"""❌ **System Error**

**Error:** {str(inner_query_error)}

I apologize for the inconvenience. The system encountered an error while processing your query. Please try again or rephrase your question.
"""
                ).send()
        except Exception as outer_query_error:
            print(terminal.error(f"Outer query processing error: {str(outer_query_error)[:100]}"))
            await cl.Message(
                content=f"""❌ **System Error**

**Error:** {str(outer_query_error)}

I apologize for the inconvenience. The system encountered an error while setting up the query processor. Please refresh the page and try again.
"""
            ).send()
    except Exception as e:
        print(terminal.error(f"Global handler error: {str(e)[:100]}"))
        logger.error(f"Error in message handler: {e}", exc_info=True)
        await cl.Message(content="I apologize, but I encountered an error while processing your message. Please try again.").send()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Process resumed chat sessions with enhanced conversation history."""
    try:
        terminal = TerminalFormatter()
        print(terminal.header("CHAT RESUMED"))
        
        # Initialize variables for tracking
        restored_messages = []
        message_count = 0
        current_user_id = cl.user_session.get("user_id", "default")
        
        try:
            # Get conversation from thread
            if 'messages' in thread:
                for msg in thread['messages']:
                    try:
                        if 'author' in msg and 'content' in msg and msg.get('content'):
                            if msg['author'] == 'User':
                                add_to_conversation_storage(current_user_id, "user", msg['content'])
                                restored_messages.append(f"User: {msg['content'][:50]}...")
                            elif msg['author'] == 'Assistant':
                                add_to_conversation_storage(current_user_id, "assistant", msg['content'])
                                restored_messages.append(f"Assistant: {msg['content'][:50]}...")
                            message_count += 1
                    except Exception as message_error:
                        print(f"Error processing message: {message_error}")
                
            # Alternative format (steps)
            elif 'steps' in thread:
                for step in thread['steps']:
                    try:
                        if 'input' in step and step.get('input'):
                            add_to_conversation_storage(current_user_id, "user", step['input'])
                            restored_messages.append(f"User: {step['input'][:50]}...")
                            message_count += 1
                        
                        if 'output' in step and step.get('output'):
                            add_to_conversation_storage(current_user_id, "assistant", step['output'])
                            restored_messages.append(f"Assistant: {step['output'][:50]}...")
                            message_count += 1
                    except Exception as step_error:
                        print(f"Error processing step: {step_error}")
            
            # Log results
            if message_count > 0:
                print(terminal.success(f"Restored {message_count} messages from conversation history"))
                print("Messages restored:")
                for msg in restored_messages:
                    print(f"  - {msg}")
                
                # Send confirmation to user
                await cl.Message(
                    content=f"""## 💬 **Chat Session Resumed**

**Previous conversation history restored.**

You can continue your conversation and reference previous questions and answers. I'll maintain context from your earlier conversation."""
                ).send()
        except Exception as resume_error:
            print(terminal.error(f"Error resuming chat: {str(resume_error)}"))
            logger.error(f"Chat resume error: {str(resume_error)}", exc_info=True)
            await cl.Message(
                content="""## 💬 **Chat Session Resumed**

There was a problem restoring your previous conversation. Let's start a new conversation!"""
            ).send()
            
    except Exception as e:
        logger.error(f"General resume error: {str(e)}", exc_info=True)
        print(f"Error in on_chat_resume: {str(e)}")

@cl.set_starters
async def set_starters():
    """Enhanced starter prompts for better user onboarding."""
    return [
        cl.Starter(
            label="Red Team Fundamentals",
            message="Explain red team operations in cybersecurity, including their main objectives, methodologies, and how they differ from blue teams in an organization's security strategy.",
            icon="/public/red-team.svg",
        ),
        cl.Starter(
            label="Bug Bounty Mastery",
            message="What are bug bounty programs and how do they work? Provide examples of major platforms, common vulnerability types, and tips for successful bug hunting.",
            icon="/public/bug-bounty.svg",
        ),
        cl.Starter(
            label="Vulnerability Assessment",
            message="What are the most critical vulnerabilities that penetration testers should look for? Explain OWASP Top 10, common attack vectors, and assessment methodologies.",
            icon="/public/vulnerability.svg",
        ),
        cl.Starter(
            label="Pentesting Career Path",
            message="I want to start a career in penetration testing and ethical hacking. What skills should I develop, certifications to pursue, and practical steps to get started?",
            icon="/public/getting-started.svg",
        )
    ]

async def handle_image_generation(message_content: str):
    """Handle image generation requests with clean, simple output."""
    try:
        logger.info("🎨 Processing image generation request")
        
        # Extract prompt from message
        prompt_patterns = [
            r"generate image[:\s]+(.+)",
            r"create image[:\s]+(.+)",
            r"make image[:\s]+(.+)",
            r"draw[:\s]+(.+)",
            r"illustrate[:\s]+(.+)",
            r"design[:\s]+(.+)"
        ]
        
        extracted_prompt = None
        for pattern in prompt_patterns:
            match = re.search(pattern, message_content, re.IGNORECASE)
            if match:
                extracted_prompt = match.group(1).strip()
                break
        
        if not extracted_prompt:
            # Use the entire message as prompt if no specific pattern found
            extracted_prompt = message_content
        
        # Remove common command words if they appear at the start
        command_words = ["generate", "create", "make", "draw", "illustrate", "design", "image", "picture"]
        words = extracted_prompt.split()
        while words and words[0].lower() in command_words:
            words.pop(0)
        extracted_prompt = " ".join(words)
        
        if not extracted_prompt:
            await cl.Message(
                content="I need a description of what image you'd like me to generate. For example: 'Generate an image of a cybersecurity professional working at a computer'"
            ).send()
            return
        
        # Show progress message
        progress_msg = cl.Message(content="🎨 Creating your image... This may take a moment.")
        await progress_msg.send()
        
        logger.info(f"🎯 Generating image with prompt: {extracted_prompt}")
        
        # Generate the image
        result = await generate_image_advanced(
            prompt=extracted_prompt,
            size="1024x1024",
            quality="hd",
            style="professional"
        )
        
        if "error" in result:
            error_message = f"❌ Image generation failed: {result['error']}"
            # Create new message instead of updating
            await cl.Message(content=error_message).send()
            return
        
        # Create proper Chainlit image element for display
        image_url = result.get("image_url")
        
        if image_url:
            try:
                # Create an image element that Chainlit can display
                image_element = cl.Image(
                    url=image_url,
                    name="generated_image",
                    display="inline"
                )
                
                # Create a clean, simple response
                response_content = f"✅ **Image Generated Successfully!**"
                
                # Delete the progress message first
                try:
                    await progress_msg.remove()
                except:
                    pass  # If removal fails, continue anyway
                
                # Send new message with the image
                await cl.Message(
                    content=response_content,
                    elements=[image_element]
                ).send()
                
                logger.info("✅ Image generated and displayed successfully")
                
            except Exception as display_error:
                logger.error(f"Error displaying image: {display_error}")
                # Fallback: show URL if display fails
                fallback_content = f"✅ **Image Generated!**\n\n🔗 **View Image:** {image_url}"
                
                await cl.Message(content=fallback_content).send()
        else:
            await cl.Message(content="❌ Image generation failed: No image URL received").send()
            
    except Exception as e:
        logger.error(f"Error in image generation handler: {str(e)}", exc_info=True)
        await cl.Message(
            content=f"❌ Image generation error: {str(e)[:200]}. Please try again with a different prompt."
        ).send()

# Enhanced blog query detection
def is_blog_related_query(message: str) -> Tuple[bool, List[str]]:
    """Enhanced detection for blog-related queries with better relevance detection."""
    blog_keywords = [
        'blog', 'article', 'post', 'tutorial', 'guide', 'write-up', 'writeup',
        'cybersecurity', 'pentest', 'red team', 'vulnerability', 'exploit',
        'analysis', 'security research', 'security blog', 'explain'
    ]
    
    # Advanced detection of blog requests
    is_blog_query = any(keyword in message.lower() for keyword in blog_keywords)
    
    # Extract blog topics for more targeted search
    potential_topics = []
    for keyword in ['cve-', 'xss', 'sqli', 'rce', 'csrf', 'ssrf', 'path traversal', 
                   'injection', 'buffer overflow', 'dos', 'authentication', 
                   'authorization', 'encryption']:
        if keyword in message.lower():
            potential_topics.append(keyword)
    
    # Look for explicit references to blog content
    if re.search(r'(in|from|about|the|your|reference|related) blogs?', message.lower()):
        is_blog_query = True
        
    # Look for questions about topics that would be in blogs
    if re.search(r'what.+(vuln|attack|exploit|technique|method|approach)', message.lower()):
        is_blog_query = True
        
    return is_blog_query, potential_topics

def extract_urls_from_message(message_content: str) -> List[Dict[str, Any]]:
    """Extract URLs from message with enhanced context awareness."""
    # First, look for standard URLs
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    found_urls = re.findall(url_pattern, message_content)
    
    # Look for URLs with @ symbol (common in chat interfaces)
    at_urls = re.findall(r'@(https?://[^\s<>"]+)', message_content)
    found_urls.extend(at_urls)
    
    # Generate context for each URL
    url_contexts = []
    
    for url in found_urls:
        # Clean URL if it has @ prefix
        clean_url = url.strip('@')
        
        # Find text surrounding this URL for context
        url_position = message_content.find(url)
        start_context = max(0, url_position - 50)
        end_context = min(len(message_content), url_position + len(url) + 50)
        surrounding_text = message_content[start_context:end_context]
        
        # Determine if this is a direct question about the URL
        is_direct_question = False
        pre_url_text = message_content[:url_position].lower().strip()
        
        for pattern in ['what is', 'tell me about', 'analyze', 'explain', 'describe', 
                       'what does', 'can you check', 'look at', 'tell me what']:
            if pre_url_text.endswith(pattern) or f"{pattern} this" in pre_url_text:
                is_direct_question = True
                break
        
        url_contexts.append({
            'url': clean_url,
            'surrounding_context': surrounding_text,
            'is_direct_question': is_direct_question
        })
    
    return url_contexts

async def enhance_blog_knowledge_retrieval(query: str, agent, potential_topics: List[str] = None):
    """Enhanced blog knowledge retrieval with better context and relevance."""
    try:
        # Find all blog files
        blog_files = glob.glob('./data/*.html')
        if not blog_files:
            return "No blog knowledge found. Please configure your knowledge base."
        
        # Always do full search if topics are provided
        force_full_search = bool(potential_topics)
        
        # Build context for query
        enhanced_query = query
        if potential_topics:
            topic_context = ", ".join(potential_topics)
            enhanced_query = f"{query}\n\nRelevant topics: {topic_context}"
        
        # Try to get existing knowledge
        if not force_full_search:
            try:
                # Extract URLs from the query
                url_dicts = extract_urls_from_message(query)
                existing_response = await check_existing_knowledge(url_dicts, agent)
                if existing_response and len(existing_response) > 200:
                    return existing_response
            except Exception as e:
                logger.warning(f"Error checking existing knowledge: {e}")
        
        # Do comprehensive search across all blogs
        all_results = []
        terminal = TerminalFormatter()
        
        print(terminal.header("SEARCHING BLOGS"))
        
        # Get tools from agent
        tools = agent._tools if hasattr(agent, "_tools") else []
        for tool in tools:
            try:
                if "search" in tool.metadata.name.lower():
                    print(terminal.info(f"Searching using {tool.metadata.name}"))
                    result = await cl.make_async(agent.chat)(f"Using {tool.metadata.name}, find detailed information about: {enhanced_query}")
                    if result and hasattr(result, 'response') and len(result.response) > 100:
                        all_results.append(result.response)
            except Exception as tool_error:
                logger.warning(f"Error with tool {getattr(tool, 'name', 'unknown')}: {str(tool_error)}")
        
        # Combine results for better completeness
        if all_results:
            combined_result = "\n\n".join(all_results)
            if len(combined_result) > 100:
                print(terminal.success(f"Found {len(all_results)} relevant results"))
                return combined_result
        
        # Fallback to direct query to the agent
        print(terminal.info("Using direct query to agent"))
        result = await cl.make_async(agent.chat)(f"Find detailed information from your knowledge base about: {enhanced_query}")
        response = result.response if hasattr(result, 'response') else str(result)
        
        if response and len(response) > 100:
            print(terminal.success("Successfully retrieved information"))
            return response
        else:
            print(terminal.warning("No relevant information found"))
            return None
    
    except Exception as e:
        logger.error(f"Error in blog knowledge retrieval: {str(e)}", exc_info=True)
        return None

def is_image_generation_request(content: str) -> bool:
    """Detect if the user is requesting image generation."""
    image_generation_patterns = [
        r'\bgenerate\s+(?:an?\s+)?image\b',
        r'\bcreate\s+(?:an?\s+)?image\b',
        r'\bmake\s+(?:an?\s+)?image\b',
        r'\bdraw\s+(?:an?\s+)?(?:image|picture)\b',
        r'\billustrate\b',
        r'\bdesign\s+(?:an?\s+)?image\b',
        r'\bgenerate\s+(?:a\s+)?picture\b',
        r'\bcreate\s+(?:a\s+)?picture\b',
        r'\bmake\s+(?:a\s+)?picture\b'
    ]
    
    content_lower = content.lower()
    return any(re.search(pattern, content_lower) for pattern in image_generation_patterns)

if __name__ == "__main__":
    print(terminal.header("PENTEST KNOWLEDGE ASSISTANT", "█"))
    print(terminal.info("Enhanced AI Assistant for Cybersecurity Red Team and Penetration Testing"))
    print(terminal.separator())
    
    print(terminal.subheader("SYSTEM STARTUP"))
    
    # Start Discord bot in separate thread if token is available
    if os.getenv("DISCORD_BOT_TOKEN"):
        print(terminal.info("Discord bot token found - starting integration"))
        discord_thread = threading.Thread(target=run_discord_bot, daemon=True)
        discord_thread.start()
        print(terminal.success("Discord bot started successfully"))
    else:
        print(terminal.warning("Discord bot token not found - skipping Discord integration"))
    
    print(terminal.separator())
    print(terminal.info("Starting Chainlit application..."))
    print(terminal.metric("Interface", "Web UI"))
    print(terminal.metric("Port", "8000"))
    print(terminal.metric("Mode", "Development"))
    print(terminal.success("Ready to serve requests"))
    
    # Run Chainlit app
    cl.run()
