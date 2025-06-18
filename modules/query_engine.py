from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings
from pathlib import Path
import os
import logging
import time
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

logger = logging.getLogger(__name__)

# Initialize colorama for cross-platform color support
colorama.init(autoreset=True)

class TerminalQueryLogger:
    """Enhanced terminal logger for query operations."""
    
    def __init__(self):
        self.width = 80
        
    def header(self, text: str) -> str:
        """Create a formatted header."""
        padding = (self.width - len(text) - 2) // 2
        return f"\n{Fore.CYAN}{'═' * padding} {text} {'═' * padding}{Style.RESET_ALL}"
    
    def blog_search_start(self, blog_name: str, query: str) -> str:
        """Log blog search start."""
        return f"{Fore.YELLOW}🔍 SEARCHING BLOG:{Style.RESET_ALL} {Fore.WHITE}{blog_name}{Style.RESET_ALL}"
    
    def query_info(self, query: str) -> str:
        """Log query information."""
        query_preview = query[:60] + "..." if len(query) > 60 else query
        return f"{Fore.BLUE}📝 Query:{Style.RESET_ALL} {query_preview}"
    
    def blog_result(self, blog_name: str, has_result: bool, char_count: int = 0) -> str:
        """Log blog search result."""
        if has_result:
            return f"{Fore.GREEN}✅ FOUND in {blog_name}:{Style.RESET_ALL} {char_count} characters"
        else:
            return f"{Fore.RED}❌ NO MATCH in {blog_name}{Style.RESET_ALL}"
    
    def cve_detection(self, cves: list) -> str:
        """Log CVE detection."""
        return f"{Fore.MAGENTA}🔒 CVE FOUND:{Style.RESET_ALL} {', '.join(cves[:3])}"
    
    def security_content(self, blog_name: str) -> str:
        """Log security content detection."""
        return f"{Fore.CYAN}🛡️  SECURITY CONTENT in {blog_name}{Style.RESET_ALL}"
    
    def synthesis_start(self, source_count: int) -> str:
        """Log synthesis start."""
        return f"{Fore.YELLOW}🎯 SYNTHESIZING RESPONSE from {source_count} sources{Style.RESET_ALL}"
    
    def function_call(self, function_name: str, blog_name: str) -> str:
        """Log function call."""
        return f"{Fore.MAGENTA}⚙️  CALLING:{Style.RESET_ALL} {function_name}({blog_name})"
    
    def timing(self, operation: str, duration: float) -> str:
        """Log timing information."""
        return f"{Fore.CYAN}⏱️  {operation}:{Style.RESET_ALL} {duration:.2f}s"

# Global terminal logger
terminal_logger = TerminalQueryLogger()

def setup_query_engine(index_set, blog_files):
    # Create LLM instance for consistency
    llm = LlamaOpenAI(
        model="gpt-4o", 
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
        max_tokens=2048
    )
    
    # Set global LLM for all query engines
    Settings.llm = llm
    
    individual_query_engine_tools = []

    print(terminal_logger.header("INITIALIZING BLOG QUERY ENGINES"))
    print(f"{Fore.CYAN}📚 Setting up {len(blog_files)} blog indexes...{Style.RESET_ALL}")

    for i, blog_file in enumerate(blog_files, 1):
        if blog_file in index_set:
            blog_name = Path(blog_file).stem
            
            print(f"{Fore.GREEN}[{i}/{len(blog_files)}]{Style.RESET_ALL} Initializing: {Fore.WHITE}{blog_name}{Style.RESET_ALL}")
            
            # Create query engine with explicit LLM configuration
            query_engine = index_set[blog_file].as_query_engine(
                llm=llm,
                similarity_top_k=3,
                response_mode="compact"
            )
            
            # Create a wrapper that logs queries during actual usage with enhanced details
            class LoggingQueryEngine:
                def __init__(self, engine, blog_name, blog_file, index_number):
                    self.engine = engine
                    self.blog_name = blog_name
                    self.blog_file = blog_file
                    self.index_number = index_number
                    self.query_count = 0
                
                def query(self, query_str):
                    self.query_count += 1
                    start_time = time.time()
                    
                    print(f"\n{terminal_logger.function_call(f'query_blog_{self.index_number}', self.blog_name)}")
                    print(f"{terminal_logger.blog_search_start(self.blog_name, query_str)}")
                    print(f"{terminal_logger.query_info(query_str)}")
                    print(f"{Fore.CYAN}📁 File:{Style.RESET_ALL} {Path(self.blog_file).name}")
                    
                    try:
                        result = self.engine.query(query_str)
                        duration = time.time() - start_time
                        
                        if result and str(result).strip():
                            response_length = len(str(result))
                            response_preview = str(result)[:120].replace('\n', ' ')
                            
                            print(f"{terminal_logger.blog_result(self.blog_name, True, response_length)}")
                            print(f"{Fore.WHITE}📄 Preview:{Style.RESET_ALL} {response_preview}...")
                            
                            # Show key information if it's a CVE or vulnerability related
                            result_text = str(result).lower()
                            if 'cve-' in result_text:
                                import re
                                cves_found = re.findall(r'cve-\d{4}-\d{4,7}', result_text, re.IGNORECASE)
                                if cves_found:
                                    print(f"{terminal_logger.cve_detection(cves_found)}")
                            
                            if any(keyword in result_text for keyword in ['vulnerability', 'exploit', 'security']):
                                print(f"{terminal_logger.security_content(self.blog_name)}")
                            
                            print(f"{terminal_logger.timing('Search completed', duration)}")
                        else:
                            print(f"{terminal_logger.blog_result(self.blog_name, False)}")
                            print(f"{terminal_logger.timing('Search completed (no results)', duration)}")
                        
                        return result
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        print(f"{Fore.RED}❌ ERROR in {self.blog_name}:{Style.RESET_ALL} {str(e)[:100]}")
                        print(f"{terminal_logger.timing('Search failed', duration)}")
                        return None
                
                async def aquery(self, query_str):
                    self.query_count += 1
                    start_time = time.time()
                    
                    print(f"\n{terminal_logger.function_call(f'aquery_blog_{self.index_number}', self.blog_name)}")
                    print(f"{terminal_logger.blog_search_start(self.blog_name, query_str)}")
                    print(f"{terminal_logger.query_info(query_str)}")
                    print(f"{Fore.CYAN}📁 File:{Style.RESET_ALL} {Path(self.blog_file).name}")
                    
                    try:
                        result = await self.engine.aquery(query_str)
                        duration = time.time() - start_time
                        
                        if result and str(result).strip():
                            response_length = len(str(result))
                            response_preview = str(result)[:120].replace('\n', ' ')
                            
                            print(f"{terminal_logger.blog_result(self.blog_name, True, response_length)}")
                            print(f"{Fore.WHITE}📄 Preview:{Style.RESET_ALL} {response_preview}...")
                            
                            # Show key information if it's a CVE or vulnerability related
                            result_text = str(result).lower()
                            if 'cve-' in result_text:
                                import re
                                cves_found = re.findall(r'cve-\d{4}-\d{4,7}', result_text, re.IGNORECASE)
                                if cves_found:
                                    print(f"{terminal_logger.cve_detection(cves_found)}")
                            
                            if any(keyword in result_text for keyword in ['vulnerability', 'exploit', 'security']):
                                print(f"{terminal_logger.security_content(self.blog_name)}")
                            
                            print(f"{terminal_logger.timing('Async search completed', duration)}")
                        else:
                            print(f"{terminal_logger.blog_result(self.blog_name, False)}")
                            print(f"{terminal_logger.timing('Async search completed (no results)', duration)}")
                        
                        return result
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        print(f"{Fore.RED}❌ ASYNC ERROR in {self.blog_name}:{Style.RESET_ALL} {str(e)[:100]}")
                        print(f"{terminal_logger.timing('Async search failed', duration)}")
                        return None
                
                def get_stats(self):
                    """Get query statistics for this blog."""
                    return {
                        "blog_name": self.blog_name,
                        "query_count": self.query_count,
                        "index_number": self.index_number
                    }
                
                def __getattr__(self, name):
                    return getattr(self.engine, name)
            
            logging_engine = LoggingQueryEngine(query_engine, blog_name, blog_file, i)
            
            tool_name = f"idx_blog_{i:02d}_{blog_name[:15]}"
            individual_query_engine_tools.append(
                QueryEngineTool(
                    query_engine=logging_engine,
                    metadata=ToolMetadata(
                        name=tool_name,
                        description=f"Search blog #{i}: '{blog_name}' for cybersecurity red team and bug bounty information",
                    ),
                )
            )
            
            print(f"{Fore.GREEN}✅ Tool created:{Style.RESET_ALL} {tool_name}")
        else:
            print(f"{Fore.RED}⚠️  Skipping {Path(blog_file).name} - indexing failed{Style.RESET_ALL}")

    if not individual_query_engine_tools:
        print(f"{Fore.RED}❌ No query engine tools created. Check if any files were successfully indexed.{Style.RESET_ALL}")
        return None, []

    print(f"\n{Fore.GREEN}✅ Successfully created {len(individual_query_engine_tools)} blog query engines{Style.RESET_ALL}")

    # Create sub-question query engine with enhanced logging
    class LoggingSubQuestionQueryEngine:
        def __init__(self, engine, tools):
            self.engine = engine
            self.tools = tools
            self.query_count = 0
        
        def query(self, query_str):
            self.query_count += 1
            start_time = time.time()
            
            print(f"\n{terminal_logger.header('MULTI-BLOG ANALYSIS')}")
            print(f"{terminal_logger.function_call('multi_blog_query', 'all_blogs')}")
            print(f"{terminal_logger.query_info(query_str)}")
            print(f"{terminal_logger.synthesis_start(len(self.tools))}")
            
            # List all blogs that will be searched
            print(f"\n{Fore.YELLOW}📚 Blogs to search:{Style.RESET_ALL}")
            for i, tool in enumerate(self.tools, 1):
                blog_name = tool.metadata.name.replace('idx_blog_', '').replace('_', ' ')
                print(f"   {Fore.WHITE}[{i:2d}]{Style.RESET_ALL} {blog_name}")
            
            print(f"\n{Fore.CYAN}🚀 Starting parallel blog searches...{Style.RESET_ALL}")
            
            try:
                result = self.engine.query(query_str)
                duration = time.time() - start_time
                
                if result:
                    final_response = str(result)
                    print(f"\n{terminal_logger.header('SYNTHESIS COMPLETE')}")
                    print(f"{Fore.GREEN}🎯 Final response ready:{Style.RESET_ALL} {len(final_response):,} characters")
                    print(f"{terminal_logger.timing('Total analysis time', duration)}")
                    
                    # Show brief preview of synthesized result
                    preview = final_response[:200].replace('\n', ' ')
                    print(f"{Fore.WHITE}📋 Response preview:{Style.RESET_ALL} {preview}...")
                    
                    # Count how many sources contributed
                    source_indicators = ['according to', 'based on', 'from the', 'research shows', 'analysis indicates']
                    source_count = sum(1 for indicator in source_indicators if indicator in final_response.lower())
                    if source_count > 0:
                        print(f"{Fore.CYAN}📊 Sources contributing:{Style.RESET_ALL} ~{source_count} knowledge sources")
                else:
                    print(f"{Fore.YELLOW}⚠️  No synthesized response generated{Style.RESET_ALL}")
                    print(f"{terminal_logger.timing('Analysis completed (no results)', duration)}")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"{Fore.RED}❌ Multi-blog analysis error:{Style.RESET_ALL} {str(e)[:100]}")
                print(f"{terminal_logger.timing('Analysis failed', duration)}")
                return None
        
        async def aquery(self, query_str):
            self.query_count += 1
            start_time = time.time()
            
            print(f"\n{terminal_logger.header('ASYNC MULTI-BLOG ANALYSIS')}")
            print(f"{terminal_logger.function_call('async_multi_blog_query', 'all_blogs')}")
            print(f"{terminal_logger.query_info(query_str)}")
            print(f"{terminal_logger.synthesis_start(len(self.tools))}")
            
            # List all blogs that will be searched
            print(f"\n{Fore.YELLOW}📚 Blogs to search:{Style.RESET_ALL}")
            for i, tool in enumerate(self.tools, 1):
                blog_name = tool.metadata.name.replace('idx_blog_', '').replace('_', ' ')
                print(f"   {Fore.WHITE}[{i:2d}]{Style.RESET_ALL} {blog_name}")
            
            print(f"\n{Fore.CYAN}🚀 Starting async parallel blog searches...{Style.RESET_ALL}")
            
            try:
                result = await self.engine.aquery(query_str)
                duration = time.time() - start_time
                
                if result:
                    final_response = str(result)
                    print(f"\n{terminal_logger.header('ASYNC SYNTHESIS COMPLETE')}")
                    print(f"{Fore.GREEN}🎯 Final async response ready:{Style.RESET_ALL} {len(final_response):,} characters")
                    print(f"{terminal_logger.timing('Total async analysis time', duration)}")
                    
                    # Show brief preview of synthesized result
                    preview = final_response[:200].replace('\n', ' ')
                    print(f"{Fore.WHITE}📋 Response preview:{Style.RESET_ALL} {preview}...")
                    
                    # Count how many sources contributed
                    source_indicators = ['according to', 'based on', 'from the', 'research shows', 'analysis indicates']
                    source_count = sum(1 for indicator in source_indicators if indicator in final_response.lower())
                    if source_count > 0:
                        print(f"{Fore.CYAN}📊 Sources contributing:{Style.RESET_ALL} ~{source_count} knowledge sources")
                else:
                    print(f"{Fore.YELLOW}⚠️  No async synthesized response generated{Style.RESET_ALL}")
                    print(f"{terminal_logger.timing('Async analysis completed (no results)', duration)}")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"{Fore.RED}❌ Async multi-blog analysis error:{Style.RESET_ALL} {str(e)[:100]}")
                print(f"{terminal_logger.timing('Async analysis failed', duration)}")
                return None
        
        def get_stats(self):
            """Get overall query statistics."""
            tool_stats = []
            for tool in self.tools:
                if hasattr(tool.query_engine, 'get_stats'):
                    tool_stats.append(tool.query_engine.get_stats())
            
            return {
                "total_queries": self.query_count,
                "blog_count": len(self.tools),
                "individual_stats": tool_stats
            }
        
        def __getattr__(self, name):
            return getattr(self.engine, name)
    
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        llm=llm,
        use_async=True
    )

    logging_query_engine = LoggingSubQuestionQueryEngine(query_engine, individual_query_engine_tools)

    print(f"\n{terminal_logger.header('QUERY ENGINE READY')}")
    print(f"{Fore.GREEN}✅ Multi-blog query engine initialized{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 Ready to search {len(individual_query_engine_tools)} blog indexes{Style.RESET_ALL}")
    print(f"{Fore.WHITE}🎯 Functions available:{Style.RESET_ALL}")
    for i, tool in enumerate(individual_query_engine_tools, 1):
        tool_name = tool.metadata.name
        print(f"   {Fore.CYAN}[{i:2d}]{Style.RESET_ALL} {tool_name}()")
    
    return logging_query_engine, individual_query_engine_tools

def display_query_statistics(query_engine):
    """Display comprehensive query statistics."""
    if hasattr(query_engine, 'get_stats'):
        stats = query_engine.get_stats()
        
        print(f"\n{terminal_logger.header('QUERY STATISTICS')}")
        print(f"{Fore.CYAN}📊 Total Queries Processed:{Style.RESET_ALL} {stats.get('total_queries', 0)}")
        print(f"{Fore.CYAN}📚 Blog Indexes Available:{Style.RESET_ALL} {stats.get('blog_count', 0)}")
        
        if 'individual_stats' in stats:
            print(f"\n{Fore.YELLOW}📋 Individual Blog Statistics:{Style.RESET_ALL}")
            for i, blog_stat in enumerate(stats['individual_stats'], 1):
                blog_name = blog_stat.get('blog_name', f'Blog {i}')
                query_count = blog_stat.get('query_count', 0)
                index_num = blog_stat.get('index_number', i)
                print(f"   {Fore.WHITE}[{index_num:2d}]{Style.RESET_ALL} {blog_name:<25} {Fore.GREEN}{query_count:3d} queries{Style.RESET_ALL}")

def show_available_blog_functions(individual_tools):
    """Display all available blog query functions."""
    print(f"\n{terminal_logger.header('AVAILABLE BLOG FUNCTIONS')}")
    print(f"{Fore.CYAN}🎯 Individual Blog Query Functions:{Style.RESET_ALL}")
    
    for i, tool in enumerate(individual_tools, 1):
        tool_name = tool.metadata.name
        description = tool.metadata.description
        function_name = f"{tool_name}(query_string)"
        
        print(f"\n{Fore.WHITE}[{i:2d}]{Style.RESET_ALL} {Fore.CYAN}{function_name}{Style.RESET_ALL}")
        print(f"     {Fore.YELLOW}Description:{Style.RESET_ALL} {description}")
        print(f"     {Fore.BLUE}Usage:{Style.RESET_ALL} Call this function to search specific blog content")
    
    print(f"\n{Fore.MAGENTA}🎯 Multi-Blog Functions:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}multi_blog_query(query_string){Style.RESET_ALL}")
    print(f"     {Fore.YELLOW}Description:{Style.RESET_ALL} Search across all blogs simultaneously and synthesize results")
    print(f"     {Fore.BLUE}Usage:{Style.RESET_ALL} Best for comprehensive analysis requiring multiple sources")

def log_function_execution(func_name: str, blog_name: str, execution_time: float, success: bool, result_length: int = 0):
    """Log detailed function execution information."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    status = "SUCCESS" if success else "FAILED"
    status_color = Fore.GREEN if success else Fore.RED
    
    print(f"\n{Fore.BLUE}[{timestamp}]{Style.RESET_ALL} {terminal_logger.function_call(func_name, blog_name)}")
    print(f"          {status_color}Status:{Style.RESET_ALL} {status}")
    print(f"          {Fore.CYAN}Duration:{Style.RESET_ALL} {execution_time:.3f}s")
    if success and result_length > 0:
        print(f"          {Fore.WHITE}Result Size:{Style.RESET_ALL} {result_length:,} characters")
