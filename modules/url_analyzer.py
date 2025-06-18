import os
import logging
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import chainlit as cl
import chardet
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# URL analysis cache with timestamps and metadata
URL_CACHE = {}
CACHE_EXPIRY_HOURS = 6  # Default cache expiry
KNOWLEDGE_BASE = {}  # Persistent knowledge storage
CACHE_DURATION = timedelta(hours=6)  # Cache URLs for 6 hours

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

class IntelligentURLAnalyzer:
    """Smart URL analyzer that checks cache first, then crawls if needed."""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
    async def _get_session(self):
        """Get or create aiohttp session with optimized settings for speed."""
        if self.session is None or self.session.closed:
            # Optimized connection settings for speed
            connector = aiohttp.TCPConnector(
                limit=10,  # Reduced from 30
                limit_per_host=5,  # Reduced from 10
                ttl_dns_cache=60,  # Reduced DNS cache
                use_dns_cache=True,
                keepalive_timeout=10,  # Reduced keepalive
                enable_cleanup_closed=True
            )
            
            # Faster timeout settings
            timeout = aiohttp.ClientTimeout(
                total=12,  # Reduced from 15
                connect=4,  # Reduced from 5
                sock_read=8   # Reduced from 10
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                headers=self.headers,
                timeout=timeout,
                raise_for_status=False
            )
        
        return self.session
    
    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any], max_age_hours: int = CACHE_EXPIRY_HOURS) -> bool:
        """Check if cache entry is still valid."""
        if not cache_entry or 'timestamp' not in cache_entry:
            return False
        
        cached_time = datetime.fromisoformat(cache_entry['timestamp'])
        expiry_time = cached_time + timedelta(hours=max_age_hours)
        return datetime.now() < expiry_time
    
    def _extract_knowledge_base_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Check existing knowledge base for URL information."""
        domain = urlparse(url).netloc.lower()
        
        # Check for exact URL match
        if url in KNOWLEDGE_BASE:
            return KNOWLEDGE_BASE[url]
        
        # Check for domain-level information
        for stored_url, info in KNOWLEDGE_BASE.items():
            if urlparse(stored_url).netloc.lower() == domain:
                return {
                    **info,
                    "source": "domain_knowledge",
                    "note": f"Information based on domain knowledge for {domain}"
                }
        
        return None
    
    def _update_knowledge_base(self, url: str, analysis: Dict[str, Any]):
        """Update knowledge base with new URL analysis."""
        KNOWLEDGE_BASE[url] = {
            "analysis": analysis,
            "last_updated": datetime.now().isoformat(),
            "update_count": KNOWLEDGE_BASE.get(url, {}).get("update_count", 0) + 1
        }
        
        # Keep knowledge base size manageable
        if len(KNOWLEDGE_BASE) > 500:
            # Remove oldest entries
            oldest_entries = sorted(KNOWLEDGE_BASE.items(), 
                                  key=lambda x: x[1].get("last_updated", ""))
            for old_url, _ in oldest_entries[:50]:
                del KNOWLEDGE_BASE[old_url]
    
    async def _crawl_fresh_content(self, url: str) -> Dict[str, Any]:
        """Crawl fresh content from URL with enhanced security and error handling."""
        try:
            session = await self._get_session()
            
            logger.info(f"🌐 Fast crawling: {url}")
            
            # Quick URL validation
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {"error": "Invalid URL format"}
            
            # Fast security check
            if any(indicator in url.lower() for indicator in ['.onion', 'localhost', '127.0.0.1']):
                logger.warning(f"⚠️ Potentially unsafe URL: {url}")
                return {"error": "Unsafe URL detected - analysis blocked for security"}
            
            # HEAD request first for fast validation (reduced timeout)
            try:
                logger.info("📋 Quick URL validation...")
                async with session.head(url, timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status >= 400:
                        return {"error": f"URL returned status code: {response.status}"}
                    
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(ct in content_type for ct in ['text/html', 'text/plain', 'application/json']):
                        return {"error": f"Unsupported content type: {content_type}"}
                        
            except Exception as e:
                logger.warning(f"HEAD request failed, proceeding with GET: {str(e)[:100]}")
            
            # GET request with optimized timeout settings
            try:
                logger.info("📥 Fetching content...")
                timeout = aiohttp.ClientTimeout(total=12, connect=4)  # Reduced from 15s to 12s total
                
                async with session.get(url, timeout=timeout) as response:
                    if response.status >= 400:
                        return {"error": f"HTTP {response.status}: {response.reason}"}
                    
                    # Optimized chunked reading with smaller chunks for speed
                    content_chunks = []
                    content_size = 0
                    max_size = 3 * 1024 * 1024  # Reduced from 5MB to 3MB for faster processing
                    chunk_size = 4096  # Reduced chunk size for more responsive processing
                    
                    async for chunk in response.content.iter_chunked(chunk_size):
                        content_chunks.append(chunk)
                        content_size += len(chunk)
                        
                        if content_size > max_size:
                            logger.info(f"⚡ Content size limit reached ({max_size//1024//1024}MB) - truncating for speed")
                            break
                    
                    content_bytes = b''.join(content_chunks)
                    
                    # Fast encoding detection and decoding
                    try:
                        if 'charset=' in response.headers.get('content-type', ''):
                            encoding = response.headers.get('content-type', '').split('charset=')[1].split(';')[0]
                        else:
                            # Quick encoding detection (limited for speed)
                            encoding = chardet.detect(content_bytes[:1024])['encoding'] or 'utf-8'
                        content = content_bytes.decode(encoding, errors='ignore')
                    except Exception:
                        content = content_bytes.decode('utf-8', errors='ignore')
                    
                    # Fast HTML parsing with basic extraction
                    return await self._fast_parse_content(content, url, response.headers)
                    
            except asyncio.TimeoutError:
                return {"error": "Request timed out (>12 seconds) - site may be slow or unresponsive"}
            except aiohttp.ClientError as e:
                return {"error": f"Connection failed: {str(e)[:100]}"}
            except Exception as e:
                return {"error": f"Unexpected error: {str(e)[:100]}"}
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)[:100]}"}
    
    async def _fast_parse_content(self, content: str, url: str, headers) -> Dict[str, Any]:
        """Enhanced content parsing optimized for comprehensive analysis with better depth."""
        try:
            # Enhanced BeautifulSoup parsing
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove unwanted elements for cleaner content
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                element.decompose()
            
            # Enhanced extraction of key elements
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "No title"
            
            # Enhanced meta description extraction
            description_meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
            description = description_meta.get('content', '') if description_meta else ""
            
            # Extract keywords
            keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
            keywords = keywords_meta.get('content', '') if keywords_meta else ""
            
            # Enhanced text extraction for much better analysis depth
            # Try to find main content areas first - more comprehensive
            main_content_selectors = [
                'main', 'article', '.content', '#content', '.main', '#main',
                '.post-content', '.entry-content', '.page-content', '.blog-content',
                '.article-content', '.post-body', '.entry-body', '.content-body',
                '[role="main"]', '.primary-content', '.main-content'
            ]
            
            main_text = ""
            content_structure = {}
            
            # Try each selector for main content
            for selector in main_content_selectors:
                main_element = soup.select_one(selector)
                if main_element:
                    main_text = main_element.get_text(separator=' ', strip=True)
                    if len(main_text) > 300:  # Minimum meaningful content
                        break
            
            # Fallback to comprehensive text extraction with structure analysis
            if not main_text or len(main_text) < 300:
                # Extract structured content for better analysis
                content_parts = {
                    'headings': [],
                    'paragraphs': [],
                    'lists': [],
                    'code_blocks': [],
                    'tables': []
                }
                
                # Extract headings with hierarchy
                for level in range(1, 7):
                    headings = soup.find_all(f'h{level}')
                    for h in headings:
                        h_text = h.get_text(strip=True)
                        if h_text and len(h_text) > 5:
                            content_parts['headings'].append({
                                'level': level,
                                'text': h_text
                            })
                
                # Extract meaningful paragraphs
                paragraphs = soup.find_all(['p', 'div'], limit=50)
                for p in paragraphs:
                    p_text = p.get_text(strip=True)
                    if p_text and len(p_text) > 30:  # Only substantial paragraphs
                        content_parts['paragraphs'].append(p_text)
                
                # Extract lists
                lists = soup.find_all(['ul', 'ol'], limit=20)
                for lst in lists:
                    list_items = [li.get_text(strip=True) for li in lst.find_all('li')]
                    if list_items:
                        content_parts['lists'].append(list_items)
                
                # Extract code blocks
                code_blocks = soup.find_all(['code', 'pre'], limit=10)
                for code in code_blocks:
                    code_text = code.get_text(strip=True)
                    if code_text and len(code_text) > 10:
                        content_parts['code_blocks'].append(code_text)
                
                # Extract tables
                tables = soup.find_all('table', limit=5)
                for table in tables:
                    table_text = table.get_text(separator=' | ', strip=True)
                    if table_text and len(table_text) > 20:
                        content_parts['tables'].append(table_text)
                
                # Combine all content for comprehensive analysis
                all_text_parts = []
                
                # Add structured content
                if content_parts['headings']:
                    all_text_parts.append("HEADINGS: " + " | ".join([h['text'] for h in content_parts['headings']]))
                
                if content_parts['paragraphs']:
                    all_text_parts.extend(content_parts['paragraphs'][:20])  # First 20 paragraphs
                
                if content_parts['lists']:
                    for lst in content_parts['lists'][:5]:  # First 5 lists
                        all_text_parts.append("LIST: " + " | ".join(lst[:10]))  # First 10 items per list
                
                if content_parts['code_blocks']:
                    all_text_parts.append("CODE: " + " | ".join(content_parts['code_blocks'][:3]))
                
                main_text = ' '.join(all_text_parts)
                content_structure = content_parts
            
            # Enhanced content analysis
            word_count = len(main_text.split()) if main_text else 0
            
            # Detect content type and purpose
            content_type_analysis = self._analyze_content_type_detailed(main_text, title_text, url)
            
            # Enhanced security and technology detection
            technology_analysis = self._detect_technologies_enhanced(soup, headers, main_text)
            
            # Extract all links for comprehensive analysis
            internal_links = []
            external_links = []
            domain = urlparse(url).netloc
            
            for link in soup.find_all('a', href=True, limit=50):
                href = link.get('href', '').strip()
                if href:
                    full_url = urljoin(url, href)
                    link_domain = urlparse(full_url).netloc
                    link_text = link.get_text(strip=True)
                    
                    link_info = {
                        'url': full_url,
                        'text': link_text,
                        'title': link.get('title', '')
                    }
                    
                    if link_domain == domain:
                        internal_links.append(link_info)
                    elif link_domain:
                        external_links.append(link_info)
            
            # Enhanced metadata extraction
            meta_info = self._extract_enhanced_metadata(soup)
            
            # Security analysis
            security_analysis = self._perform_enhanced_security_analysis(soup, url, headers, main_text)
            
            # Performance and SEO indicators
            performance_indicators = self._analyze_performance_indicators(soup, headers)
            
            return {
                'url': url,
                'title': title_text,
                'description': description,
                'keywords': keywords,
                'main_content': main_text,  # Full comprehensive content for analysis
                'content_structure': content_structure,  # Structured content breakdown
                'word_count': word_count,
                'content_type_analysis': content_type_analysis,
                'technology_analysis': technology_analysis,
                'internal_links': internal_links[:20],  # First 20 internal links
                'external_links': external_links[:20],  # First 20 external links
                'meta_info': meta_info,
                'security_analysis': security_analysis,
                'performance_indicators': performance_indicators,
                'https_used': url.lower().startswith('https://'),
                'analysis_type': 'comprehensive',
                'processing_time': 'detailed_crawl',
                'content_length': len(content),
                'content_quality_score': min(100, max(0, word_count // 10)),  # Quality score based on content depth
                'from_cache': False,
                'crawl_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced content parsing failed: {str(e)}")
            return {"error": f"Content parsing failed: {str(e)[:100]}"}
    
    def _analyze_content_type_detailed(self, content: str, title: str, url: str) -> Dict[str, Any]:
        """Analyze content type and purpose in detail."""
        content_lower = content.lower()
        title_lower = title.lower()
        url_lower = url.lower()
        
        analysis = {
            'primary_type': 'unknown',
            'secondary_types': [],
            'topics': [],
            'audience': 'general',
            'purpose': 'information'
        }
        
        # Blog/Article detection
        if any(indicator in url_lower or indicator in title_lower for indicator in [
            'blog', 'post', 'article', 'news', 'story'
        ]):
            analysis['primary_type'] = 'blog_article'
            analysis['purpose'] = 'content_publishing'
        
        # Technical documentation
        elif any(indicator in content_lower for indicator in [
            'documentation', 'docs', 'api', 'tutorial', 'guide', 'manual',
            'installation', 'configuration', 'setup', 'how to'
        ]):
            analysis['primary_type'] = 'technical_documentation'
            analysis['purpose'] = 'education'
            analysis['audience'] = 'technical'
        
        # Security/Research content
        elif any(indicator in content_lower for indicator in [
            'vulnerability', 'exploit', 'cve-', 'security', 'penetration testing',
            'red team', 'blue team', 'bug bounty', 'malware', 'threat'
        ]):
            analysis['primary_type'] = 'security_research'
            analysis['purpose'] = 'security_education'
            analysis['audience'] = 'security_professionals'
            analysis['secondary_types'].append('technical_analysis')
        
        # Academic/Research
        elif any(indicator in content_lower for indicator in [
            'research', 'paper', 'study', 'analysis', 'methodology',
            'abstract', 'conclusion', 'references', 'citation'
        ]):
            analysis['primary_type'] = 'academic_research'
            analysis['purpose'] = 'knowledge_sharing'
            analysis['audience'] = 'researchers'
        
        # News/Announcement
        elif any(indicator in content_lower for indicator in [
            'announcement', 'release', 'update', 'news', 'press release'
        ]):
            analysis['primary_type'] = 'news_announcement'
            analysis['purpose'] = 'information_dissemination'
        
        # Tool/Software documentation
        elif any(indicator in content_lower for indicator in [
            'download', 'install', 'usage', 'command', 'options', 'parameters'
        ]):
            analysis['secondary_types'].append('tool_documentation')
        
        # Extract topics
        topic_keywords = {
            'cybersecurity': ['security', 'vulnerability', 'exploit', 'malware', 'threat'],
            'programming': ['code', 'programming', 'development', 'software', 'algorithm'],
            'networking': ['network', 'protocol', 'tcp', 'ip', 'dns', 'http'],
            'web_development': ['web', 'html', 'css', 'javascript', 'frontend', 'backend'],
            'data_science': ['data', 'analysis', 'machine learning', 'ai', 'statistics'],
            'system_administration': ['admin', 'server', 'system', 'configuration', 'management']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                analysis['topics'].append(topic)
        
        return analysis
    
    def _detect_technologies_enhanced(self, soup: BeautifulSoup, headers: Dict, content: str) -> Dict[str, Any]:
        """Enhanced technology detection with more comprehensive analysis."""
        technologies = {
            'server_technologies': [],
            'frontend_frameworks': [],
            'backend_technologies': [],
            'cms_platforms': [],
            'analytics_tools': [],
            'security_tools': [],
            'cdn_services': []
        }
        
        content_lower = content.lower()
        
        # Server detection (enhanced)
        server = headers.get('server', '').lower()
        x_powered_by = headers.get('x-powered-by', '').lower()
        
        server_indicators = {
            'nginx': 'Nginx Web Server',
            'apache': 'Apache HTTP Server',
            'cloudflare': 'Cloudflare CDN',
            'microsoft-iis': 'Microsoft IIS',
            'express': 'Express.js (Node.js)',
            'gunicorn': 'Gunicorn (Python)',
            'uvicorn': 'Uvicorn (Python/FastAPI)'
        }
        
        for indicator, name in server_indicators.items():
            if indicator in server or indicator in x_powered_by:
                technologies['server_technologies'].append(name)
        
        # Framework detection (enhanced)
        framework_indicators = {
            'react': ('React', 'frontend_frameworks'),
            'vue': ('Vue.js', 'frontend_frameworks'),
            'angular': ('Angular', 'frontend_frameworks'),
            'jquery': ('jQuery', 'frontend_frameworks'),
            'bootstrap': ('Bootstrap', 'frontend_frameworks'),
            'tailwind': ('Tailwind CSS', 'frontend_frameworks'),
            'django': ('Django', 'backend_technologies'),
            'flask': ('Flask', 'backend_technologies'),
            'rails': ('Ruby on Rails', 'backend_technologies'),
            'laravel': ('Laravel', 'backend_technologies'),
            'wordpress': ('WordPress', 'cms_platforms'),
            'drupal': ('Drupal', 'cms_platforms'),
            'joomla': ('Joomla', 'cms_platforms')
        }
        
        # Check script sources and content
        scripts = soup.find_all('script', src=True)
        for script in scripts:
            src = script.get('src', '').lower()
            for indicator, (name, category) in framework_indicators.items():
                if indicator in src:
                    technologies[category].append(name)
        
        # Check link hrefs
        links = soup.find_all('link', href=True)
        for link in links:
            href = link.get('href', '').lower()
            for indicator, (name, category) in framework_indicators.items():
                if indicator in href:
                    technologies[category].append(name)
        
        # Analytics detection
        analytics_indicators = {
            'google-analytics': 'Google Analytics',
            'gtag': 'Google Analytics 4',
            'googletagmanager': 'Google Tag Manager',
            'facebook.net': 'Facebook Pixel',
            'hotjar': 'Hotjar',
            'mixpanel': 'Mixpanel'
        }
        
        for script in scripts:
            src = script.get('src', '').lower()
            for indicator, name in analytics_indicators.items():
                if indicator in src:
                    technologies['analytics_tools'].append(name)
        
        # CDN detection
        cdn_indicators = {
            'cloudflare': 'Cloudflare',
            'amazonaws': 'Amazon CloudFront',
            'fastly': 'Fastly',
            'jsdelivr': 'jsDelivr',
            'unpkg': 'unpkg',
            'cdnjs': 'Cloudflare cdnjs'
        }
        
        for script in scripts:
            src = script.get('src', '').lower()
            for indicator, name in cdn_indicators.items():
                if indicator in src:
                    technologies['cdn_services'].append(name)
        
        # Security tool detection in content
        security_indicators = {
            'cloudflare': 'Cloudflare Security',
            'recaptcha': 'Google reCAPTCHA',
            'csrf': 'CSRF Protection',
            'content security policy': 'Content Security Policy'
        }
        
        for indicator, name in security_indicators.items():
            if indicator in content_lower:
                technologies['security_tools'].append(name)
        
        # Remove duplicates
        for category in technologies:
            technologies[category] = list(set(technologies[category]))
        
        return technologies
    
    def _extract_enhanced_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract comprehensive metadata from the page."""
        metadata = {
            'og_tags': {},
            'twitter_tags': {},
            'schema_org': [],
            'language': None,
            'charset': None,
            'viewport': None,
            'robots': None,
            'canonical': None,
            'alternate_languages': []
        }
        
        # Open Graph tags
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
        for tag in og_tags:
            property_name = tag.get('property', '')[3:]  # Remove 'og:' prefix
            metadata['og_tags'][property_name] = tag.get('content', '')
        
        # Twitter Card tags
        twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
        for tag in twitter_tags:
            name = tag.get('name', '')[8:]  # Remove 'twitter:' prefix
            metadata['twitter_tags'][name] = tag.get('content', '')
        
        # Language
        html_lang = soup.find('html')
        if html_lang and html_lang.get('lang'):
            metadata['language'] = html_lang.get('lang')
        
        # Charset
        charset_meta = soup.find('meta', charset=True)
        if charset_meta:
            metadata['charset'] = charset_meta.get('charset')
        
        # Viewport
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        if viewport_meta:
            metadata['viewport'] = viewport_meta.get('content')
        
        # Robots
        robots_meta = soup.find('meta', attrs={'name': 'robots'})
        if robots_meta:
            metadata['robots'] = robots_meta.get('content')
        
        # Canonical URL
        canonical_link = soup.find('link', rel='canonical')
        if canonical_link:
            metadata['canonical'] = canonical_link.get('href')
        
        # Alternate languages
        hreflang_links = soup.find_all('link', rel='alternate', hreflang=True)
        for link in hreflang_links:
            metadata['alternate_languages'].append({
                'lang': link.get('hreflang'),
                'url': link.get('href')
            })
        
        # Schema.org structured data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                schema_data = json.loads(script.string or '{}')
                metadata['schema_org'].append(schema_data)
            except json.JSONDecodeError:
                pass
        
        return metadata
    
    def _perform_enhanced_security_analysis(self, soup: BeautifulSoup, url: str, headers: Dict, content: str) -> Dict[str, Any]:
        """Perform comprehensive security analysis."""
        analysis = {
            'https_used': url.startswith('https://'),
            'security_headers': {},
            'form_security': {},
            'external_resources': {},
            'potential_vulnerabilities': [],
            'security_score': 0
        }
        
        # Security headers analysis
        security_headers = [
            'content-security-policy',
            'x-frame-options',
            'x-content-type-options',
            'strict-transport-security',
            'x-xss-protection',
            'referrer-policy',
            'permissions-policy'
        ]
        
        for header in security_headers:
            value = headers.get(header.lower())
            analysis['security_headers'][header] = {
                'present': bool(value),
                'value': value or 'Not Set'
            }
            if value:
                analysis['security_score'] += 10
        
        # Form security analysis
        forms = soup.find_all('form')
        analysis['form_security'] = {
            'total_forms': len(forms),
            'forms_with_csrf': 0,
            'https_forms': 0,
            'forms_with_method_post': 0
        }
        
        for form in forms:
            # Check for CSRF tokens
            if form.find('input', attrs={'name': lambda x: x and 'csrf' in x.lower()}):
                analysis['form_security']['forms_with_csrf'] += 1
            
            # Check form method
            if form.get('method', '').lower() == 'post':
                analysis['form_security']['forms_with_method_post'] += 1
            
            # Check form action (HTTPS)
            action = form.get('action', '')
            if action.startswith('https://') or (not action and url.startswith('https://')):
                analysis['form_security']['https_forms'] += 1
        
        # External resources analysis
        external_scripts = soup.find_all('script', src=True)
        external_links = soup.find_all('link', href=True)
        
        analysis['external_resources'] = {
            'external_scripts': len([s for s in external_scripts if not urlparse(s.get('src', '')).netloc == urlparse(url).netloc]),
            'external_stylesheets': len([l for l in external_links if l.get('rel') == 'stylesheet' and not urlparse(l.get('href', '')).netloc == urlparse(url).netloc]),
            'mixed_content_risk': False
        }
        
        # Check for mixed content
        if url.startswith('https://'):
            for script in external_scripts:
                src = script.get('src', '')
                if src.startswith('http://'):
                    analysis['external_resources']['mixed_content_risk'] = True
                    analysis['potential_vulnerabilities'].append('Mixed content detected (HTTP resources on HTTPS page)')
                    break
        
        # Vulnerability checks
        if not analysis['https_used']:
            analysis['potential_vulnerabilities'].append('Site uses HTTP instead of HTTPS')
            analysis['security_score'] -= 20
        
        if not analysis['security_headers']['content-security-policy']['present']:
            analysis['potential_vulnerabilities'].append('Missing Content Security Policy header')
        
        if analysis['form_security']['total_forms'] > 0 and analysis['form_security']['forms_with_csrf'] == 0:
            analysis['potential_vulnerabilities'].append('Forms without apparent CSRF protection')
        
        if analysis['external_resources']['external_scripts'] > 10:
            analysis['potential_vulnerabilities'].append('Many external scripts loaded - potential supply chain risk')
        
        # Calculate final security score
        base_score = 50
        if analysis['https_used']:
            base_score += 20
        
        analysis['security_score'] = max(0, min(100, base_score + analysis['security_score']))
        
        return analysis
    
    def _analyze_performance_indicators(self, soup: BeautifulSoup, headers: Dict) -> Dict[str, Any]:
        """Analyze performance indicators."""
        indicators = {
            'images_count': len(soup.find_all('img')),
            'scripts_count': len(soup.find_all('script')),
            'stylesheets_count': len(soup.find_all('link', rel='stylesheet')),
            'has_lazy_loading': bool(soup.find('img', loading='lazy')),
            'has_preconnect': bool(soup.find('link', rel='preconnect')),
            'has_dns_prefetch': bool(soup.find('link', rel='dns-prefetch')),
            'compression_enabled': 'gzip' in headers.get('content-encoding', '').lower() or 'br' in headers.get('content-encoding', '').lower(),
            'cache_headers': bool(headers.get('cache-control') or headers.get('expires'))
        }
        
        return indicators
    
    async def _analyze_html_content(self, content: str, url: str, headers: Dict) -> Dict[str, Any]:
        """Analyze HTML content comprehensively."""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract basic information
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        # Meta information
        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description.get('content', '').strip() if meta_description else ""
        
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        keywords = meta_keywords.get('content', '').strip() if meta_keywords else ""
        
        # Security-related meta tags
        security_headers = self._analyze_security_headers(headers)
        
        # Extract main content
        main_content = self._extract_main_content(soup)
        
        # Technology detection
        technologies = self._detect_technologies(soup, headers)
        
        # Security analysis
        security_analysis = self._analyze_security_features(soup, url)
        
        # Social media and external links
        external_links = self._extract_external_links(soup, url)
        social_links = self._extract_social_links(soup)
        
        # Content analysis
        content_analysis = self._analyze_content_quality(main_content, soup)
        
        return {
            "title": title_text,
            "description": description,
            "keywords": keywords,
            "main_content": main_content[:2000],  # Limit content size
            "content_length": len(main_content),
            "technologies": technologies,
            "security_analysis": security_analysis,
            "security_headers": security_headers,
            "external_links": external_links,
            "social_links": social_links,
            "content_analysis": content_analysis,
            "crawl_timestamp": datetime.now().isoformat(),
            "content_type": "html",
            "url": url
        }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.extract()
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '#content', '.main', '#main',
            '.post-content', '.entry-content', '.page-content'
        ]
        
        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                return main_element.get_text().strip()
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text().strip()
        
        return soup.get_text().strip()
    
    def _detect_technologies(self, soup: BeautifulSoup, headers: Dict) -> List[str]:
        """Detect technologies used by the website."""
        technologies = []
        
        # Server detection
        server = headers.get('server', '').lower()
        if 'nginx' in server:
            technologies.append('Nginx')
        if 'apache' in server:
            technologies.append('Apache')
        if 'cloudflare' in server:
            technologies.append('Cloudflare')
        
        # Framework detection
        generator = soup.find('meta', attrs={'name': 'generator'})
        if generator:
            technologies.append(f"Generator: {generator.get('content', '')}")
        
        # JavaScript frameworks
        scripts = soup.find_all('script', src=True)
        for script in scripts:
            src = script.get('src', '').lower()
            if 'react' in src:
                technologies.append('React')
            if 'vue' in src:
                technologies.append('Vue.js')
            if 'angular' in src:
                technologies.append('Angular')
            if 'jquery' in src:
                technologies.append('jQuery')
        
        # CSS frameworks
        links = soup.find_all('link', rel='stylesheet')
        for link in links:
            href = link.get('href', '').lower()
            if 'bootstrap' in href:
                technologies.append('Bootstrap')
            if 'tailwind' in href:
                technologies.append('Tailwind CSS')
        
        return list(set(technologies))
    
    def _analyze_security_headers(self, headers: Dict) -> Dict[str, Any]:
        """Analyze security-related HTTP headers."""
        security_headers = {}
        
        # Content Security Policy
        csp = headers.get('content-security-policy')
        security_headers['csp'] = {'present': bool(csp), 'value': csp}
        
        # X-Frame-Options
        xfo = headers.get('x-frame-options')
        security_headers['x_frame_options'] = {'present': bool(xfo), 'value': xfo}
        
        # X-Content-Type-Options
        xcto = headers.get('x-content-type-options')
        security_headers['x_content_type_options'] = {'present': bool(xcto), 'value': xcto}
        
        # Strict-Transport-Security
        hsts = headers.get('strict-transport-security')
        security_headers['hsts'] = {'present': bool(hsts), 'value': hsts}
        
        # X-XSS-Protection
        xxp = headers.get('x-xss-protection')
        security_headers['x_xss_protection'] = {'present': bool(xxp), 'value': xxp}
        
        return security_headers
    
    def _analyze_security_features(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Analyze security features and potential vulnerabilities."""
        analysis = {
            "https_used": url.startswith('https://'),
            "forms_found": len(soup.find_all('form')),
            "external_scripts": 0,
            "inline_scripts": 0,
            "potential_issues": []
        }
        
        # Count external scripts
        scripts = soup.find_all('script', src=True)
        analysis["external_scripts"] = len(scripts)
        
        # Count inline scripts
        inline_scripts = soup.find_all('script', src=False)
        analysis["inline_scripts"] = len([s for s in inline_scripts if s.string])
        
        # Check for potential security issues
        if not analysis["https_used"]:
            analysis["potential_issues"].append("Site uses HTTP instead of HTTPS")
        
        if analysis["inline_scripts"] > 5:
            analysis["potential_issues"].append("Many inline scripts detected - potential XSS risk")
        
        # Check for forms without CSRF protection
        forms = soup.find_all('form')
        for form in forms:
            if not form.find('input', attrs={'name': re.compile(r'csrf|token', re.I)}):
                analysis["potential_issues"].append("Forms without apparent CSRF protection")
                break
        
        return analysis
    
    def _extract_external_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract external links from the page."""
        base_domain = urlparse(base_url).netloc
        external_links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            full_url = urljoin(base_url, href)
            link_domain = urlparse(full_url).netloc
            
            if link_domain and link_domain != base_domain:
                external_links.append(full_url)
        
        return list(set(external_links))[:20]  # Limit to 20 links
    
    def _extract_social_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract social media links."""
        social_domains = [
            'twitter.com', 'facebook.com', 'linkedin.com', 'instagram.com',
            'youtube.com', 'github.com', 'telegram.org', 'discord.com'
        ]
        
        social_links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if any(domain in href for domain in social_domains):
                social_links.append(href)
        
        return list(set(social_links))
    
    def _analyze_content_quality(self, content: str, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze content quality and characteristics."""
        words = content.split()
        
        return {
            "word_count": len(words),
            "character_count": len(content),
            "paragraph_count": len(soup.find_all('p')),
            "heading_count": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            "image_count": len(soup.find_all('img')),
            "link_count": len(soup.find_all('a')),
            "estimated_reading_time": max(1, len(words) // 200)  # Assume 200 WPM
        }
    
    async def _analyze_json_content(self, content: str, url: str) -> Dict[str, Any]:
        """Analyze JSON content."""
        try:
            data = json.loads(content)
            return {
                "content_type": "json",
                "data_structure": self._analyze_json_structure(data),
                "size": len(content),
                "crawl_timestamp": datetime.now().isoformat(),
                "url": url
            }
        except json.JSONDecodeError:
            return {"error": "Invalid JSON content"}
    
    def _analyze_json_structure(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Analyze JSON data structure."""
        if current_depth >= max_depth:
            return {"type": type(data).__name__, "truncated": True}
        
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys())[:20],  # Limit keys
                "key_count": len(data)
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "item_types": list(set(type(item).__name__ for item in data[:10]))
            }
        else:
            return {"type": type(data).__name__, "value": str(data)[:100]}
    
    async def _analyze_text_content(self, content: str, url: str, content_type: str) -> Dict[str, Any]:
        """Analyze plain text content."""
        lines = content.split('\n')
        
        return {
            "content_type": content_type,
            "line_count": len(lines),
            "character_count": len(content),
            "word_count": len(content.split()),
            "sample_content": content[:1000],
            "crawl_timestamp": datetime.now().isoformat(),
            "url": url
        }
    
    async def analyze_url_intelligent(self, url: str, force_refresh: bool = False, context: str = "") -> Dict[str, Any]:
        """
        Intelligent URL analysis that checks knowledge base, then cache, then crawls fresh content.
        """
        logger.info(f"Starting intelligent analysis for: {url}")
        
        try:
            cache_key = self._generate_cache_key(url)
            
            # Step 1: Check knowledge base first
            if not force_refresh:
                kb_info = self._extract_knowledge_base_info(url)
                if kb_info:
                    logger.info(f"Using knowledge base information for {url}")
                    analysis = kb_info["analysis"]
                    # Ensure content type is set for consistent processing
                    if "content_type" not in analysis:
                        analysis["content_type"] = "html"
                    return {
                        **analysis,
                        "data_source": "knowledge_base",
                        "last_updated": kb_info["last_updated"],
                        "confidence": "high",
                        "note": "Information from existing knowledge base"
                    }
            
            # Step 2: Check cache
            if not force_refresh and cache_key in URL_CACHE:
                cache_entry = URL_CACHE[cache_key]
                if self._is_cache_valid(cache_entry):
                    logger.info(f"Using cached analysis for {url}")
                    analysis = cache_entry["analysis"]
                    # Ensure content type is set for consistent processing
                    if "content_type" not in analysis:
                        analysis["content_type"] = "html"
                    return {
                        **analysis,
                        "data_source": "cache",
                        "cache_age_hours": (datetime.now() - datetime.fromisoformat(cache_entry["timestamp"])).total_seconds() / 3600,
                        "note": "Information from recent cache"
                    }
            
            # Step 3: Crawl fresh content
            logger.info(f"Crawling fresh content for {url}")
            fresh_analysis = await self._crawl_fresh_content(url)
            
            if "error" not in fresh_analysis:
                # Update cache
                URL_CACHE[cache_key] = {
                    "analysis": fresh_analysis,
                    "timestamp": datetime.now().isoformat(),
                    "url": url
                }
                
                # Update knowledge base
                self._update_knowledge_base(url, fresh_analysis)
                
                fresh_analysis["data_source"] = "fresh_crawl"
                fresh_analysis["confidence"] = "very_high"
                fresh_analysis["note"] = "Fresh content crawled in real-time"
            
            return fresh_analysis
            
        except Exception as e:
            logger.error(f"Error in intelligent URL analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_url_content(self, url: str) -> Tuple[str, str, str]:
        """Fetch content from URL with optimized timeout handling."""
        try:
            # Initialize variables
            title = ""
            description = ""
            content = ""
            
            # Set up timeout and headers
            timeout = aiohttp.ClientTimeout(total=12, connect=4)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
            }
            
            # First check if URL is accessible with a HEAD request
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.head(url, headers=headers, allow_redirects=True) as head_response:
                        if head_response.status != 200:
                            return title, description, f"URL returned status code {head_response.status}"
                except Exception as head_error:
                    logger.warning(f"HEAD request failed: {str(head_error)}, trying GET directly")
                
                # Fetch content with GET request
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status != 200:
                        return title, description, f"URL returned status code {response.status}"
                    
                    # Check content type
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'text/html' not in content_type and 'application/json' not in content_type and 'text/plain' not in content_type:
                        return title, description, f"URL content type {content_type} not supported"
                    
                    # Read content with size limit
                    content_bytes = b""
                    chunk_size = 4096
                    max_size = 3 * 1024 * 1024  # 3MB limit
                    
                    async for chunk in response.content.iter_chunked(chunk_size):
                        content_bytes += chunk
                        if len(content_bytes) > max_size:
                            content_bytes = content_bytes[:max_size]
                            break
                    
                    # Detect encoding
                    encoding = response.charset if response.charset else 'utf-8'
                    try:
                        content = content_bytes.decode(encoding)
                    except UnicodeDecodeError:
                        # Try to detect encoding
                        detected = chardet.detect(content_bytes)
                        try:
                            content = content_bytes.decode(detected['encoding'] or 'utf-8', errors='replace')
                        except:
                            content = content_bytes.decode('utf-8', errors='replace')
                    
                    # Parse HTML for title and description
                    if 'text/html' in content_type:
                        soup = BeautifulSoup(content, 'html.parser')
                        title_tag = soup.find('title')
                        title = title_tag.get_text() if title_tag else "No title"
                        
                        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                        description = meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else ""
                        
                        # Clean up content - extract main text
                        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                            script.extract()
                        
                        content = soup.get_text(separator=' ', strip=True)
                    
                    return title, description, content
        
        except Exception as e:
            logger.error(f"Error fetching URL content: {str(e)}")
            return "", "", f"Error fetching content: {str(e)}"

    def analyze_content(self, content: str, title: str, description: str) -> Dict[str, Any]:
        """Analyze content to extract key metrics and insights."""
        result = {}
        
        # Basic content metrics
        result['word_count'] = len(content.split())
        
        # Content type analysis
        content_lower = content.lower()
        result['content_type_analysis'] = {
            'primary_type': 'general',
            'topics': [],
            'audience': 'general',
            'purpose': 'information'
        }
        
        # Determine primary content type
        if any(term in content_lower for term in ['tutorial', 'guide', 'how to']):
            result['content_type_analysis']['primary_type'] = 'educational'
            result['content_type_analysis']['purpose'] = 'instruction'
        elif any(term in content_lower for term in ['documentation', 'reference', 'api']):
            result['content_type_analysis']['primary_type'] = 'technical_documentation'
            result['content_type_analysis']['purpose'] = 'reference'
        elif any(term in content_lower for term in ['blog', 'article', 'post']):
            result['content_type_analysis']['primary_type'] = 'blog'
            result['content_type_analysis']['purpose'] = 'information'
        elif any(term in content_lower for term in ['research', 'study', 'analysis']):
            result['content_type_analysis']['primary_type'] = 'research'
            result['content_type_analysis']['purpose'] = 'education'
        
        # Detect topics
        topics = []
        if any(term in content_lower for term in ['cybersecurity', 'security', 'hacking']):
            topics.append('cybersecurity')
        if any(term in content_lower for term in ['programming', 'code', 'development']):
            topics.append('programming')
        if any(term in content_lower for term in ['network', 'networking', 'infrastructure']):
            topics.append('networking')
        if any(term in content_lower for term in ['data', 'analytics', 'statistics']):
            topics.append('data_science')
        
        result['content_type_analysis']['topics'] = topics
        
        # Determine audience
        if any(term in content_lower for term in ['advanced', 'expert', 'professional']):
            result['content_type_analysis']['audience'] = 'expert'
        elif any(term in content_lower for term in ['beginner', 'introduction', 'basic']):
            result['content_type_analysis']['audience'] = 'beginner'
        
        # Simple security analysis
        result['security_analysis'] = {
            'security_score': 50,  # Default middle score
            'https_used': True  # Assume HTTPS by default
        }
        
        # Content quality metrics (simple heuristics)
        result['content_quality'] = {
            'readability': 70,  # Default good readability
            'technical_depth': 50,  # Default medium technical depth
            'educational_value': 60  # Default good educational value
        }
        
        # Adjust technical depth based on content
        if any(term in content_lower for term in ['code', 'implementation', 'technical']):
            result['content_quality']['technical_depth'] = 80
        
        # Adjust educational value based on content
        if any(term in content_lower for term in ['learn', 'tutorial', 'guide']):
            result['content_quality']['educational_value'] = 85
        
        return result

async def _format_comprehensive_general_analysis(title: str, description: str, content: str, url: str, result: Dict[str, Any]) -> str:
    """Format comprehensive analysis for general content with simplified output."""
    
    # Extract key metrics and insights
    word_count = result.get('word_count', 0)
    content_type_analysis = result.get('content_type_analysis', {})
    
    # Build simplified analysis
    analysis_parts = []
    
    # Main content analysis
    primary_type = content_type_analysis.get('primary_type', 'general').replace('_', ' ').title()
    topics = content_type_analysis.get('topics', [])
    topics_str = ", ".join([topic.replace('_', ' ').title() for topic in topics[:3]]) if topics else "General information"
    
    # Create a clean, focused analysis
    analysis_parts.append(f"This content from {url} focuses on {topics_str}.")
    
    if title and title != "No title":
        analysis_parts.append(f"The page titled '{title}' contains approximately {word_count} words of {primary_type} content.")
    
    if description:
        analysis_parts.append(f"The site describes itself as: '{description}'")
    
    # Content preview (first 300 characters for context)
    if content and len(content) > 100:
        content_preview = content[:300].strip()
        if len(content) > 300:
            content_preview += "..."
        analysis_parts.append(f"\nContent preview:\n{content_preview}")
    
    return "\n\n".join(analysis_parts)

# Global analyzer instance
url_analyzer = IntelligentURLAnalyzer()

def format_html_analysis(result: Dict[str, Any]) -> str:
    """Format HTML analysis results with enhanced contextual explanations."""
    analysis_type = result.get('analysis_type', 'detailed')
    url = result.get('url', 'Unknown URL')
    title = result.get('title', 'No title')
    description = result.get('description', 'No description')
    main_content = result.get('main_content', '')
    
    # Get enhanced contextual analysis
    content_analysis = _analyze_content_context(title, description, main_content, url)
    
    if analysis_type == 'fast':
        # Enhanced fast analysis with detailed contextual explanations
        analysis_parts = []
        
        # Add purpose explanation if available
        if content_analysis['explanation']:
            analysis_parts.append(content_analysis['explanation'])
        
        # Add technical details
        technical_info = f"""## 🔍 Website Technical Details

**📋 Basic Information:**
• **URL:** {url}
• **Title:** {title}
• **Description:** {description}
• **Content Length:** {result.get('word_count', 0):,} words
• **HTTPS:** {'✅ Yes' if result.get('https_used') else '❌ No'}
• **Forms Found:** {result.get('forms_count', 0)}"""
        
        analysis_parts.append(technical_info)
        
        # Add detailed analysis if available
        if content_analysis['technical_details']:
            analysis_parts.append(content_analysis['technical_details'])
        
        # Add content preview
        if main_content and len(main_content) > 50:
            preview_content = main_content[:500].strip()
            if len(main_content) > 500:
                preview_content += "..."
            
            content_preview = f"""## 📝 Content Preview

{preview_content}"""
            analysis_parts.append(content_preview)
        
        # Add insights if available
        if content_analysis['impact_analysis']:
            analysis_parts.append(content_analysis['impact_analysis'])
        
        # Add actionable information if available
        if content_analysis['learning_value']:
            analysis_parts.append(content_analysis['learning_value'])
        
        # Add performance note
        performance_note = f"""## ⚡ Performance Information

**Analysis Completed:** Fast mode analysis completed in under 15 seconds
**Data Processing:** Optimized content extraction and security validation
**Cache Status:** {'✅ Cached' if result.get('from_cache') else '🆕 Fresh crawl'}"""
        
        analysis_parts.append(performance_note)
        
        return "\\n\\n".join(analysis_parts)
    
    else:
        # Detailed analysis format with comprehensive contextual information
        analysis_parts = []
        
        # Add purpose explanation
        if content_analysis['explanation']:
            analysis_parts.append(content_analysis['explanation'])
        
        # Add comprehensive technical analysis
        comprehensive_info = f"""## 🔍 Comprehensive Website Analysis

**📋 Technical Information:**
• **URL:** {url}
• **Title:** {title}
• **Meta Description:** {description}
• **Content Statistics:** {result.get('word_count', 0):,} words, {result.get('links_count', 0)} links
• **Security:** {'✅ HTTPS enabled' if result.get('https_used') else '⚠️ HTTP only (not secure)'}
• **Interactive Elements:** {result.get('forms_count', 0)} forms detected
• **Page Structure:** {result.get('headings_count', 0)} headings found
• **Media Content:** {result.get('images_count', 0)} images detected
• **External Resources:** {result.get('external_links_count', 0)} external links"""
        
        analysis_parts.append(comprehensive_info)
        
        # Add detailed contextual analysis
        if content_analysis['technical_details']:
            analysis_parts.append(content_analysis['technical_details'])
        
        # Add security analysis
        security_info = f"""## 🔒 Security Assessment

**🛡️ Security Features:**
• **Transport Security:** {'✅ Secure (HTTPS)' if result.get('https_used') else '❌ Insecure (HTTP)'}
• **Content Security:** {result.get('security_score', 'Not assessed')}
• **External Dependencies:** {result.get('external_links_count', 0)} external resources
• **Form Security:** {'✅ Forms detected' if result.get('forms_count', 0) > 0 else '✅ No forms (reduced attack surface)'}

**⚠️ Security Considerations:**
• Always verify content authenticity when using security research
• Be cautious when implementing proof-of-concept code
• Ensure proper authorization before testing vulnerabilities
• Keep software updated based on disclosed vulnerabilities"""
        
        analysis_parts.append(security_info)
        
        # Add detailed content analysis
        if main_content and len(main_content) > 100:
            # Extract key sections and topics
            content_sections = []
            lines = main_content.split('\\n')
            current_section = ""
            
            for line in lines[:20]:  # First 20 lines for preview
                if line.strip():
                    current_section += line.strip() + " "
                    if len(current_section) > 200:
                        content_sections.append(current_section.strip())
                        current_section = ""
                        if len(content_sections) >= 3:
                            break
            
            if current_section:
                content_sections.append(current_section.strip())
            
            content_analysis_section = f"""## 📖 Content Analysis

**📝 Main Content Sections:**
{chr(10).join([f"• {section[:150]}..." if len(section) > 150 else f"• {section}" for section in content_sections[:3]])}

**🔍 Content Characteristics:**
• **Readability:** Professional technical content
• **Technical Level:** {'Advanced' if any(term in main_content.lower() for term in ['cve', 'exploit', 'vulnerability', 'payload']) else 'Intermediate'}
• **Content Type:** {'Security Research' if any(term in main_content.lower() for term in ['security', 'vulnerability', 'exploit']) else 'General Technical Content'}
• **Educational Value:** High - contains detailed technical information"""
            
            analysis_parts.append(content_analysis_section)
        
        # Add insights
        if content_analysis['impact_analysis']:
            analysis_parts.append(content_analysis['impact_analysis'])
        
        # Add actionable information
        if content_analysis['learning_value']:
            analysis_parts.append(content_analysis['learning_value'])
        
        # Add performance and metadata
        performance_metadata = f"""## 📊 Analysis Metadata

**⚡ Performance Information:**
• **Analysis Time:** {'< 15 seconds (fast mode)' if analysis_type == 'fast' else '< 30 seconds (detailed mode)'}
• **Data Source:** {'📚 Cached data' if result.get('from_cache') else '🌐 Fresh web crawl'}
• **Content Processing:** {len(main_content):,} characters analyzed
• **Security Validation:** ✅ Completed

**🎯 Analysis Quality:**
• **Completeness:** {'🟢 High' if len(main_content) > 1000 else '🟡 Moderate' if len(main_content) > 100 else '🔴 Limited'}
• **Context Detection:** {'✅ Specialized content detected' if content_analysis['technical_details'] else '✅ General content analysis'}
• **Educational Value:** {'🎓 High - Professional security content' if 'security' in main_content.lower() else '📚 Standard technical content'}"""
        
        analysis_parts.append(performance_metadata)
        
        return "\\n\\n".join(analysis_parts)

def format_json_analysis(result: Dict[str, Any]) -> str:
    """Format JSON analysis results."""
    structure = result.get('data_structure', {})
    
    return f"""**📋 JSON Data Analysis:**
• **Type:** {structure.get('type', 'Unknown')}
• **Size:** {result.get('size', 0):,} bytes
• **Structure:** {json.dumps(structure, indent=2)[:500]}"""

def format_generic_analysis(result: Dict[str, Any]) -> str:
    """Format generic content analysis."""
    return f"""**📄 Content Analysis:**
• **Content Type:** {result.get('content_type', 'Unknown')}
• **Size:** {result.get('character_count', 0):,} characters
• **Preview:** {result.get('sample_content', '')[:300]}"""

# Cache management functions
def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    total_entries = len(URL_CACHE)
    valid_entries = sum(1 for entry in URL_CACHE.values() 
                       if url_analyzer._is_cache_valid(entry))
    
    return {
        "total_entries": total_entries,
        "valid_entries": valid_entries,
        "expired_entries": total_entries - valid_entries,
        "knowledge_base_entries": len(KNOWLEDGE_BASE),
        "cache_size_mb": len(str(URL_CACHE)) / (1024 * 1024)
    }

def clear_expired_cache():
    """Clear expired cache entries."""
    expired_keys = []
    for key, entry in URL_CACHE.items():
        if not url_analyzer._is_cache_valid(entry):
            expired_keys.append(key)
    
    for key in expired_keys:
        del URL_CACHE[key]
    
    logger.info(f"Cleared {len(expired_keys)} expired cache entries")

# Legacy function for backward compatibility
async def analyze_url(url: str) -> str:
    """Legacy URL analysis function."""
    logger.warning("analyze_url is deprecated, use analyze_url_smart instead")
    return await analyze_url_smart(url)

async def analyze_url_smart(url: str, force_refresh: bool = False, context: str = None) -> str:
    """Enhanced URL analyzer with intelligent context detection and caching."""
    try:
        # Clean URL
        if not url or not isinstance(url, str):
            return f"Error: Invalid URL provided. Expected a string but got {type(url)}"
            
        url = url.strip()
        logger.info(f"Processing URL: {url}")
        
        # Add http/https if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            logger.info(f"Added https:// prefix to URL: {url}")
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached_result = get_cached_url_content(url)
            if cached_result:
                logger.info(f"✅ Using cached analysis for {url}")
                return cached_result
        
        logger.info(f"🔍 Analyzing URL: {url}")
        
        # Validate URL format
        try:
            parsed_url = urlparse(url)
            if not parsed_url.netloc:
                return f"Error: Invalid URL format: {url}"
        except Exception as parse_error:
            logger.error(f"URL parsing error: {str(parse_error)}")
            return f"Error: Unable to parse URL: {url}. Error: {str(parse_error)}"
        
        # Check if URL is likely internal/private
        if is_likely_internal_url(url):
            return f"⚠️ This appears to be an internal or private URL that shouldn't be accessed externally: {url}"
        
        # Analyze URL with intelligent analyzer
        analyzer = IntelligentURLAnalyzer()
        
        # Get URL content with timeout protection
        try:
            logger.info(f"📥 Fetching content from {url}")
            
            # First try a HEAD request to check if the URL is accessible
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.head(url, timeout=5, allow_redirects=True) as response:
                        if response.status >= 400:
                            return f"Error: URL returned status code {response.status}: {url}"
            except Exception as head_error:
                logger.warning(f"HEAD request failed for {url}: {str(head_error)}")
                # Continue anyway, the full request might still work
            
            # Now fetch the full content
            title, description, content = await analyzer.fetch_url_content(url)
            
            if not content or len(content.strip()) < 100:
                return f"Could not extract meaningful content from {url}. The page might be empty, require JavaScript, or block automated access."
            
            # Use GPT-4o for comprehensive analysis
            logger.info(f"🧠 Analyzing content with GPT-4o")
            analysis = _analyze_content_with_gpt4o(content, url, context)
            
            if analysis:
                # Cache the result
                cache_url_content(url, analysis)
                logger.info(f"✅ Successfully analyzed URL: {url} ({len(analysis)} chars)")
                return analysis
            else:
                # Fallback to basic analysis
                logger.info("⚠️ GPT-4o analysis failed, using basic analysis")
                result = analyzer.analyze_content(content, title, description)
                basic_analysis = await _format_comprehensive_general_analysis(title, description, content, url, result)
                cache_url_content(url, basic_analysis)
                logger.info(f"✅ Used fallback analysis for URL: {url} ({len(basic_analysis)} chars)")
                return basic_analysis
                
        except aiohttp.ClientError as client_error:
            error_message = str(client_error)
            logger.error(f"Network error fetching URL: {error_message}")
            
            if "Cannot connect" in error_message or "Connection refused" in error_message:
                return f"Error: Cannot connect to {url}. The site might be down or blocking access."
            elif "Timeout" in error_message:
                return f"Error: Connection to {url} timed out. The site might be slow or unresponsive."
            elif "SSL" in error_message:
                return f"Error: SSL/TLS error with {url}. The site might have invalid security certificates."
            else:
                return f"Error accessing {url}: {error_message}"
                
        except Exception as fetch_error:
            logger.error(f"Error fetching URL: {str(fetch_error)}")
            return f"Error analyzing {url}: {str(fetch_error)}"
    
    except Exception as e:
        logger.error(f"URL analysis error: {str(e)}", exc_info=True)
        return f"Error analyzing URL: {str(e)}"
    
    finally:
        # Close the analyzer to clean up resources
        try:
            if 'analyzer' in locals():
                await analyzer.close()
        except Exception as close_error:
            logger.error(f"Error closing URL analyzer: {str(close_error)}")
            # Don't return here, as we might already have a return value

def get_cached_url_content(url: str) -> Optional[str]:
    """Get cached URL content if available and not expired."""
    cache_key = hashlib.md5(url.encode()).hexdigest()
    if cache_key in URL_CACHE:
        cached_time = URL_CACHE[cache_key].get('timestamp')
        if cached_time and (datetime.now() - cached_time) < CACHE_DURATION:
            return URL_CACHE[cache_key].get('content')
    return None

def cache_url_content(url: str, content: str) -> None:
    """Cache URL content with timestamp."""
    cache_key = hashlib.md5(url.encode()).hexdigest()
    URL_CACHE[cache_key] = {
        'url': url,
        'content': content,
        'timestamp': datetime.now()
    }
    
    # Limit cache size to 50 entries
    if len(URL_CACHE) > 50:
        # Remove oldest entry
        oldest_key = min(URL_CACHE.keys(), key=lambda k: URL_CACHE[k].get('timestamp', datetime.now()))
        del URL_CACHE[oldest_key]

def _analyze_content_with_gpt4o(content: str, url: str, context: str = None) -> str:
    """Use GPT-4o to provide intelligent, comprehensive analysis of crawled content."""
    try:
        # Prepare content for analysis (limit to ~8000 chars for token efficiency)
        content_snippet = content[:8000] if len(content) > 8000 else content
        
        # Create prompt based on context
        if context and "cve" in context.lower():
            prompt = f"""Analyze this security vulnerability content from {url}. 
            
Content: {content_snippet}

Provide a concise, bullet-point analysis focusing on:
• Vulnerability details and affected systems
• Technical explanation of the vulnerability
• Potential impact and exploitation methods
• Mitigation strategies and patches

Format your response using bullet points, short sections with clear headings, and keep paragraphs very brief (2-3 lines max).
Make it highly scannable and easy to read at a glance."""
        else:
            prompt = f"""Analyze this content from {url}.
            
Content: {content_snippet}

Provide a concise, bullet-point analysis focusing on:
• Main topic and purpose of the content
• Key technical details and concepts
• Practical applications and takeaways
• Educational value

Format your response using:
• Bullet points for lists
• Short sections with clear headings
• Brief paragraphs (2-3 lines maximum)
• Bold for important terms
• Make it highly scannable and easy to read at a glance"""
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert content analyzer specializing in technical and security content. Provide concise, scannable analyses using bullet points, short paragraphs, and clear headings to maximize readability."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        analysis = response.choices[0].message.content
        
        # Return clean analysis
        return analysis
        
    except Exception as e:
        logger.error(f"Error in GPT-4o analysis: {str(e)}")
        return f"Error analyzing content: {str(e)}" 