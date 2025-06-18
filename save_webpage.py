import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from urllib.parse import urlparse, urljoin
import shutil
import os
import logging
import time
import socket
from typing import List, Optional

# Configure logging (optimized - no log file, console only for better performance)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustWebScraper:
    def __init__(self, headless: bool = True, timeout: int = 20):  # Reduced timeout for faster processing
        self.headless = headless
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def setup_webdriver(self) -> Optional[webdriver.Chrome]:
        """Setup Chrome webdriver with optimized performance settings."""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            
            # Performance optimizations
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")  # Skip images for faster loading
            chrome_options.add_argument("--disable-javascript")  # Skip JS for faster loading
            chrome_options.add_argument("--window-size=1280,720")  # Smaller window for faster rendering
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup webdriver: {e}")
            return None

    def validate_url(self, url: str) -> bool:
        """Validate if URL is reachable with faster timeout."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                logger.warning(f"Invalid URL format: {url}")
                return False
            
            # Quick DNS resolution test
            socket.gethostbyname(parsed.netloc)
            
            # Quick HTTP HEAD request with reduced timeout
            response = self.session.head(url, timeout=5, allow_redirects=True)  # Reduced from 10s to 5s
            if response.status_code < 400:
                logger.info(f"✅ URL is reachable: {url}")
                return True
            else:
                logger.warning(f"❌ URL returned {response.status_code}: {url}")
                return False
                
        except socket.gaierror as e:
            logger.error(f"❌ DNS resolution failed for {url}: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error for {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error validating {url}: {e}")
            return False

    def save_webpage_with_requests(self, url: str, filename: str) -> bool:
        """Try to save webpage using requests first (faster) with optimized timeout."""
        try:
            response = self.session.get(url, timeout=10, allow_redirects=True)  # Reduced from 15s to 10s
            response.raise_for_status()
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            logger.info(f"📄 Saved with requests: {filename}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Requests failed for {url}: {e}")
            return False

    def save_webpage_with_selenium(self, url: str, filename: str) -> bool:
        """Save webpage using Selenium with optimized performance."""
        driver = self.setup_webdriver()
        if not driver:
            return False
        
        try:
            logger.info(f"🌐 Loading with Selenium: {url}")
            driver.get(url)
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Reduced wait time for dynamic content
            time.sleep(1)  # Reduced from 2s to 1s
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            
            logger.info(f"📄 Saved with Selenium: {filename}")
            return True
            
        except TimeoutException:
            logger.error(f"⏱️ Timeout loading {url}")
            return False
        except WebDriverException as e:
            logger.error(f"🚫 WebDriver error for {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error for {url}: {e}")
            return False
        finally:
            driver.quit()

    def save_webpage(self, url: str, filename: str, retries: int = 1) -> bool:  # Reduced retries from 2 to 1
        """Save webpage with fallback methods and optimized retry logic."""
        for attempt in range(retries + 1):
            if attempt > 0:
                logger.info(f"🔄 Retry {attempt}/{retries} for {url}")
                time.sleep(1 * attempt)  # Reduced backoff time
            
            # Try requests first (faster)
            if self.save_webpage_with_requests(url, filename):
                return True
            
            # Fallback to Selenium
            if self.save_webpage_with_selenium(url, filename):
                return True
        
        logger.error(f"❌ Failed to save {url} after {retries + 1} attempts")
        return False

    def get_blog_links(self, main_url: str, max_links: int = 30) -> List[str]:  # Reduced from 50 to 30
        """Get blog links with improved performance and link filtering."""
        if not self.validate_url(main_url):
            return []
        
        driver = self.setup_webdriver()
        if not driver:
            logger.error(f"Cannot setup webdriver for {main_url}")
            return []
        
        try:
            logger.info(f"🔍 Extracting links from: {main_url}")
            driver.get(main_url)
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            main_domain = urlparse(main_url).netloc
            
            blog_links = []
            for a in soup.find_all('a', href=True):
                full_url = urljoin(main_url, a['href'])
                parsed_url = urlparse(full_url)
                
                # Filter links with better performance
                if (parsed_url.netloc == main_domain and 
                    not any(skip in full_url.lower() for skip in ['#', 'mailto:', 'javascript:', 'tel:']) and
                    not full_url.endswith(('.pdf', '.zip', '.jpg', '.png', '.gif', '.css', '.js'))):
                    
                    if full_url not in blog_links:
                        blog_links.append(full_url)
                        
                        if len(blog_links) >= max_links:
                            logger.info(f"📊 Reached max links limit ({max_links}) for {main_url}")
                            break
            
            logger.info(f"🔗 Found {len(blog_links)} links from {main_url}")
            return blog_links
            
        except Exception as e:
            logger.error(f"❌ Error extracting links from {main_url}: {e}")
            return []
        finally:
            driver.quit()

    def ensure_data_directory(self):
        """Ensure data directory exists."""
        if not os.path.exists('data'):
            os.makedirs('data')
            logger.info("📁 Created data directory")

def main():
    scraper = RobustWebScraper()
    scraper.ensure_data_directory()
    
    # Read blog URLs
    try:
        with open('blogs.txt', 'r') as file:
            blog_urls = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logger.error("❌ blogs.txt file not found")
        return
    
    logger.info(f"🚀 Starting optimized scraper with {len(blog_urls)} blog URLs")
    
    total_saved = 0
    total_failed = 0
    
    for i, blog_url in enumerate(blog_urls, 1):
        logger.info(f"\n{'='*60}")  # Shorter separator for cleaner output
        logger.info(f"📚 Processing Blog {i}/{len(blog_urls)}: {blog_url}")
        logger.info(f"{'='*60}")
        
        # Validate main blog URL first
        if not scraper.validate_url(blog_url):
            logger.error(f"❌ Skipping unreachable blog: {blog_url}")
            continue
        
        # Get all links from the blog with reduced limit
        blog_links = scraper.get_blog_links(blog_url, max_links=20)  # Reduced from 30 to 20
        
        if not blog_links:
            logger.warning(f"⚠️ No links found for {blog_url}")
            continue
        
        # Process each link
        successful_saves = 0
        for j, link in enumerate(blog_links, 1):
            filename = f"blog_{i}_{j}.html"
            filepath = os.path.join('data', filename)
            
            logger.info(f"📄 Processing {j}/{len(blog_links)}: {link}")
            
            if scraper.save_webpage(link, filename):
                try:
                    shutil.move(filename, filepath)
                    logger.info(f"✅ Moved {filename} to data folder")
                    successful_saves += 1
                    total_saved += 1
                except Exception as e:
                    logger.error(f"❌ Error moving {filename}: {e}")
                    total_failed += 1
            else:
                logger.error(f"❌ Failed to save: {link}")
                total_failed += 1
            
            # Reduced delay between requests for faster processing
            time.sleep(0.5)  # Reduced from 1s to 0.5s
        
        logger.info(f"📊 Blog {i} summary: {successful_saves}/{len(blog_links)} pages saved")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Total pages saved: {total_saved}")
    logger.info(f"❌ Total failures: {total_failed}")
    logger.info(f"📁 Data saved to: ./data/ directory")
    logger.info(f"🚀 Optimized for performance - no log files created")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n🛑 Scraping interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise
