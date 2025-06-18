import asyncio
import logging
from typing import List, Dict, Any
import chainlit as cl
from modules.file_handler import handle_file_upload, handle_url
from modules.media_handler import handle_media_upload
from modules.image_analysis import analyze_image
import time

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Enhanced batch processor for handling multiple files and URLs efficiently."""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = None
        self._closed = False
    
    def _get_semaphore(self):
        """Get or create semaphore for current event loop."""
        if self.semaphore is None or self._closed:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)
            self._closed = False
        return self.semaphore
    
    async def close(self):
        """Clean up resources."""
        if self.semaphore and not self._closed:
            # Release any remaining semaphore permits
            while self.semaphore.locked():
                try:
                    self.semaphore.release()
                except ValueError:
                    break
            self._closed = True
    
    async def process_files_batch(self, files: List[cl.File], progress_callback=None) -> Dict[str, Any]:
        """Process multiple files concurrently with progress tracking."""
        if not files:
            return {"success": [], "errors": [], "summary": "No files to process"}
        
        results = {
            "success": [],
            "errors": [],
            "total_files": len(files),
            "processed": 0,
            "summary": ""
        }
        
        # Process files concurrently
        tasks = []
        for i, file in enumerate(files):
            task = self._process_single_file(file, i, progress_callback)
            tasks.append(task)
        
        try:
            # Wait for all tasks to complete
            file_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(file_results):
                if isinstance(result, Exception):
                    results["errors"].append({
                        "file": files[i].name,
                        "error": str(result)
                    })
                else:
                    if result["success"]:
                        results["success"].append(result)
                    else:
                        results["errors"].append(result)
                
                results["processed"] += 1
                
                if progress_callback:
                    await progress_callback(results["processed"], len(files), files[i].name)
            
            # Generate summary
            success_count = len(results["success"])
            error_count = len(results["errors"])
            
            results["summary"] = f"Processed {results['total_files']} files: {success_count} successful, {error_count} failed"
            
        finally:
            # Clean up semaphore
            await self.close()
        
        return results
    
    async def _process_single_file(self, file: cl.File, index: int, progress_callback=None) -> Dict[str, Any]:
        """Process a single file with semaphore for concurrency control."""
        semaphore = self._get_semaphore()
        async with semaphore:
            try:
                start_time = time.time()
                
                # Determine file type and process accordingly
                if hasattr(file, 'type') and file.type.startswith('image/'):
                    # Handle image files
                    content = await analyze_image(file)
                    file_type = "image"
                elif hasattr(file, 'type') and (file.type.startswith('audio/') or file.type.startswith('video/')):
                    # Handle media files
                    content = await handle_media_upload(file)
                    file_type = "media"
                else:
                    # Handle document files
                    content = await handle_file_upload(file)
                    file_type = "document"
                
                processing_time = time.time() - start_time
                
                return {
                    "success": True,
                    "file_name": file.name,
                    "file_type": file_type,
                    "content": content,
                    "processing_time": processing_time,
                    "index": index
                }
                
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {str(e)}")
                return {
                    "success": False,
                    "file_name": file.name,
                    "error": str(e),
                    "index": index
                }
    
    async def process_urls_batch(self, urls: List[str], progress_callback=None) -> Dict[str, Any]:
        """Process multiple URLs concurrently with progress tracking."""
        if not urls:
            return {"success": [], "errors": [], "summary": "No URLs to process"}
        
        results = {
            "success": [],
            "errors": [],
            "total_urls": len(urls),
            "processed": 0,
            "summary": ""
        }
        
        # Process URLs concurrently
        tasks = []
        for i, url in enumerate(urls):
            task = self._process_single_url(url, i, progress_callback)
            tasks.append(task)
        
        try:
            # Wait for all tasks to complete
            url_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(url_results):
                if isinstance(result, Exception):
                    results["errors"].append({
                        "url": urls[i],
                        "error": str(result)
                    })
                else:
                    if result["success"]:
                        results["success"].append(result)
                    else:
                        results["errors"].append(result)
                
                results["processed"] += 1
                
                if progress_callback:
                    await progress_callback(results["processed"], len(urls), urls[i])
            
            # Generate summary
            success_count = len(results["success"])
            error_count = len(results["errors"])
            
            results["summary"] = f"Processed {results['total_urls']} URLs: {success_count} successful, {error_count} failed"
            
        finally:
            # Clean up semaphore
            await self.close()
        
        return results
    
    async def _process_single_url(self, url: str, index: int, progress_callback=None) -> Dict[str, Any]:
        """Process a single URL with semaphore for concurrency control."""
        semaphore = self._get_semaphore()
        async with semaphore:
            try:
                start_time = time.time()
                content = await handle_url(url)
                processing_time = time.time() - start_time
                
                if content.startswith("Error"):
                    return {
                        "success": False,
                        "url": url,
                        "error": content,
                        "index": index
                    }
                
                return {
                    "success": True,
                    "url": url,
                    "content": content,
                    "processing_time": processing_time,
                    "index": index
                }
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                return {
                    "success": False,
                    "url": url,
                    "error": str(e),
                    "index": index
                }
    
    async def process_mixed_content(self, files: List[cl.File] = None, urls: List[str] = None, 
                                   progress_callback=None) -> Dict[str, Any]:
        """Process a mix of files and URLs with comprehensive reporting."""
        results = {
            "files": {"success": [], "errors": []},
            "urls": {"success": [], "errors": []},
            "summary": "",
            "total_items": 0,
            "successful_items": 0,
            "failed_items": 0
        }
        
        # Process files if provided
        if files:
            file_results = await self.process_files_batch(files, progress_callback)
            results["files"] = {
                "success": file_results["success"],
                "errors": file_results["errors"]
            }
        
        # Process URLs if provided
        if urls:
            url_results = await self.process_urls_batch(urls, progress_callback)
            results["urls"] = {
                "success": url_results["success"],
                "errors": url_results["errors"]
            }
        
        # Calculate totals
        results["total_items"] = len(files or []) + len(urls or [])
        results["successful_items"] = len(results["files"]["success"]) + len(results["urls"]["success"])
        results["failed_items"] = len(results["files"]["errors"]) + len(results["urls"]["errors"])
        
        # Generate comprehensive summary
        summary_parts = []
        if files:
            summary_parts.append(f"{len(files)} files")
        if urls:
            summary_parts.append(f"{len(urls)} URLs")
        
        content_type = " and ".join(summary_parts)
        results["summary"] = f"Processed {content_type}: {results['successful_items']} successful, {results['failed_items']} failed"
        
        return results
    
    def format_batch_results(self, results: Dict[str, Any]) -> str:
        """Format batch processing results for display."""
        output = f"**📊 Batch Processing Results**\n\n"
        output += f"**📈 Summary:** {results['summary']}\n\n"
        
        # File results
        if results.get("files", {}).get("success"):
            output += f"**📁 Successfully Processed Files ({len(results['files']['success'])}):**\n"
            for item in results["files"]["success"]:
                output += f"✅ {item['file_name']} ({item.get('file_type', 'unknown')}) - {item.get('processing_time', 0):.2f}s\n"
            output += "\n"
        
        if results.get("files", {}).get("errors"):
            output += f"**❌ Failed Files ({len(results['files']['errors'])}):**\n"
            for item in results["files"]["errors"]:
                file_name = item.get('file_name', item.get('file', 'Unknown'))
                output += f"❌ {file_name}: {item['error']}\n"
            output += "\n"
        
        # URL results
        if results.get("urls", {}).get("success"):
            output += f"**🔗 Successfully Processed URLs ({len(results['urls']['success'])}):**\n"
            for item in results["urls"]["success"]:
                output += f"✅ {item['url'][:50]}... - {item.get('processing_time', 0):.2f}s\n"
            output += "\n"
        
        if results.get("urls", {}).get("errors"):
            output += f"**❌ Failed URLs ({len(results['urls']['errors'])}):**\n"
            for item in results["urls"]["errors"]:
                url = item.get('url', 'Unknown')
                output += f"❌ {url[:50]}...: {item['error']}\n"
            output += "\n"
        
        return output
    
    def combine_content(self, results: Dict[str, Any]) -> str:
        """Combine all successfully processed content into a single string."""
        combined_content = ""
        
        # Add file content
        for item in results.get("files", {}).get("success", []):
            combined_content += f"\n\n{item['content']}\n"
        
        # Add URL content
        for item in results.get("urls", {}).get("success", []):
            combined_content += f"\n\n{item['content']}\n"
        
        return combined_content.strip()

# Global batch processor instance
batch_processor = BatchProcessor(max_concurrent=3)

# Convenience functions for easy import
async def process_files_batch(files: List[cl.File], progress_callback=None) -> Dict[str, Any]:
    """Convenience function to process multiple files."""
    return await batch_processor.process_files_batch(files, progress_callback)

async def process_urls_batch(urls: List[str], progress_callback=None) -> Dict[str, Any]:
    """Convenience function to process multiple URLs."""
    return await batch_processor.process_urls_batch(urls, progress_callback)

async def process_mixed_content(files: List[cl.File] = None, urls: List[str] = None, 
                               progress_callback=None) -> Dict[str, Any]:
    """Convenience function to process mixed content."""
    return await batch_processor.process_mixed_content(files, urls, progress_callback) 