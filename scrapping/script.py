#!/usr/bin/env python3
"""
Celebrity Image Scraper
A responsible web scraping tool for collecting celebrity images for ML training.
Designed for Ubuntu/Linux with Brave browser compatibility.
"""

import os
import requests
import time
import json
import hashlib
from urllib.parse import urljoin, urlparse
from pathlib import Path
import logging
from typing import List, Dict, Optional
import argparse
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third-party imports (install with: pip install requests beautifulsoup4 pillow)
try:
    from bs4 import BeautifulSoup
    from PIL import Image
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install requests beautifulsoup4 pillow")
    exit(1)

@dataclass
class ImageInfo:
    url: str
    filename: str
    celebrity: str
    source_url: str
    size: Optional[tuple] = None

class CelebrityImageScraper:
    def __init__(self, output_dir: str = "celebrity_images", max_workers: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.session = requests.Session()
        self.lock = threading.Lock()
        
        # Headers to appear more like a regular browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_downloaded': 0,
            'total_skipped': 0,
            'total_errors': 0
        }

    def create_celebrity_dir(self, celebrity_name: str) -> Path:
        """Create and return directory for a specific celebrity."""
        safe_name = "".join(c for c in celebrity_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_').lower()
        celeb_dir = self.output_dir / safe_name
        celeb_dir.mkdir(exist_ok=True)
        return celeb_dir

    def get_image_hash(self, image_content: bytes) -> str:
        """Generate hash for image deduplication."""
        return hashlib.md5(image_content).hexdigest()

    def is_valid_image(self, image_content: bytes, min_size: tuple = (100, 100)) -> bool:
        """Validate image content and size."""
        try:
            with Image.open(io.BytesIO(image_content)) as img:
                return img.size[0] >= min_size[0] and img.size[1] >= min_size[1]
        except Exception:
            return False

    def download_image(self, image_info: ImageInfo) -> bool:
        """Download a single image."""
        try:
            celeb_dir = self.create_celebrity_dir(image_info.celebrity)
            file_path = celeb_dir / image_info.filename
            
            # Skip if already exists
            if file_path.exists():
                with self.lock:
                    self.stats['total_skipped'] += 1
                self.logger.info(f"Skipping existing: {image_info.filename}")
                return True

            # Download image
            response = self.session.get(image_info.url, timeout=10)
            response.raise_for_status()
            
            # Validate image
            if not self.is_valid_image(response.content):
                self.logger.warning(f"Invalid image: {image_info.url}")
                with self.lock:
                    self.stats['total_errors'] += 1
                return False

            # Save image
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Save metadata
            metadata = {
                'url': image_info.url,
                'source_url': image_info.source_url,
                'celebrity': image_info.celebrity,
                'download_time': time.time()
            }
            
            metadata_path = celeb_dir / f"{file_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            with self.lock:
                self.stats['total_downloaded'] += 1
            
            self.logger.info(f"Downloaded: {image_info.filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading {image_info.url}: {str(e)}")
            with self.lock:
                self.stats['total_errors'] += 1
            return False

    def scrape_google_images(self, celebrity_name: str, max_images: int = 100) -> List[ImageInfo]:
        """
        Scrape images from Google Images search using multiple approaches.
        """
        images = []
        query = celebrity_name.replace(' ', '+')
        search_url = f"https://www.google.com/search?q={query}&tbm=isch&hl=en"
        
        try:
            # Add random delay
            time.sleep(1 + (hash(celebrity_name) % 3))
            
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()
            
            self.logger.info(f"Google Images response status: {response.status_code}")
            
            # Try multiple parsing approaches
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Method 1: Look for JSON data in script tags
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and 'AF_initDataCallback' in script.string:
                    script_content = script.string
                    # Extract image URLs from the JavaScript data
                    import re
                    urls = re.findall(r'https://[^"]*\.(?:jpg|jpeg|png|webp)', script_content)
                    
                    for i, url in enumerate(urls[:max_images]):
                        if 'encrypted' not in url and len(url) < 500:  # Filter out encrypted/very long URLs
                            filename = f"{celebrity_name.replace(' ', '_').lower()}_google_{i+1:04d}.jpg"
                            images.append(ImageInfo(
                                url=url,
                                filename=filename,
                                celebrity=celebrity_name,
                                source_url=search_url
                            ))
                    
                    if images:
                        break
            
            # Method 2: Traditional img tag scraping (fallback)
            if not images:
                img_tags = soup.find_all('img')
                for i, img in enumerate(img_tags[:max_images]):
                    img_url = img.get('src') or img.get('data-src') or img.get('data-original')
                    
                    if not img_url or img_url.startswith('data:') or 'logo' in img_url.lower():
                        continue
                    
                    if not img_url.startswith('http'):
                        img_url = urljoin(search_url, img_url)
                    
                    # Skip very small images (likely thumbnails)
                    if 'w=' in img_url and any(size in img_url for size in ['w=50', 'w=100', 'w=150']):
                        continue
                    
                    filename = f"{celebrity_name.replace(' ', '_').lower()}_google_{len(images)+1:04d}.jpg"
                    images.append(ImageInfo(
                        url=img_url,
                        filename=filename,
                        celebrity=celebrity_name,
                        source_url=search_url
                    ))
            
            self.logger.info(f"Found {len(images)} image URLs from Google Images for {celebrity_name}")
                
        except Exception as e:
            self.logger.error(f"Error scraping Google Images for {celebrity_name}: {str(e)}")
        
        return images

    def scrape_bing_images(self, celebrity_name: str, max_images: int = 50) -> List[ImageInfo]:
        """Scrape from Bing Images - often more accessible than Google."""
        images = []
        query = celebrity_name.replace(' ', '%20')
        search_url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2&first=1&tsc=ImageBasicHover"
        
        try:
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Bing uses different selectors
            img_containers = soup.find_all('a', class_='iusc')
            
            for i, container in enumerate(img_containers[:max_images]):
                # Extract JSON data from the onclick attribute
                onclick = container.get('m', '')
                if onclick:
                    try:
                        import json
                        data = json.loads(onclick)
                        img_url = data.get('murl') or data.get('turl')
                        
                        if img_url:
                            filename = f"{celebrity_name.replace(' ', '_').lower()}_bing_{i+1:04d}.jpg"
                            images.append(ImageInfo(
                                url=img_url,
                                filename=filename,
                                celebrity=celebrity_name,
                                source_url=search_url
                            ))
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            # Fallback method
            if not images:
                img_tags = soup.find_all('img', class_='mimg')
                for i, img in enumerate(img_tags[:max_images]):
                    img_url = img.get('src')
                    if img_url and img_url.startswith('http'):
                        filename = f"{celebrity_name.replace(' ', '_').lower()}_bing_{i+1:04d}.jpg"
                        images.append(ImageInfo(
                            url=img_url,
                            filename=filename,
                            celebrity=celebrity_name,
                            source_url=search_url
                        ))
            
            self.logger.info(f"Found {len(images)} image URLs from Bing Images for {celebrity_name}")
                
        except Exception as e:
            self.logger.error(f"Error scraping Bing Images for {celebrity_name}: {str(e)}")
        
        return images

    def scrape_duckduckgo_images(self, celebrity_name: str, max_images: int = 50) -> List[ImageInfo]:
        """Scrape from DuckDuckGo Images - usually more permissive."""
        images = []
        query = celebrity_name.replace(' ', '+')
        
        # DuckDuckGo requires a token, so we'll get it first
        try:
            # First, get the search page
            search_url = f"https://duckduckgo.com/?q={query}&t=h_&iax=images&ia=images"
            response = self.session.get(search_url, timeout=15)
            
            # Then get the actual images via their API
            token_url = "https://duckduckgo.com/i.js"
            params = {
                'l': 'us-en',
                'o': 'json',
                'q': celebrity_name,
                'vqd': '',  # This needs to be extracted from the first response
                'f': ',,,',
                'p': '1',
                's': '100'  # Number of results
            }
            
            # Extract vqd token (simplified approach)
            import re
            vqd_match = re.search(r'vqd="([^"]+)"', response.text)
            if vqd_match:
                params['vqd'] = vqd_match.group(1)
                
                api_response = self.session.get(token_url, params=params, timeout=15)
                
                if api_response.status_code == 200:
                    data = api_response.json()
                    results = data.get('results', [])
                    
                    for i, result in enumerate(results[:max_images]):
                        img_url = result.get('image')
                        if img_url:
                            filename = f"{celebrity_name.replace(' ', '_').lower()}_ddg_{i+1:04d}.jpg"
                            images.append(ImageInfo(
                                url=img_url,
                                filename=filename,
                                celebrity=celebrity_name,
                                source_url=search_url
                            ))
            
            self.logger.info(f"Found {len(images)} image URLs from DuckDuckGo for {celebrity_name}")
                
        except Exception as e:
            self.logger.error(f"Error scraping DuckDuckGo Images for {celebrity_name}: {str(e)}")
        
        return images

    def scrape_celebrity(self, celebrity_name: str, max_images: int = 100, sources: List[str] = None) -> None:
        """Scrape images for a single celebrity from multiple sources."""
        if sources is None:
            sources = ['bing', 'duckduckgo', 'google']
        
        all_images = []
        
        self.logger.info(f"Starting scrape for: {celebrity_name}")
        
        for source in sources:
            source_images = []
            if source == 'google':
                source_images = self.scrape_google_images(celebrity_name, max_images // len(sources))
            elif source == 'bing':
                source_images = self.scrape_bing_images(celebrity_name, max_images // len(sources))
            elif source == 'duckduckgo':
                source_images = self.scrape_duckduckgo_images(celebrity_name, max_images // len(sources))
            else:
                self.logger.warning(f"Unknown source: {source}")
                continue
            
            self.logger.info(f"Source {source}: Found {len(source_images)} images for {celebrity_name}")
            all_images.extend(source_images)
            time.sleep(2)  # Be respectful to servers
        
        self.logger.info(f"Total images found for {celebrity_name}: {len(all_images)}")
        
        if not all_images:
            self.logger.warning(f"No images found for {celebrity_name}. Try checking the celebrity name spelling.")
            return
        
        # Download images concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_image = {executor.submit(self.download_image, img): img for img in all_images}
            
            for future in as_completed(future_to_image):
                image = future_to_image[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in thread for {image.filename}: {str(e)}")
                    
                # Small delay between downloads
                time.sleep(0.1)

    def scrape_multiple_celebrities(self, celebrities: List[str], max_images_per_celebrity: int = 100) -> None:
        """Scrape images for multiple celebrities."""
        for celebrity in celebrities:
            self.scrape_celebrity(celebrity, max_images_per_celebrity)
            time.sleep(2)  # Delay between celebrities
            
        self.print_stats()

    def print_stats(self) -> None:
        """Print scraping statistics."""
        total = sum(self.stats.values())
        print("\n" + "="*50)
        print("SCRAPING STATISTICS")
        print("="*50)
        print(f"Total Downloaded: {self.stats['total_downloaded']}")
        print(f"Total Skipped: {self.stats['total_skipped']}")
        print(f"Total Errors: {self.stats['total_errors']}")
        print(f"Total Processed: {total}")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Celebrity Image Scraper')
    parser.add_argument('--celebrities', nargs='+', required=True, 
                        help='List of celebrity names to search for')
    parser.add_argument('--max-images', type=int, default=100,
                        help='Maximum images per celebrity (default: 100)')
    parser.add_argument('--output-dir', default='celebrity_images',
                        help='Output directory (default: celebrity_images)')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of download workers (default: 5)')
    parser.add_argument('--sources', nargs='+', default=['bing', 'duckduckgo', 'google'],
                        help='Sources to scrape from (default: bing duckduckgo google)')
    
    args = parser.parse_args()
    
    # Example usage
    if not args.celebrities:
        print("Example usage:")
        print("python celebrity_scraper.py --celebrities 'Leonardo DiCaprio' 'Emma Watson' --max-images 150")
        return
    
    scraper = CelebrityImageScraper(
        output_dir=args.output_dir,
        max_workers=args.workers
    )
    
    print(f"Starting scrape for: {', '.join(args.celebrities)}")
    print(f"Max images per celebrity: {args.max_images}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Output directory: {args.output_dir}")
    
    scraper.scrape_multiple_celebrities(args.celebrities, args.max_images)

if __name__ == "__main__":
    # Import io here to avoid issues if PIL import fails
    import io
    main()