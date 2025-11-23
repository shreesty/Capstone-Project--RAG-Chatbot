# edu_scraper.py
from seed_urls import SEED_URLS

import os
import re
import time
import json
from collections import deque
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 15

# Delay between requests (seconds)
REQUEST_DELAY = 1.0

# Max pages per domain to avoid infinite crawling
MAX_PAGES_PER_DOMAIN = 40

# Max depth from seed URL (0 = only seed page, 1 = seed + links from it, etc.)
MAX_DEPTH = 2

# Base directory where all data will be stored
BASE_OUTPUT_DIR = "scraped_data"



def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def normalize_domain(url: str) -> str:
    """
    Extract and normalize domain from URL.
    For example:
        https://www.tu.edu.np/ -> tu.edu.np
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def is_same_domain(url: str, domain: str) -> bool:
    """Check if given URL belongs to the same domain."""
    parsed = urlparse(url)
    target = parsed.netloc.lower()
    if target.startswith("www."):
        target = target[4:]
    return target == domain


def sanitize_filename(text: str, max_length: int = 120) -> str:
    """
    Turn arbitrary text (like URL) into a safe filename.
    Remove weird characters and trim length.
    """
    text = text.strip().replace("://", "_")
    text = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", text)
    return text[:max_length]


def guess_extension_from_url(url: str) -> str:
    """
    Try to guess file extension from URL path.
    Returns something like '.pdf', '.jpg', '.html', or '' if unknown.
    """
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext.lower()


def is_document_link(url: str) -> bool:
    """Check if URL looks like a document (PDF, DOC, DOCX)."""
    ext = guess_extension_from_url(url)
    return ext in (".pdf", ".doc", ".docx")


def is_image_link(url: str) -> bool:
    """Check if URL looks like an image."""
    ext = guess_extension_from_url(url)
    return ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")


def is_html_like(url: str) -> bool:
    """
    Decide if an URL is an HTML page we should crawl:
    - no extension
    - or typical web-page extensions: .html, .htm, .php, .asp, .aspx, etc.
    """
    ext = guess_extension_from_url(url)
    if ext == "":
        return True
    return ext in (".html", ".htm", ".php", ".asp", ".aspx", ".jsp")


def fetch_url(url: str) -> requests.Response | None:
    """
    Fetch a URL with requests.
    Returns Response on success, None on error.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        # Basic check for HTTP 200
        if resp.status_code == 200:
            return resp
        else:
            print(f"[WARN] Non-200 status for {url}: {resp.status_code}")
            return None
    except requests.RequestException as e:
        print(f"[ERROR] Request failed for {url}: {e}")
        return None
    



def save_page_html(domain: str, url: str, html: str) -> str:
    """
    Save raw HTML to file and return file path.
    Structure: scraped_data/{domain}/html/{filename}.html
    """
    dir_path = os.path.join(BASE_OUTPUT_DIR, domain, "html")
    ensure_dir(dir_path)

    filename = sanitize_filename(url) + ".html"
    filepath = os.path.join(dir_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath


def extract_visible_text_from_html(html: str) -> str:
    """
    Use BeautifulSoup to extract visible text from HTML.
    Removes scripts, styles, etc., and returns clean text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Get text with line breaks between blocks
    text = soup.get_text(separator="\n")

    # Clean up excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned


def save_page_text(domain: str, url: str, text: str) -> str:
    """
    Save extracted text to file and return file path.
    Structure: scraped_data/{domain}/text/{filename}.txt
    """
    dir_path = os.path.join(BASE_OUTPUT_DIR, domain, "text")
    ensure_dir(dir_path)

    filename = sanitize_filename(url) + ".txt"
    filepath = os.path.join(dir_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return filepath


def save_metadata(domain: str, url: str, depth: int, html_path: str, text_path: str) -> None:
    """
    Save JSON metadata per page:
    - URL
    - domain
    - depth
    - file paths
    This is helpful later for indexing/embeddings.
    """
    dir_path = os.path.join(BASE_OUTPUT_DIR, domain)
    ensure_dir(dir_path)

    metadata_file = os.path.join(dir_path, "metadata.jsonl")

    record = {
        "url": url,
        "domain": domain,
        "depth": depth,
        "html_path": html_path,
        "text_path": text_path,
        "timestamp": time.time(),
    }

    # Append as one JSON object per line
    with open(metadata_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_binary_file(domain: str, url: str, subfolder: str) -> str | None:
    """
    Download binary file (PDF, DOCX, image, etc.) into:
        scraped_data/{domain}/{subfolder}/filename.ext

    Returns saved path or None on error.
    """
    resp = fetch_url(url)
    if resp is None:
        return None

    dir_path = os.path.join(BASE_OUTPUT_DIR, domain, subfolder)
    ensure_dir(dir_path)

    ext = guess_extension_from_url(url)
    if not ext:
        # Fallback for unknown extension
        ext = ".bin"

    filename = sanitize_filename(url) + ext
    filepath = os.path.join(dir_path, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(resp.content)
        print(f"[INFO] Saved {subfolder} file: {filepath}")
        return filepath
    except OSError as e:
        print(f"[ERROR] Could not save file {filepath}: {e}")
        return None


class SimpleCrawler:
    """
    A simple BFS crawler limited by:
    - max pages per domain
    - max depth from seed
    - same-domain constraint (no external domains)
    """

    def __init__(
        self,
        seeds: list[str],
        max_pages_per_domain: int = MAX_PAGES_PER_DOMAIN,
        max_depth: int = MAX_DEPTH,
        delay: float = REQUEST_DELAY,
    ) -> None:
        self.seeds = seeds
        self.max_pages_per_domain = max_pages_per_domain
        self.max_depth = max_depth
        self.delay = delay

        # Track visited URLs so we don't re-fetch
        self.visited: set[str] = set()

        # Count pages per domain: {domain: count}
        self.domain_page_count: dict[str, int] = {}

    def crawl(self) -> None:
        """Start crawling from all seed URLs."""
        for seed in self.seeds:
            domain = normalize_domain(seed)
            print(f"\n[DOMAIN] Starting crawl for: {domain} ({seed})")
            self.crawl_domain(seed, domain)

    def crawl_domain(self, seed_url: str, domain: str) -> None:
        """
        Crawl a single domain starting from seed_url.
        BFS over URLs with (url, depth).
        """
        # Initialize visited count for this domain if not present
        if domain not in self.domain_page_count:
            self.domain_page_count[domain] = 0

        queue = deque()
        queue.append((seed_url, 0))

        while queue:
            url, depth = queue.popleft()

            # Stop if depth exceeds configured max
            if depth > self.max_depth:
                continue

            # Respect per-domain page limit
            if self.domain_page_count[domain] >= self.max_pages_per_domain:
                print(f"[INFO] Reached max pages for {domain}: {self.max_pages_per_domain}")
                break

            # Skip if already visited
            if url in self.visited:
                continue

            # Skip if URL is not in this domain
            if not is_same_domain(url, domain):
                continue

            print(f"[CRAWL] {url} (depth={depth})")

            # Mark visited before fetching to avoid duplicates
            self.visited.add(url)

            # Fetch page
            resp = fetch_url(url)
            if resp is None:
                continue

            content_type = resp.headers.get("Content-Type", "").lower()

            # If this is likely HTML, parse with BeautifulSoup
            if "text/html" in content_type or is_html_like(url):
                self.handle_html_page(domain, url, resp.text, depth, queue)
            else:
                # If not HTML, but still at seed/known link, we can try downloading it
                if is_document_link(url):
                    download_binary_file(domain, url, subfolder="docs")
                elif is_image_link(url):
                    download_binary_file(domain, url, subfolder="images")

            # Be polite – delay between requests
            time.sleep(self.delay)

    def handle_html_page(
        self,
        domain: str,
        url: str,
        html: str,
        depth: int,
        queue: deque,
    ) -> None:
        """
        Process an HTML page:
        - save HTML
        - extract and save text
        - update metadata
        - find links (HTML, PDFs, docs, images) and enqueue or download
        """
        # Update page count for domain
        self.domain_page_count[domain] += 1

        # Save raw HTML
        html_path = save_page_html(domain, url, html)

        # Extract visible text
        text = extract_visible_text_from_html(html)

        # Save text
        text_path = save_page_text(domain, url, text)

        # Save metadata
        save_metadata(domain, url, depth, html_path, text_path)

        # Parse HTML with BeautifulSoup to find links & resources
        soup = BeautifulSoup(html, "html.parser")

        # Handle all <a href="..."> links (for HTML pages, PDFs, docs, etc.)
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            # Resolve relative links like "/about" to full URL
            full_url = urljoin(url, href)

            # Skip mailto:, tel:, etc.
            if full_url.startswith("mailto:") or full_url.startswith("tel:"):
                continue

            # Documents (pdf/doc/docx)
            if is_document_link(full_url):
                download_binary_file(domain, full_url, subfolder="docs")
                continue

            # HTML-like pages (follow them)
            if is_html_like(full_url):
                if full_url not in self.visited and is_same_domain(full_url, domain):
                    # Enqueue for crawling at next depth
                    if depth + 1 <= self.max_depth:
                        queue.append((full_url, depth + 1))

        # Handle images separately (<img src="...">)
        for img in soup.find_all("img", src=True):
            src = img["src"].strip()
            full_url = urljoin(url, src)
            if is_image_link(full_url) and is_same_domain(full_url, domain):
                download_binary_file(domain, full_url, subfolder="images")

if __name__ == "__main__":
    crawler = SimpleCrawler(
        seeds=SEED_URLS,
        max_pages_per_domain=MAX_PAGES_PER_DOMAIN,
        max_depth=MAX_DEPTH,
        delay=REQUEST_DELAY,
    )

    crawler.crawl()

