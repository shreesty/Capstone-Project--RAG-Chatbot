
import os
import re
import time
import json
from collections import deque
from urllib.parse import urlparse, urljoin


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 15

# Delay between requests (seconds) -> be polite
REQUEST_DELAY = 1.0

# Max pages per domain to avoid infinite crawling
MAX_PAGES_PER_DOMAIN = 40

# Max depth from seed URL (0 = only seed page, 1 = seed + links from it, etc.)
MAX_DEPTH = 2

# Base directory where all data will be stored
BASE_OUTPUT_DIR = "scraped_data"

