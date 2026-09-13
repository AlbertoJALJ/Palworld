"""A deliberately polite HTTP client for scraping.

Community wikis are run by volunteers and pay for their own bandwidth. This
client is built so that using it correctly is the path of least resistance:

- `robots.txt` is fetched first and honoured, including `Crawl-delay`.
- Requests are rate limited, never concurrent.
- Everything is cached on disk, so a re-run or a crashed run costs zero
  requests, and developing the parser does not mean re-fetching anything.
- The User-Agent identifies the tool rather than impersonating a browser.

None of this is optional politeness: a scraper that hammers a fan wiki tends to
get the whole IP range blocked, which breaks the tool for everyone.
"""

from __future__ import annotations

import hashlib
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self
from urllib.parse import urljoin, urlparse

import httpx

from .base import IngestError

DEFAULT_USER_AGENT = (
    "palworld-data-api/0.1 (+https://github.com/AlbertoJALJ/Palworld) "
    "python-httpx"
)


@dataclass
class PoliteClient:
    """Rate-limited, cached, robots-aware HTTP GET."""

    cache_dir: Path
    user_agent: str = DEFAULT_USER_AGENT
    min_delay: float = 1.0
    timeout: float = 30.0
    respect_robots: bool = True
    max_retries: int = 3

    _robots: dict[str, urllib.robotparser.RobotFileParser] = field(
        default_factory=dict, init=False, repr=False
    )
    _last_request: float = field(default=0.0, init=False, repr=False)
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Self:
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
            follow_redirects=True,
        )
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.html"

    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(origin, "/robots.txt")
            try:
                assert self._client is not None
                response = self._client.get(robots_url)
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    # No robots.txt is permission by omission, not a reason to
                    # bail; but anything other than 200 means we learned nothing.
                    parser.parse([])
            except httpx.HTTPError:
                # If robots.txt itself is unreachable, assume the worst rather
                # than assuming consent.
                raise IngestError(
                    f"could not fetch {robots_url}; refusing to crawl blind. "
                    f"Pass respect_robots=False only if you have permission."
                ) from None
            self._robots[origin] = parser
        return self._robots[origin]

    def _wait_turn(self, url: str) -> None:
        delay = self.min_delay
        if self.respect_robots:
            crawl_delay = self._robots_for(url).crawl_delay(self.user_agent)
            if crawl_delay is not None:
                delay = max(delay, float(crawl_delay))
        elapsed = time.monotonic() - self._last_request
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request = time.monotonic()

    def get(self, url: str, *, use_cache: bool = True) -> str:
        """Fetch `url` as text, from cache when possible."""
        if self._client is None:
            raise IngestError("PoliteClient must be used as a context manager")

        cache_path = self._cache_path(url)
        if use_cache and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        if self.respect_robots and not self._robots_for(url).can_fetch(
            self.user_agent, url
        ):
            raise IngestError(
                f"robots.txt disallows fetching {url}. Stopping rather than "
                f"ignoring it -- contact the site owner if you need this data."
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._wait_turn(url)
            try:
                response = self._client.get(url)
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(2.0 * (2**attempt))
                continue

            if response.status_code == 429 or response.status_code >= 500:
                # Back off hard on rate limiting; the site is telling us to slow
                # down and the only correct response is to comply.
                retry_after = response.headers.get("Retry-After")
                pause = float(retry_after) if retry_after else 5.0 * (2**attempt)
                last_error = IngestError(
                    f"{url} returned {response.status_code}"
                )
                time.sleep(pause)
                continue

            if response.status_code != 200:
                raise IngestError(f"{url} returned HTTP {response.status_code}")

            cache_path.write_text(response.text, encoding="utf-8")
            return response.text

        raise IngestError(f"giving up on {url} after {self.max_retries} attempts: {last_error}")
