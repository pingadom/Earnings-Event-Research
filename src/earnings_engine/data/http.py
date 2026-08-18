"""Polite HTTP client with retries, throttling, and an on-disk response cache."""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """A remote resource remained unavailable after polite retries."""


@dataclass
class HttpClient:
    """Sequential retrying client suitable for SEC and small public datasets."""

    user_agent: str
    cache_dir: Path | None = None
    min_interval: float = 0.0
    timeout: float = 30.0
    retries: int = 4
    backoff: float = 1.0
    session: requests.Session = field(default_factory=requests.Session)
    _last_request: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/csv,text/plain,*/*",
            }
        )

    def get_bytes(
        self,
        url: str,
        *,
        force: bool = False,
        max_age_seconds: float | None = None,
    ) -> bytes:
        cache_path = self._cache_path(url)
        if cache_path and cache_path.exists() and not force:
            age = time.time() - cache_path.stat().st_mtime
            if max_age_seconds is None or age <= max_age_seconds:
                return cache_path.read_bytes()

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 404:
                    raise DownloadError(f"HTTP 404 for {url}")
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"retryable HTTP {response.status_code} for {url}", response=response
                    )
                response.raise_for_status()
                content = response.content
                if cache_path:
                    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
                    temporary.write_bytes(content)
                    temporary.replace(cache_path)
                return content
            except (requests.RequestException, DownloadError) as exc:
                last_error = exc
                if attempt >= self.retries or isinstance(exc, DownloadError):
                    break
                delay = self.backoff * (2**attempt) + random.uniform(0.0, 0.25)
                log.warning(
                    "request failed (%s); retry %d/%d in %.1fs: %s",
                    exc,
                    attempt + 1,
                    self.retries,
                    delay,
                    url,
                )
                time.sleep(delay)
        raise DownloadError(f"failed after {self.retries + 1} attempts: {url}: {last_error}")

    def get_json(self, url: str, **kwargs):
        import json

        try:
            return json.loads(self.get_bytes(url, **kwargs).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise DownloadError(f"malformed JSON from {url}: {exc}") from exc

    def get_text(self, url: str, **kwargs) -> str:
        content = self.get_bytes(url, **kwargs)
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise DownloadError(f"could not decode response from {url}")

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.response"

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()
