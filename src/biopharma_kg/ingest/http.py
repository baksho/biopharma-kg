"""HTTP client with retries, per-host rate limiting and an on-disk response cache.

The cache makes re-runs cheap and polite to the public APIs: a response is stored under
``data/cache/<sha1>.json`` keyed by URL + sorted query params.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised for network / API failures, with a human-actionable message."""


class HttpClient:
    def __init__(
        self,
        cache_dir: Path,
        timeout_s: float = 60,
        max_retries: int = 5,
        backoff_s: float = 2.0,
        user_agent: str = "biopharma-kg",
        min_interval_s: dict[str, float] | None = None,
        use_cache: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.use_cache = use_cache
        self.min_interval_s = min_interval_s or {}
        self._last_call: dict[str, float] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    # ------------------------------------------------------------------ cache
    def _key(self, url: str, params: dict | None) -> Path:
        blob = url + "?" + json.dumps(params or {}, sort_keys=True, default=str)
        return self.cache_dir / (hashlib.sha1(blob.encode()).hexdigest() + ".cache")

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        gap = self.min_interval_s.get(host, 0.0)
        if gap:
            wait = self._last_call.get(host, 0.0) + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        self._last_call[host] = time.monotonic()

    # ---------------------------------------------------------------- request
    def get(
        self,
        url: str,
        params: dict | None = None,
        *,
        accept_404: bool = False,
    ) -> tuple[int, str]:
        """GET and return ``(status_code, text)``. Retries 429/5xx/connection errors.

        With ``accept_404=True`` a 404 is returned (and cached) instead of raised: openFDA
        answers "no matches" with HTTP 404.
        """
        path = self._key(url, params)
        if self.use_cache and path.exists():
            payload = json.loads(path.read_text())
            return payload["status"], payload["text"]

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle(url)
            try:
                r = self.session.get(url, params=params, timeout=self.timeout_s)
            except requests.RequestException as e:  # connection reset, DNS, TLS, timeout ...
                last_err = e
                log.warning("GET %s failed (%s), attempt %d/%d", url, type(e).__name__, attempt, self.max_retries)
                time.sleep(self.backoff_s * attempt)
                continue
            if r.status_code == 200 or (accept_404 and r.status_code == 404):
                if self.use_cache:
                    path.write_text(json.dumps({"status": r.status_code, "text": r.text}))
                return r.status_code, r.text
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = FetchError(f"HTTP {r.status_code} from {url}")
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff_s * attempt
                log.warning("HTTP %s from %s, retrying in %.1fs (%d/%d)", r.status_code, url, delay, attempt, self.max_retries)
                time.sleep(delay)
                continue
            raise FetchError(f"HTTP {r.status_code} from {r.url}: {r.text[:300]!r}")
        raise FetchError(
            f"Could not reach {url} after {self.max_retries} attempts ({last_err}). "
            "Check your internet connection / proxy / firewall; the pipeline needs outbound HTTPS "
            "to the data providers listed in the README."
        )

    def get_json(self, url: str, params: dict | None = None, *, accept_404: bool = False) -> Any:
        status, text = self.get(url, params, accept_404=accept_404)
        if status == 404:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise FetchError(f"Non-JSON response from {url}: {text[:200]!r}") from e

    def download(self, url: str, dest: Path, chunk: int = 1 << 20) -> Path:
        """Stream a (possibly large) file to ``dest``."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with self.session.get(url, stream=True, timeout=self.timeout_s) as r:
                if r.status_code != 200:
                    raise FetchError(f"HTTP {r.status_code} downloading {url}")
                with open(tmp, "wb") as fh:
                    for part in r.iter_content(chunk):
                        fh.write(part)
        except requests.RequestException as e:
            raise FetchError(f"Download of {url} failed: {e}") from e
        tmp.replace(dest)
        return dest
