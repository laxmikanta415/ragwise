"""Document lifecycle — staleness detection, TTL expiry, purge."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _parse_as_of(as_of: str | datetime | None) -> datetime:
    if as_of is None or as_of == "now":
        return datetime.now(tz=UTC).replace(tzinfo=None)
    if isinstance(as_of, datetime):
        return as_of
    return datetime.fromisoformat(str(as_of))


@dataclass
class StaleSource:
    """A source whose valid_until is in the past (or expiring soon)."""

    source: str
    valid_until: str
    chunk_count: int = 0
    expired_days_ago: int = 0  # negative means "expires in N days" (not yet expired)


@dataclass
class PurgeResult:
    purged_sources: list[str] = field(default_factory=list)
    purged_chunks: int = 0


class StalenessChecker:
    """Inspects store metadata to find expired/expiring documents."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def list_stale(
        self,
        as_of: str | datetime | None = "now",
        expiring_soon_days: int = 0,
    ) -> tuple[list[StaleSource], list[StaleSource]]:
        """Return (expired, expiring_soon) source lists.

        expired: sources whose valid_until < as_of
        expiring_soon: sources expiring within expiring_soon_days (not yet expired)
        """
        as_of_dt = _parse_as_of(as_of)
        sources_meta = await self._store.list_sources_with_metadata()

        # Count chunks per source
        chunk_counts: dict[str, int] = {}
        if hasattr(self._store, "_docs"):
            for doc in self._store._docs:
                chunk_counts[doc.source] = chunk_counts.get(doc.source, 0) + 1

        expired: list[StaleSource] = []
        expiring_soon: list[StaleSource] = []

        seen: set[str] = set()
        for meta in sources_meta:
            source = meta.get("source", "")
            if source in seen:
                continue
            seen.add(source)

            valid_until_raw = meta.get("valid_until")
            if valid_until_raw is None:
                continue

            try:
                valid_until_dt = datetime.fromisoformat(str(valid_until_raw))
            except ValueError:
                continue

            delta = as_of_dt - valid_until_dt
            days_diff = delta.days  # positive = expired, negative = future

            ss = StaleSource(
                source=source,
                valid_until=str(valid_until_raw),
                chunk_count=chunk_counts.get(source, 0),
                expired_days_ago=days_diff,
            )

            if days_diff > 0:
                expired.append(ss)
            elif expiring_soon_days > 0 and -days_diff <= expiring_soon_days:
                expiring_soon.append(ss)

        return expired, expiring_soon

    async def purge_stale(self, as_of: str | datetime | None = "now") -> PurgeResult:
        """Delete all chunks from expired sources. Returns purge summary."""
        expired, _ = await self.list_stale(as_of=as_of, expiring_soon_days=0)
        result = PurgeResult()
        for ss in expired:
            await self._store.delete(ss.source)
            result.purged_sources.append(ss.source)
            result.purged_chunks += ss.chunk_count
        return result

    async def staleness_report(
        self,
        as_of: str | datetime | None = "now",
        expiring_soon_days: int = 30,
    ) -> str:
        """Return a human-readable staleness report string."""
        as_of_dt = _parse_as_of(as_of)
        all_sources = await self._store.list_sources()
        expired, expiring_soon = await self.list_stale(
            as_of=as_of_dt, expiring_soon_days=expiring_soon_days
        )

        lines = [
            f"Staleness Report — {as_of_dt.strftime('%Y-%m-%d')}",
            "─" * 35,
            f"Total sources indexed: {len(all_sources)}",
            f"Expired (valid_until in past): {len(expired)}",
        ]
        for ss in expired:
            lines.append(f"  - {ss.source} (expired {ss.expired_days_ago} days ago)")

        lines.append(f"Expiring soon (within {expiring_soon_days} days): {len(expiring_soon)}")
        for ss in expiring_soon:
            lines.append(f"  - {ss.source} (expires in {-ss.expired_days_ago} days)")

        return "\n".join(lines)
