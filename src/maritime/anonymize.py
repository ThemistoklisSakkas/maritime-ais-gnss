"""Pseudonymise vessel identities in everything that is published.

Processed data under `data/` keep the real MMSI so the analysis can be audited locally. Tables,
figures and the dashboard bundle carry a keyed pseudonym instead (e.g. `V-3F9A1C`) and no vessel
names, so no identifiable ship appears next to a "dark" or "encounter" label: an unexplained AIS
gap is a lead for an analyst, not evidence against a vessel.

The key is random and stored in `data/` (git-ignored). Pseudonyms are therefore stable across
runs on this machine, but cannot be reversed by hashing the ~600 M valid MMSIs, which a plain or
publicly salted hash would allow.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

import polars as pl

from maritime.config import Config

ID_COLS = {"mmsi": "vessel", "mmsi_a": "vessel_a", "mmsi_b": "vessel_b"}
NAME_COLS = ("name", "name_a", "name_b")


def load_key(cfg: Config) -> bytes:
    """Read the local pseudonymisation key, creating it on first use."""
    path = cfg.paths.processed.parent / "pseudonym_key.hex"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
    return bytes.fromhex(path.read_text().strip())


def pseudonym(mmsi: int, key: bytes) -> str:
    """Stable, non-reversible label for one MMSI."""
    digest = hmac.new(key, str(int(mmsi)).encode(), hashlib.sha256).hexdigest()
    return "V-" + digest[:6].upper()


def pseudonymize(df: pl.DataFrame, key: bytes) -> pl.DataFrame:
    """Replace MMSI columns by pseudonyms (renamed to vessel*) and drop vessel names."""
    present = {src: dst for src, dst in ID_COLS.items() if src in df.columns}
    out = df.with_columns(
        pl.col(src).map_elements(lambda m: pseudonym(m, key), return_dtype=pl.String)
        for src in present)
    return out.rename(present).drop([c for c in NAME_COLS if c in df.columns])
