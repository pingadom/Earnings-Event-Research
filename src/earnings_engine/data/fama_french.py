"""Kenneth French daily factor downloads."""

from __future__ import annotations

from io import BytesIO, StringIO
from zipfile import ZipFile

import pandas as pd

from .http import HttpClient

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF3_URL = f"{BASE}/F-F_Research_Data_Factors_daily_CSV.zip"
FF5_URL = f"{BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = f"{BASE}/F-F_Momentum_Factor_daily_CSV.zip"


def _read_daily_zip(payload: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith((".csv", ".txt"))]
        if not names:
            raise ValueError("Fama-French ZIP contains no CSV/text member")
        text = archive.read(names[0]).decode("latin-1")

    lines = text.splitlines()
    header = next(
        (i for i, line in enumerate(lines) if line.lstrip().startswith(",") and "Mkt-RF" in line),
        None,
    )
    if header is None:
        header = next(
            (i for i, line in enumerate(lines) if line.lstrip().startswith(",") and "Mom" in line),
            None,
        )
    if header is None:
        raise ValueError("could not locate factor-table header")
    body = [lines[header]]
    for line in lines[header + 1 :]:
        first = line.split(",", 1)[0].strip()
        if not (len(first) == 8 and first.isdigit()):
            break
        body.append(line)
    frame = pd.read_csv(StringIO("\n".join(body)))
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="coerce")
    for column in frame.columns.drop("date"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100.0
    return frame.dropna(subset=["date"])


def download_fama_french_daily(
    client: HttpClient, start: str, end: str, *, force: bool = False
) -> pd.DataFrame:
    """Return FF3 plus RMW/CMA and momentum, all in decimal-return units."""
    max_age = 24 * 60 * 60
    ff3 = _read_daily_zip(client.get_bytes(FF3_URL, force=force, max_age_seconds=max_age))
    ff5 = _read_daily_zip(client.get_bytes(FF5_URL, force=force, max_age_seconds=max_age))
    mom = _read_daily_zip(client.get_bytes(MOM_URL, force=force, max_age_seconds=max_age))

    ff3 = ff3.rename(columns=lambda c: str(c).strip())
    ff5 = ff5.rename(columns=lambda c: str(c).strip())
    mom = mom.rename(columns=lambda c: str(c).strip())
    selected = ff3[[c for c in ("date", "Mkt-RF", "SMB", "HML", "RF") if c in ff3]]
    additions = ff5[[c for c in ("date", "RMW", "CMA") if c in ff5]]
    momentum_col = next((c for c in mom.columns if c.lower() == "mom"), None)
    if momentum_col:
        momentum = mom[["date", momentum_col]].rename(columns={momentum_col: "Mom"})
        out = selected.merge(additions, on="date", how="outer").merge(
            momentum, on="date", how="outer"
        )
    else:
        out = selected.merge(additions, on="date", how="outer")
        out["Mom"] = pd.NA
    mask = out["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    out = out.loc[mask].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["source"] = "Kenneth French Data Library"
    return out
