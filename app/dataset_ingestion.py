"""Dynamic dataset ingestion and additive schema evolution (Final Upgrade P2).

Pipeline: receive upload (size, extension, SHA-256) -> detect structure -> profile + infer types (full-file
streaming pass) -> normalise identifiers -> resolve target -> create table or compare with the existing one ->
preview (no writes) -> commit (one transaction: additive ALTER, COPY of pre-validated values, metadata) ->
ANALYZE -> SchemaInspector cache invalidation.

Rules that have NO code path at all: dropping/renaming columns, dropping/truncating tables, narrowing types,
changing constraints, NOT NULL. Uploaded content is never executed as SQL; identifiers are always quoted.
This is a different feature from `app/ingestion.py` (slow-query *workload log* ingestion), which is unchanged.
"""
from __future__ import annotations

import asyncio
import codecs
import csv
import datetime as dt
import decimal
import difflib
import hashlib
import io
import json
import re
import secrets
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app.config import Settings, get_settings
from app.errors import ErrorCategory, QuerySentinelError
from app.pii import PIIDetector
from app.provenance import Provenance
from app.structured_log import TraceLogger

LINEAGE_COLUMN = "_qs_ingest_id"
HIDDEN_PREFIX = "_qs_"
ALLOWED_EXT = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl"}
# smallint is deliberately never inferred: a later upload with larger values would otherwise need widening
TYPE_ORDER = ["boolean", "integer", "bigint", "numeric", "double precision", "date", "timestamp",
              "timestamptz", "uuid", "jsonb", "text"]
BOOL_TRUE = {"true", "t", "yes", "y"}
BOOL_FALSE = {"false", "f", "no", "n"}
BOOL_NAME_HINT = re.compile(r"(^is_|^has_|^can_|_flag$|^flag|active|enabled|deleted|valid|bool)")
_INT_RE = re.compile(r"^[+-]?\d+$")
_DEC_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_MONEY_RE = re.compile(r"^[+-]?[$€£₹¥]?\s?\d{1,3}(,\d{3})+(\.\d+)?$|^[+-]?[$€£₹¥]\s?\d+(\.\d+)?$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLASH_DATE = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")
_ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?$")
_ISO_TSTZ = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}(:?\d{2})?)$")
PG_RESERVED = {
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "asymmetric", "both", "case", "cast", "check",
    "collate", "column", "constraint", "create", "current_catalog", "current_date", "current_role", "current_time",
    "current_timestamp", "current_user", "default", "deferrable", "desc", "distinct", "do", "else", "end", "except",
    "false", "fetch", "for", "foreign", "from", "grant", "group", "having", "in", "initially", "intersect", "into",
    "lateral", "leading", "limit", "localtime", "localtimestamp", "not", "null", "offset", "on", "only", "or",
    "order", "placing", "primary", "references", "returning", "select", "session_user", "some", "symmetric",
    "table", "then", "to", "trailing", "true", "union", "unique", "user", "using", "variadic", "when", "where",
    "window", "with", "authorization", "binary", "collation", "concurrently", "cross", "current_schema",
    "freeze", "full", "ilike", "inner", "is", "isnull", "join", "left", "like", "natural", "notnull", "outer",
    "overlaps", "right", "similar", "tablesample", "verbose", "oid", "ctid", "xmin", "xmax", "cmin", "cmax",
    "tableoid"}
# safe widening chains (existing -> allowed wider incoming target)
WIDENING = {"smallint": ["integer", "bigint", "numeric"], "integer": ["bigint", "numeric"], "bigint": ["numeric"],
            "date": ["timestamp", "timestamptz"], "timestamp": ["timestamptz"], "double precision": []}
_PII = PIIDetector()


# =====================================================================================================
# identifiers
# =====================================================================================================
def normalize_identifier(raw: str, fallback: str = "col", _part: bool = False) -> str:
    if "__" in str(raw or "") and not _part:      # flattened JSON keys keep their parent__child separator
        parts = [normalize_identifier(x, fallback, True) for x in str(raw).split("__") if x.strip("_ ")]
        if len(parts) > 1:
            return "__".join(parts).encode("utf-8")[:63].decode("utf-8", "ignore").rstrip("_")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(raw or "").strip())
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"c_{s}"
    if s.startswith("qs_"):
        s = f"u_{s}"            # never collide with hidden lineage columns (_qs_*)
    if s in PG_RESERVED and not _part:
        s = f"{s}_col"
    return s.encode("utf-8")[:63].decode("utf-8", "ignore").rstrip("_") or fallback


def dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        base, k = n, seen.get(n, 0)
        while n in seen or n == LINEAGE_COLUMN:
            k += 1
            suffix = f"_{k + 1}"
            n = base[: 63 - len(suffix)] + suffix
        seen[base] = k
        seen[n] = 0
        out.append(n)
    return out


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# =====================================================================================================
# structure detection / readers (streaming)
# =====================================================================================================
def detect_encoding(path: Path) -> str:
    head = path.open("rb").read(4)
    if head.startswith(codecs.BOM_UTF8):
        enc = "utf-8-sig"
    else:
        enc = "utf-8"
    dec = codecs.getincrementaldecoder(enc)()
    try:
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                dec.decode(chunk)
            dec.decode(b"", final=True)
        return enc
    except UnicodeDecodeError:
        return "latin-1"


SINGLE_COLUMN_DELIM = "\x1f"   # unit separator: never in normal text -> whole line is one field


def detect_csv_dialect(sample: str) -> tuple[str, bool]:
    """Pick the delimiter that yields a consistent column count (header > 1 column) across the sample.
    If no candidate splits the header, the file is single-column (commas inside values are kept)."""
    lines = sample.splitlines()
    if len(sample) >= 64 * 1024 and len(lines) > 1:
        lines = lines[:-1]                       # last line may be cut in the middle
    body = "\n".join(lines[:400])
    best, best_score = None, (-1.0, 0)
    for cand in (",", ";", "\t", "|"):
        try:
            rows = [r for r in csv.reader(io.StringIO(body), delimiter=cand) if r]
        except csv.Error:
            continue
        if not rows or len(rows[0]) < 2:
            continue
        n = len(rows[0])
        consistency = sum(1 for r in rows if len(r) == n) / len(rows)
        score = (round(consistency, 3), n)
        if score > best_score:
            best, best_score = cand, score
    delim = best if best is not None and best_score[0] >= 0.5 else SINGLE_COLUMN_DELIM
    first = next(csv.reader(io.StringIO(body), delimiter=delim), [])
    while first and not first[-1].strip():       # trailing empty header cells (",,,,," exports)
        first = first[:-1]
    cells = [c.strip() for c in first if c.strip()]
    try:
        header = csv.Sniffer().has_header(body[:20000]) if delim != SINGLE_COLUMN_DELIM else True
    except csv.Error:
        header = True
    # a first row whose non-empty cells are all non-numeric text is a header even if the sniffer disagrees
    if cells and all(not _DEC_RE.match(c) for c in cells):
        header = True
    if cells and all(_DEC_RE.match(c) for c in cells):
        header = False                           # purely numeric first row -> data, not a header
    return delim, header


def flatten_record(rec: dict[str, Any]) -> dict[str, Any]:
    """One nesting level becomes parent__child; arrays and deeper nesting stay as JSON (-> jsonb)."""
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            for ck, cv in v.items():
                out[f"{k}__{ck}"] = json.dumps(cv) if isinstance(cv, (dict, list)) else cv
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def _json_records(path: Path, enc: str, fmt: str) -> tuple[list[dict], str]:
    text = path.read_text(encoding=enc)
    stripped = text.lstrip()
    if fmt == "jsonl" or (stripped and stripped[0] == "{" and "\n{" in stripped and not stripped.rstrip().endswith("]")
                          and _looks_jsonl(stripped)):
        recs = []
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, f"invalid JSON on line {i}",
                                             stage="ingestion") from None
                if not isinstance(obj, dict):
                    raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, f"line {i} is not a JSON object",
                                             stage="ingestion")
                recs.append(obj)
        return recs, "json_lines"
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, f"invalid JSON (line {exc.lineno})",
                                 stage="ingestion") from None
    if isinstance(doc, list):
        kind = "json_array"
    elif isinstance(doc, dict) and len([v for v in doc.values() if isinstance(v, list)]) == 1:
        doc = next(v for v in doc.values() if isinstance(v, list))
        kind = "json_object_with_array"
    else:
        raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED,
                                 "JSON must be an array of objects, JSON Lines, or an object with one array key",
                                 stage="ingestion")
    if not all(isinstance(r, dict) for r in doc):
        raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, "every JSON array element must be an object",
                                 stage="ingestion")
    return doc, kind


def _looks_jsonl(text: str) -> bool:
    try:
        json.loads(text)
        return False
    except json.JSONDecodeError:
        return True


# =====================================================================================================
# type inference
# =====================================================================================================
@dataclass
class ColumnProfile:
    original: str
    name: str
    candidates: list[str] = field(default_factory=lambda: list(TYPE_ORDER))
    non_null: int = 0
    nulls: int = 0
    distinct: set = field(default_factory=set)
    distinct_capped: bool = False
    leading_zero: bool = False
    money: int = 0
    plain_numeric: int = 0
    day_first: bool = False
    ambiguous_dates: int = 0
    min_v: Any = None
    max_v: Any = None
    samples: list[str] = field(default_factory=list)
    nonnumeric: dict = field(default_factory=dict)       # short non-numeric tokens -> count (placeholder detection)
    numeric_like: int = 0
    extra_nulls: set = field(default_factory=set)        # column-specific placeholder codes treated as NULL

    def observe(self, raw: Any, null_tokens: set[str]) -> None:
        if raw is None or (isinstance(raw, str) and (raw.strip() in null_tokens or raw.strip() in self.extra_nulls)):
            self.nulls += 1
            return
        self.non_null += 1
        if isinstance(raw, bool):
            v = "true" if raw else "false"
        elif isinstance(raw, (int, float)):
            v = repr(raw) if isinstance(raw, float) else str(raw)
        else:
            v = str(raw).strip()
        if len(self.distinct) < 10000:
            self.distinct.add(v)
        else:
            self.distinct_capped = True
        if len(self.samples) < 5 and v not in self.samples:
            self.samples.append(v)
        if _DEC_RE.match(v) or _MONEY_RE.match(v):
            self.numeric_like += 1
        elif len(v) <= 3 and len(self.nonnumeric) <= 3:
            self.nonnumeric[v] = self.nonnumeric.get(v, 0) + 1
        self.candidates = [t for t in self.candidates if _fits(t, v, self)]

    @property
    def inferred(self) -> str:
        if self.non_null == 0:
            return "text"
        for t in self.candidates:
            if t == "boolean" and not self._bool_ok():
                continue
            if t in ("smallint", "integer", "bigint", "numeric", "double precision") and (
                    self.leading_zero or (self.money and self.plain_numeric)):
                continue
            return t
        return "text"

    def _bool_ok(self) -> bool:
        lowered = {d.lower() for d in self.distinct}
        if lowered <= {"0", "1"} or lowered <= {"y", "n", "t", "f"}:
            return bool(BOOL_NAME_HINT.search(self.name))   # bare 0/1 or single letters need a name hint
        return True


def _fits(t: str, v: str, p: ColumnProfile) -> bool:
    lv = v.lower()
    if t == "boolean":
        return lv in BOOL_TRUE or lv in BOOL_FALSE or lv in ("0", "1")
    if t in ("smallint", "integer", "bigint"):
        if not _INT_RE.match(v):
            return False
        digits = v.lstrip("+-")
        if len(digits) > 1 and digits.startswith("0"):
            p.leading_zero = True
        if len(digits) > 18:
            return False
        n = int(v)
        return {"smallint": 32767, "integer": 2147483647, "bigint": 9223372036854775807}[t] >= abs(n)
    if t == "numeric":
        if _DEC_RE.match(v):
            if "e" not in lv:
                p.plain_numeric += 1
                digits = v.lstrip("+-").split(".")[0]
                if len(digits) > 1 and digits.startswith("0"):
                    p.leading_zero = True
                return len(v) <= 40
            return False
        if _MONEY_RE.match(v):
            p.money += 1
            return True
        return False
    if t == "double precision":
        try:
            float(v)
            return lv not in ("nan", "inf", "-inf", "infinity") and bool(re.search(r"\d", v))
        except ValueError:
            return False
    if t == "date":
        return _parse_date(v, p) is not None
    if t == "timestamp":
        return bool(_ISO_TS.match(v)) and _safe_ts(v) is not None
    if t == "timestamptz":
        return bool(_ISO_TSTZ.match(v)) and _safe_ts(v) is not None
    if t == "uuid":
        return bool(_UUID_RE.match(v))
    if t == "jsonb":
        if v[:1] in "[{":
            try:
                json.loads(v)
                return True
            except ValueError:
                return False
        return False
    return t == "text"


def _parse_date(v: str, p: ColumnProfile | None = None) -> dt.date | None:
    try:
        if _ISO_DATE.match(v):
            return dt.date.fromisoformat(v)
        m = _SLASH_DATE.match(v)
        if not m:
            return None
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if p is not None:
            if a > 12:
                p.day_first = True
            elif a <= 12 and b <= 12 and a != b:
                p.ambiguous_dates += 1
        day_first = p.day_first if p is not None else False
        return dt.date(y, a, b) if not day_first and a <= 12 else dt.date(y, b, a)
    except ValueError:
        return None


def _safe_ts(v: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None


def convert(value: Any, pg_type: str, null_tokens: set[str], day_first: bool = False) -> Any:
    """Convert one raw value to the Python object asyncpg COPY expects. Raises ValueError if not castable."""
    if value is None or (isinstance(value, str) and value.strip() in null_tokens):
        return None
    v = value if not isinstance(value, str) else value.strip()
    if pg_type == "text":
        return v if isinstance(v, str) else (json.dumps(v) if isinstance(v, (dict, list)) else str(v).lower()
                                             if isinstance(v, bool) else str(v))
    if isinstance(v, bool):
        if pg_type == "boolean":
            return v
        raise ValueError("boolean")
    s = str(v)
    ls = s.lower()
    if pg_type == "boolean":
        if ls in BOOL_TRUE or ls == "1":
            return True
        if ls in BOOL_FALSE or ls == "0":
            return False
        raise ValueError("boolean")
    if pg_type in ("smallint", "integer", "bigint"):
        if isinstance(v, float) and v.is_integer():
            s = str(int(v))
        if not _INT_RE.match(s):
            raise ValueError("integer")
        n = int(s)
        if abs(n) > {"smallint": 32767, "integer": 2147483647, "bigint": 9223372036854775807}[pg_type]:
            raise ValueError("out of range")
        return n
    if pg_type == "numeric":
        s2 = re.sub(r"[$€£₹¥,\s]", "", s) if _MONEY_RE.match(s) else s
        try:
            return decimal.Decimal(s2)
        except decimal.InvalidOperation:
            raise ValueError("numeric") from None
    if pg_type == "double precision":
        return float(s)
    if pg_type == "date":
        if _ISO_DATE.match(s):
            return dt.date.fromisoformat(s)
        m = _SLASH_DATE.match(s)
        if not m:
            raise ValueError("date")
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return dt.date(y, b, a) if day_first else dt.date(y, a, b)
    if pg_type in ("timestamp", "timestamptz"):
        ts = _safe_ts(s) if (_ISO_TS.match(s) or _ISO_TSTZ.match(s) or _ISO_DATE.match(s)) else None
        if ts is None:
            raise ValueError("timestamp")
        if pg_type == "timestamp":
            if ts.tzinfo is not None:
                raise ValueError("timestamp with zone")
            return ts
        return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
    if pg_type == "uuid":
        return uuid.UUID(s)
    if pg_type == "jsonb":
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        json.loads(s)
        return s
    raise ValueError(pg_type)


# =====================================================================================================
# analysis (preview)
# =====================================================================================================
@dataclass
class Analysis:
    filename: str
    sha256: str
    size_bytes: int
    fmt: str
    encoding: str | None
    delimiter: str | None
    header: bool
    columns: list[ColumnProfile]
    rows_read: int
    ragged_rows: list[int]
    warnings: list[str]
    path: Path

    def column_types(self) -> dict[str, str]:
        return {c.name: c.inferred for c in self.columns}


def _iter_rows(a: Analysis) -> Iterator[tuple[int, list[Any] | None]]:
    """Yield (row_number, values aligned to columns) - None for ragged rows that must be rejected."""
    n = len(a.columns)
    if a.fmt == "csv":
        with a.path.open("r", encoding=a.encoding or "utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=a.delimiter or ",")
            if a.header:
                next(reader, None)
            rownum = 1 if a.header else 0
            for row in reader:
                rownum += 1
                if not row or all(not c.strip() for c in row):
                    continue
                if len(row) < n:
                    row = row + [None] * (n - len(row))      # tolerated: trailing missing cells -> NULL
                elif len(row) > n:
                    if all(not c.strip() for c in row[n:]):
                        row = row[:n]
                    else:
                        yield rownum, None
                        continue
                yield rownum, row
    else:
        recs, _ = _json_records(a.path, a.encoding or "utf-8", a.fmt)
        keys = [c.original for c in a.columns]
        for i, rec in enumerate(recs, 1):
            flat = flatten_record(rec)
            yield i, [flat.get(k) for k in keys]


def analyze_file(path: Path, filename: str, settings: Settings | None = None, *, canonical_csv: bool = False) -> Analysis:
    s = settings or get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        if ext == ".sql":
            raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED,
                                     "SQL files are never ingested into the main database; use 'Upload SQL Schema "
                                     "(BYOD)' instead", stage="ingestion")
        raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, f"unsupported file type '{ext}' (CSV or JSON only)",
                                 stage="ingestion")
    size = path.stat().st_size
    if size > s.ingest_max_upload_mb * 1024 * 1024:
        raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, f"file exceeds {s.ingest_max_upload_mb} MB",
                                 stage="ingestion")
    if size == 0:
        raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, "file is empty", stage="ingestion")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    enc = detect_encoding(path)
    warnings: list[str] = []
    if enc == "latin-1":
        warnings.append("file is not valid UTF-8; decoded as Latin-1")
    fmt = ALLOWED_EXT[ext]
    nulls = s.null_tokens()
    delimiter, header = None, True
    if fmt == "csv":
        with path.open("r", encoding=enc, newline="") as f:
            sample = f.read(64 * 1024)
        delimiter, header = (",", True) if canonical_csv else detect_csv_dialect(sample)
        first = next(csv.reader(io.StringIO(sample), delimiter=delimiter), [])
        if header:
            trimmed = 0
            while first and not first[-1].strip():   # drop trailing unnamed columns (",,,,," exports)
                first = first[:-1]
                trimmed += 1
            if trimmed:
                warnings.append(f"{trimmed} empty unnamed column(s) at the end of each line were ignored")
        originals = first if header else [f"column_{i + 1}" for i in range(len(first))]
        if not header:
            warnings.append("no header row detected; columns named column_1..column_n")
    else:
        recs, kind = _json_records(path, enc, fmt)
        fmt = "jsonl" if kind == "json_lines" else "json"
        originals = []
        for r in recs:
            for k in flatten_record(r):
                if k not in originals:
                    originals.append(k)       # union of keys across records; absent keys -> NULL
        if not recs:
            raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, "JSON contains no records", stage="ingestion")
    if not originals:
        raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED, "no columns detected", stage="ingestion")
    names = dedupe([normalize_identifier(o, f"column_{i + 1}") for i, o in enumerate(originals)])
    for o, n in zip(originals, names):
        if str(o) != n:
            pass  # original names are preserved in metadata (dataset_columns.original_name)
    cols = [ColumnProfile(original=str(o), name=n) for o, n in zip(originals, names)]
    a = Analysis(filename=filename, sha256=h.hexdigest(), size_bytes=size, fmt=fmt, encoding=enc,
                 delimiter=delimiter, header=header, columns=cols, rows_read=0, ragged_rows=[], warnings=warnings,
                 path=path)
    for rownum, row in _iter_rows(a):
        a.rows_read += 1
        if row is None:
            a.ragged_rows.append(rownum)
            continue
        for c, v in zip(cols, row):
            c.observe(v, nulls)
    redo = []
    for i, c in enumerate(cols):
        placeholders = sum(c.nonnumeric.values())
        if (c.inferred == "text" and c.nonnumeric and len(c.nonnumeric) <= 2
                and c.numeric_like + placeholders == c.non_null and c.numeric_like > 0
                and placeholders <= 0.05 * c.non_null):
            redo.append((i, set(c.nonnumeric)))
    for i, codes in redo:                         # second pass only for those columns
        old = cols[i]
        fresh = ColumnProfile(original=old.original, name=old.name, extra_nulls=codes)
        for _rownum, row in _iter_rows(a):
            if row is not None:
                fresh.observe(row[i], nulls)
        if fresh.inferred != "text":
            cols[i] = fresh
            warnings.append(f"column '{old.name}': placeholder value(s) {sorted(codes)} in "
                            f"{sum(old.nonnumeric.values())} row(s) treated as empty (NULL) so the column stays "
                            f"{fresh.inferred}")
    a.columns = cols
    for c in cols:
        t = c.inferred
        if t == "date" and c.ambiguous_dates and not c.day_first:
            warnings.append(f"column '{c.name}': ambiguous day/month dates interpreted month-first")
        if t == "numeric" and c.money:
            warnings.append(f"column '{c.name}': currency symbols / thousands separators removed (consistent "
                            "across the column)")
        if c.leading_zero and t == "text":
            warnings.append(f"column '{c.name}': values with leading zeros kept as text")
    if a.ragged_rows:
        warnings.append(f"{len(a.ragged_rows)} ragged row(s) with extra values will be rejected "
                        f"(first: row {a.ragged_rows[0]})")
    return a


def column_stats(c: ColumnProfile, rows: int) -> dict[str, Any]:
    """Profiling stats stored for NL-to-SQL. Sample values only for low-cardinality text, PII-filtered."""
    t = c.inferred
    st: dict[str, Any] = {"null_ratio": round(c.nulls / rows, 4) if rows else 0.0,
                          "distinct_estimate": len(c.distinct), "distinct_capped": c.distinct_capped}
    vals = []
    for d in c.distinct:
        try:
            vals.append(convert(d, t, set(), c.day_first))
        except (ValueError, decimal.InvalidOperation):
            continue
    if vals and t not in ("boolean", "jsonb", "uuid", "text"):
        st["min"], st["max"] = str(min(vals)), str(max(vals))
    if t == "text" and 0 < len(c.distinct) <= 20 and not c.distinct_capped:
        safe = sorted(v for v in c.distinct if len(v) <= 40 and not _PII.scan(v).pii_detected
                      and not _PII.scan(v).secret_detected)
        st["sample_values"] = safe[:8]
    return st


# =====================================================================================================
# commit
# =====================================================================================================
@dataclass
class PreviewEntry:
    preview_id: str
    analysis: Analysis
    dataset_name: str
    table_name: str
    session_id: str | None
    created: float


class DatasetIngestionService:
    """Two-step API: preview (no writes) and commit (one transaction). Writes use admin credentials (trusted
    internal path only); reads by the app go through qs_app / qs_nlq_reader."""

    PREVIEW_TTL_S = 1800

    def __init__(self, settings: Settings | None = None, audit=None):
        self.settings = settings or get_settings()
        self.audit = audit
        self._previews: dict[str, PreviewEntry] = {}
        self.tmpdir = Path(tempfile.mkdtemp(prefix="qs_ingest_"))

    # ---------------------------------------------------------------- preview
    def _gc(self) -> None:
        now = time.time()
        for pid, e in list(self._previews.items()):
            if now - e.created > self.PREVIEW_TTL_S:
                self.discard(pid)

    def discard(self, preview_id: str) -> None:
        e = self._previews.pop(preview_id, None)
        if e is not None:
            e.analysis.path.unlink(missing_ok=True)

    def stage_bytes(self, data: bytes, filename: str) -> Path:
        if len(data) > self.settings.ingest_max_upload_mb * 1024 * 1024:
            raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED,
                                     f"file exceeds {self.settings.ingest_max_upload_mb} MB", stage="ingestion")
        p = self.tmpdir / f"up_{secrets.token_hex(8)}{Path(filename).suffix.lower()[:8]}"
        p.write_bytes(data)
        return p

    async def preview(self, path: Path, filename: str, dataset_name: str | None = None,
                      entity_name: str | None = None, session_id: str | None = None,
                      target_source_id: str | None = None, selection: str | None = None,
                      archive_member: str | None = None) -> dict[str, Any]:
        self._gc()
        from app.structured_formats import STRUCTURED_EXT, convert_file
        original_name = filename
        notes = []
        if Path(filename).suffix.lower() in STRUCTURED_EXT:
            original_path = path
            path = self.tmpdir / f"converted_{secrets.token_hex(8)}.csv"
            try:
                filename, notes = await asyncio.to_thread(convert_file, original_path, filename, path,
                    int(self.settings.ingest_max_upload_mb * 1024 * 1024), selection, archive_member)
            finally:
                original_path.unlink(missing_ok=True)
        try:
            a = await asyncio.to_thread(analyze_file, path, filename, self.settings, canonical_csv=(filename == "converted.csv" and bool(notes)))
            a.filename = original_name
            a.warnings.extend(notes)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        ds = normalize_identifier(dataset_name or Path(original_name).stem, "dataset")[:40]
        ent = normalize_identifier(entity_name, "data")[:30] if entity_name else None
        try:
            table = await self._table_for_session(ds, ent, session_id)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        pid = f"pv_{secrets.token_hex(10)}"
        self._previews[pid] = PreviewEntry(pid, a, ds, table, session_id, time.time())
        try:
            plan = await self._plan(a, table, target_source_id, session_id)
        except BaseException:
            self.discard(pid)
            raise
        return {"preview_id": pid, "filename": original_name, "sha256": a.sha256, "size_bytes": a.size_bytes,
                "detected_format": a.fmt, "encoding": a.encoding, "header": a.header,
                "delimiter": "(single column)" if a.delimiter == SINGLE_COLUMN_DELIM else a.delimiter,
                "rows_read": a.rows_read, "rows_rejected_ragged": len(a.ragged_rows),
                "dataset": ds, "target_table": f"{self.settings.ingest_target_schema}.{table}",
                "columns": [{"name": c.name, "original_name": c.original, "inferred_type": c.inferred,
                             "nullable": True, "null_ratio": round(c.nulls / max(a.rows_read, 1), 4),
                             "classification": _classify(c)} for c in a.columns],
                "plan": plan, "warnings": a.warnings, "provenance": [Provenance.POSTGRES_OBSERVED.value]
                if plan.get("existing_table") else []}

    async def _owner_of(self, conn, table: str) -> tuple[bool, str | None]:
        """(registered?, owner_session_id) of an existing qs_dynamic table."""
        row = await conn.fetchrow(
            "SELECT s.owner_session_id FROM qs_meta.dataset_tables t JOIN qs_meta.datasets d ON d.id = t.dataset_id "
            "JOIN qs_meta.data_sources s ON s.id = d.data_source_id WHERE t.schema_name = $1 AND t.table_name = $2",
            self.settings.ingest_target_schema, table)
        if row is not None:
            return True, row["owner_session_id"]
        exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL",
                                     f'{self.settings.ingest_target_schema}."{table}"')
        return bool(exists), None if not exists else "__unregistered__"

    async def _table_for_session(self, ds: str, ent: str | None, session_id: str | None) -> str:
        """Table names are global in qs_dynamic: if <dataset>__<entity> already belongs to ANOTHER session, this
        session gets its own name (<dataset>_<session tag>__<entity>) so uploads never touch someone else's data."""
        import asyncpg
        # Convention (U25): the table is named after the dataset (`shop` -> qs_dynamic.shop); an explicit entity name
        # adds a second table to the same dataset (`shop` + `returns` -> shop__returns).
        base = f"{ds}__{ent}" if ent else ds
        if s_single():
            return base
        try:
            conn = await asyncpg.connect(**self.settings.admin_dsn_kwargs, timeout=self.settings.db_connect_timeout_s)
        except Exception:  # noqa: BLE001 - commit re-checks ownership inside its transaction
            return base
        try:
            exists, owner = await self._owner_of(conn, base)
            if not exists or owner == session_id:
                return base
            tag = hashlib.sha256((session_id or "anon").encode()).hexdigest()[:6]
            return f"{ds[:30]}_{tag}" + (f"__{ent}" if ent else "")
        finally:
            await conn.close()

    async def _existing_columns(self, conn, table: str) -> dict[str, str] | None:
        rows = await conn.fetch(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod) AS t FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = $1 AND c.relname = $2 AND a.attnum > 0 AND NOT a.attisdropped",
            self.settings.ingest_target_schema, table)
        if not rows:
            return None
        return {r["attname"]: _norm_type(r["t"]) for r in rows if not r["attname"].startswith(HIDDEN_PREFIX)}

    async def _plan(self, a: Analysis, table: str, target_source_id: str | None, session_id: str | None,
                    conn=None) -> dict[str, Any]:
        import asyncpg
        own = conn is None
        if own:
            try:
                conn = await asyncpg.connect(**self.settings.admin_dsn_kwargs, timeout=self.settings.db_connect_timeout_s)
            except Exception:  # noqa: BLE001
                return {"action": "unknown", "note": "database unavailable for comparison"}
        try:
            existing = await self._existing_columns(conn, table)
            if existing is None:
                return {"action": "create_table", "existing_table": False, "new_columns": [c.name for c in a.columns]}
            return evolution_plan(existing, a, self.settings.ingest_allow_widening)
        finally:
            if own:
                await conn.close()

    # ---------------------------------------------------------------- commit
    async def commit(self, preview_id: str, session_id: str | None = None, allow_duplicate: bool = False,
                     allow_widening: bool = False) -> dict[str, Any]:
        import asyncpg
        e = self._previews.get(preview_id)
        if e is None:
            raise QuerySentinelError(ErrorCategory.NOT_FOUND, "preview expired or unknown", stage="ingestion")
        if e.session_id and e.session_id != session_id and not s_single():
            raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "preview belongs to another session",
                                     stage="ingestion")
        a, s = e.analysis, self.settings
        log = TraceLogger("dataset_ingestion")
        schema = s.ingest_target_schema
        widening_ok = allow_widening and s.ingest_allow_widening
        run_id = f"ing_{secrets.token_hex(8)}"
        conn = await asyncpg.connect(**s.admin_dsn_kwargs, timeout=s.db_connect_timeout_s)
        try:
            dup = await conn.fetchval("SELECT count(*) FROM qs_meta.ingestion_runs WHERE sha256 = $1 AND "
                                      "status = 'committed'", a.sha256)
            if dup and not allow_duplicate:
                raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED,
                                         "this exact file was already ingested; set allow_duplicate=true to load "
                                         "it again", stage="ingestion", details={"sha256": a.sha256})
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", f"{schema}.{e.table_name}")
                exists, owner = await self._owner_of(conn, e.table_name)
                if exists and owner != session_id and not s_single():
                    raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "target table belongs to another "
                                             "session; preview again", stage="ingestion")
                existing = await self._existing_columns(conn, e.table_name)
                plan = (evolution_plan(existing, a, widening_ok) if existing is not None
                        else {"action": "create_table", "new_columns": [c.name for c in a.columns]})
                if plan.get("conflicts"):
                    raise QuerySentinelError(ErrorCategory.SCHEMA_CONFLICT,
                                             "incoming data is incompatible with the existing table; nothing was "
                                             "written", stage="ingestion", details={"conflicts": plan["conflicts"]})
                types = a.column_types()
                target_types = dict(existing or {})
                changes: list[dict[str, Any]] = []
                qt = f"{quote_ident(schema)}.{quote_ident(e.table_name)}"
                if existing is None:
                    cols_sql = ", ".join(f"{quote_ident(c.name)} {types[c.name]}" for c in a.columns)
                    await conn.execute(f"CREATE TABLE {qt} ({cols_sql}, {quote_ident(LINEAGE_COLUMN)} text)")
                    target_types = dict(types)
                    changes.append({"change": "create_table", "table": e.table_name})
                else:
                    for col in plan.get("new_columns", []):
                        await conn.execute(f"ALTER TABLE {qt} ADD COLUMN {quote_ident(col)} {types[col]}")
                        target_types[col] = types[col]
                        changes.append({"change": "add_column", "column": col, "type": types[col]})
                    for w in plan.get("widenings", []):
                        await conn.execute(f"ALTER TABLE {qt} ALTER COLUMN {quote_ident(w['column'])} "
                                           f"TYPE {w['to']}")
                        target_types[w["column"]] = w["to"]
                        changes.append({"change": "widen_type", **w})
                # convert + COPY (pre-validated values; nothing from the file is executed as SQL)
                names = [c.name for c in a.columns]
                day_first = {c.name: c.day_first for c in a.columns}
                nulls = s.null_tokens()
                col_nulls = {c.name: nulls | c.extra_nulls for c in a.columns}
                records, rejected = [], list(a.ragged_rows)
                for rownum, row in _iter_rows(a):
                    if row is None:
                        continue
                    try:
                        records.append(tuple(convert(v, target_types[n], col_nulls[n], day_first[n])
                                             for n, v in zip(names, row)) + (run_id,))
                    except (ValueError, decimal.InvalidOperation):
                        rejected.append(rownum)
                ratio = len(rejected) / max(a.rows_read, 1)
                if ratio > s.ingest_max_reject_ratio:
                    raise QuerySentinelError(ErrorCategory.UPLOAD_REJECTED,
                                             f"rejected-row ratio {ratio:.2%} exceeds INGEST_MAX_REJECT_RATIO "
                                             f"({s.ingest_max_reject_ratio:.2%}); nothing was written",
                                             stage="ingestion", details={"rejected_rows_sample": rejected[:20]})
                if records:
                    await conn.copy_records_to_table(e.table_name, schema_name=schema, records=records,
                                                     columns=names + [LINEAGE_COLUMN])
                # ---- registry
                source_id, version = await self._register(conn, e, a, target_types, changes, session_id)
                await conn.execute(
                    "INSERT INTO qs_meta.ingestion_runs (id, data_source_id, dataset_id, filename, sha256, size_bytes,"
                    " status, rows_read, rows_loaded, rows_rejected, schema_changes, warnings) VALUES "
                    "($1,$2,$3,$4,$5,$6,'committed',$7,$8,$9,$10::jsonb,$11::jsonb)",
                    run_id, source_id, f"dset_{source_id}", a.filename[:200], a.sha256, a.size_bytes, a.rows_read,
                    len(records), len(rejected), json.dumps(changes), json.dumps(a.warnings))
                await conn.execute("INSERT INTO qs_meta.audit_log (data_source_id, session_id, event_type, details) "
                                   "VALUES ($1, $2, 'INGESTION_COMMITTED', $3::jsonb)", source_id, session_id,
                                   json.dumps({"run_id": run_id, "rows_loaded": len(records),
                                               "schema_changes": len(changes)}))
            await conn.execute(f"ANALYZE {qt}")
        except QuerySentinelError as exc:
            await self._record_failure(conn, run_id, a, exc)
            log.warning("ingestion rejected", error_category=exc.category.value, stage="ingestion")
            raise
        finally:
            await conn.close()
        self.discard(preview_id)
        from app.schema_inspector import schema_cache
        schema_cache.invalidate(source_id)
        log.data_source_id = source_id
        log.info("ingestion committed", stage="ingestion")
        return {"run_id": run_id, "data_source_id": source_id, "table": f"{schema}.{e.table_name}",
                "schema_version": version, "rows_read": a.rows_read, "rows_loaded": len(records),
                "rows_rejected": len(rejected), "rejected_rows_sample": rejected[:20], "schema_changes": changes,
                "warnings": a.warnings, "provenance": [Provenance.POSTGRES_OBSERVED.value]}

    async def _record_failure(self, conn, run_id: str, a: Analysis, exc: QuerySentinelError) -> None:
        try:
            await conn.execute(
                "INSERT INTO qs_meta.ingestion_runs (id, filename, sha256, size_bytes, status, rows_read, warnings) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb) ON CONFLICT (id) DO NOTHING",
                run_id, a.filename[:200], a.sha256, a.size_bytes, f"rejected:{exc.category.value}", a.rows_read,
                json.dumps([exc.message]))
        except Exception:  # noqa: BLE001 - audit of a failure must not mask the failure
            pass

    async def delete_source(self, source_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Delete an uploaded dataset: drop its tables in the dataset schema and remove its registry rows, in one
        transaction. Only `dynamic` sources can be deleted (never the demo database). The source row is kept with
        status 'deleted' so the audit history stays intact; the table name becomes free for a new upload and the
        same file may be uploaded again."""
        import asyncpg
        s = self.settings
        schema = s.ingest_target_schema
        conn = await asyncpg.connect(**s.admin_dsn_kwargs, timeout=s.db_connect_timeout_s)
        try:
            async with conn.transaction():
                src = await conn.fetchrow("SELECT id, kind, name, owner_session_id, status FROM qs_meta.data_sources "
                                          "WHERE id = $1 FOR UPDATE", source_id)
                if src is None or src["kind"] != "dynamic" or src["status"] == "deleted":
                    raise QuerySentinelError(ErrorCategory.SOURCE_NOT_FOUND, "unknown uploaded dataset",
                                             stage="dataset_delete")
                if src["owner_session_id"] != session_id and not s_single():
                    raise QuerySentinelError(ErrorCategory.SOURCE_FORBIDDEN, "dataset belongs to another session",
                                             stage="dataset_delete")
                tables = await conn.fetch(
                    "SELECT t.id, t.schema_name, t.table_name FROM qs_meta.dataset_tables t JOIN qs_meta.datasets d "
                    "ON d.id = t.dataset_id WHERE d.data_source_id = $1", source_id)
                for t in tables:
                    if t["schema_name"] != schema:          # never drop anything outside the dataset schema
                        continue
                    await conn.execute(f"DROP TABLE IF EXISTS {quote_ident(schema)}.{quote_ident(t['table_name'])}")
                ids = [t["id"] for t in tables]
                await conn.execute("DELETE FROM qs_meta.dataset_columns WHERE table_id = ANY($1::bigint[])", ids)
                await conn.execute("DELETE FROM qs_meta.dataset_tables WHERE id = ANY($1::bigint[])", ids)
                await conn.execute("DELETE FROM qs_meta.datasets WHERE data_source_id = $1", source_id)
                await conn.execute("UPDATE qs_meta.ingestion_runs SET status = 'deleted' WHERE data_source_id = $1 "
                                   "AND status = 'committed'", source_id)
                await conn.execute("UPDATE qs_meta.data_sources SET status = 'deleted' WHERE id = $1", source_id)
        finally:
            await conn.close()
        TraceLogger("dataset_ingestion").info("dataset deleted", stage="dataset_delete")
        return {"data_source_id": source_id, "name": src["name"], "status": "deleted",
                "tables_dropped": [t["table_name"] for t in tables]}

    async def _register(self, conn, e: PreviewEntry, a: Analysis, target_types: dict[str, str],
                        changes: list[dict], session_id: str | None) -> tuple[str, int]:
        schema = self.settings.ingest_target_schema
        row = await conn.fetchrow("SELECT t.id AS table_id, d.data_source_id FROM qs_meta.dataset_tables t "
                                  "JOIN qs_meta.datasets d ON d.id = t.dataset_id "
                                  "WHERE t.schema_name = $1 AND t.table_name = $2", schema, e.table_name)
        if row is None:
            ds_row = await conn.fetchrow("SELECT d.id, d.data_source_id FROM qs_meta.datasets d JOIN "
                                         "qs_meta.data_sources s ON s.id = d.data_source_id WHERE d.name = $1 AND "
                                         "s.status = 'ready' AND ($3 OR s.owner_session_id IS NOT DISTINCT FROM $2)",
                                         e.dataset_name, session_id, s_single())
            if ds_row is None:
                source_id = f"dyn_{secrets.token_hex(6)}"
                await conn.execute("INSERT INTO qs_meta.data_sources (id, kind, name, owner_session_id, status, "
                                   "schema_name, role_name, schema_version) VALUES ($1,'dynamic',$2,$3,'ready',$4,"
                                   "'qs_nlq_reader',1)", source_id, e.dataset_name, session_id, schema)
                dataset_id = f"dset_{source_id}"
                await conn.execute("INSERT INTO qs_meta.datasets (id, data_source_id, name, original_name) VALUES "
                                   "($1,$2,$3,$4)", dataset_id, source_id, e.dataset_name, a.filename[:200])
                version = 1
            else:
                source_id, dataset_id = ds_row["data_source_id"], ds_row["id"]
                version = await conn.fetchval("UPDATE qs_meta.data_sources SET schema_version = schema_version + 1 "
                                              "WHERE id = $1 RETURNING schema_version", source_id)
            table_id = await conn.fetchval(
                "INSERT INTO qs_meta.dataset_tables (dataset_id, schema_name, table_name, original_name) "
                "VALUES ($1,$2,$3,$4) RETURNING id", dataset_id, schema, e.table_name, a.filename[:200])
        else:
            source_id, table_id = row["data_source_id"], row["table_id"]
            version = await conn.fetchval("SELECT schema_version FROM qs_meta.data_sources WHERE id = $1", source_id)
            if changes:
                version = await conn.fetchval("UPDATE qs_meta.data_sources SET schema_version = schema_version + 1 "
                                              "WHERE id = $1 RETURNING schema_version", source_id)
        await conn.execute("UPDATE qs_meta.dataset_tables SET row_count = (SELECT count(*) FROM "
                           f"{quote_ident(schema)}.{quote_ident(e.table_name)}) WHERE id = $1", table_id)
        for c in a.columns:
            await conn.execute(
                "INSERT INTO qs_meta.dataset_columns (table_id, column_name, original_name, data_type, nullable, "
                "added_in_version, stats) VALUES ($1,$2,$3,$4,TRUE,$5,$6::jsonb) ON CONFLICT (table_id, column_name) "
                "DO UPDATE SET data_type = EXCLUDED.data_type, stats = EXCLUDED.stats",
                table_id, c.name, c.original[:200], target_types[c.name], version,
                json.dumps(column_stats(c, a.rows_read)))
        if changes:
            await conn.execute("INSERT INTO qs_meta.schema_versions (data_source_id, version, diff) VALUES "
                               "($1,$2,$3::jsonb) ON CONFLICT (data_source_id, version) DO UPDATE SET diff = "
                               "qs_meta.schema_versions.diff || EXCLUDED.diff",
                               source_id, version, json.dumps({"table": e.table_name, "changes": changes}))
        return source_id, version


def s_single() -> bool:
    return get_settings().single_user


def _norm_type(t: str) -> str:
    t = t.lower()
    if t.startswith("numeric"):
        return "numeric"
    if t.startswith("timestamp") and "with time zone" in t and "without" not in t:
        return "timestamptz"
    if t.startswith("timestamp"):
        return "timestamp"
    if t.startswith("character varying") or t.startswith("character") or t == "text":
        return "text"
    return t


def _classify(c: ColumnProfile) -> str:
    t = c.inferred
    if t in ("date", "timestamp", "timestamptz"):
        return "temporal"
    if t in ("smallint", "integer", "bigint", "numeric", "double precision"):
        return "identifier" if c.name == "id" or c.name.endswith("_id") else "measure"
    if t == "text" and 0 < len(c.distinct) <= 50 and not c.distinct_capped:
        return "category"
    return {"boolean": "flag", "uuid": "identifier", "jsonb": "document"}.get(t, "text")


def evolution_plan(existing: dict[str, str], a: Analysis, allow_widening: bool) -> dict[str, Any]:
    """Compare incoming data with an existing table. Additive only."""
    incoming = a.column_types()
    new_cols = [c for c in incoming if c not in existing]
    missing = [c for c in existing if c not in incoming]
    conflicts, widenings, castable = [], [], []
    nulls = get_settings().null_tokens()
    for c in a.columns:
        if c.name not in existing:
            continue
        et, it = existing[c.name], c.inferred
        if et == it or it == "text" and et == "text":
            continue
        # does every observed value cast to the existing type?
        bad = []
        for v in list(c.distinct)[:10000]:
            try:
                convert(v, et, nulls | c.extra_nulls, c.day_first)
            except (ValueError, decimal.InvalidOperation):
                bad.append(v)
                if len(bad) >= 3:
                    break
        if not bad:
            castable.append(c.name)
            continue
        if it in WIDENING.get(et, []):
            if allow_widening:
                widenings.append({"column": c.name, "from": et, "to": it})
                continue
            conflicts.append({"column": c.name, "existing_type": et, "incoming_type": it,
                              "reason": "wider type needed (widening disabled: set INGEST_ALLOW_WIDENING=true and "
                                        "allow_widening in the request)", "sample_values": bad[:3],
                              "row_numbers": _rows_with(a, c.name, set(bad))})
        else:
            conflicts.append({"column": c.name, "existing_type": et, "incoming_type": it,
                              "reason": "unrelated type", "sample_values": [v[:40] for v in bad[:3]],
                              "row_numbers": _rows_with(a, c.name, set(bad))})
    suggestions = []
    for m in missing:
        close = difflib.get_close_matches(m, new_cols, n=1, cutoff=0.75)
        if close:
            suggestions.append(f"column '{close[0]}' looks like a rename of '{m}' (suggestion only; no rename is "
                               "performed)")
    return {"action": "evolve" if (new_cols or widenings) else "append", "existing_table": True,
            "new_columns": new_cols, "missing_columns_filled_null": missing, "castable_columns": castable,
            "widenings": widenings, "conflicts": conflicts, "suggestions": suggestions}


def _rows_with(a: Analysis, col: str, bad: set[str], limit: int = 5) -> list[int]:
    idx = [c.name for c in a.columns].index(col)
    out = []
    for rownum, row in _iter_rows(a):
        if row is not None and row[idx] is not None and str(row[idx]).strip() in bad:
            out.append(rownum)
            if len(out) >= limit:
                break
    return out
