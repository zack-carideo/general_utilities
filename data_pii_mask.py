"""
data_masker.py — Generalizable data masking utility for tabular data,
with PII-aware masking for categorical/string fields.

Guarantees:
    • Row-wise uniqueness     — distinct rows stay distinct after masking
                                (1:1 per-column value mapping)
    • Attribute relationships — every occurrence of value v in column c maps
                                to the same masked token (functional
                                dependencies, equality joins, co-occurrence
                                patterns all preserved)
    • Column dtypes           — numerics stay numeric, datetimes stay datetime,
                                strings stay strings
    • PII format preservation — emails look like emails, phones look like
                                phones, SSNs look like SSNs (configurable)

Each fitted instance carries a unique salt, so the same DataFrame fitted
twice produces two different masks. Pass `salt=...` for reproducibility.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import string
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# PII detection patterns (anchored — used for column-level detection).
# Ordered most-specific → least-specific. Detection short-circuits on the
# first pattern whose match rate clears the threshold, so this order
# prevents e.g. SSNs / credit cards being misclassified as phones.
# ============================================================
PII_PATTERNS_ORDERED = [
    ("credit_card", re.compile(r"^(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"
                               r"|3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5})$")),
    ("ssn",         re.compile(r"^\d{3}-\d{2}-\d{4}$")),
    ("iban",        re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{8,30}$")),
    ("ipv4",        re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")),
    ("email",       re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")),
    ("url",         re.compile(r"^https?://[^\s]+$")),
    ("phone",       re.compile(r"^(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")),
    ("zip_us",      re.compile(r"^\d{5}(-\d{4})?$")),
]
PII_PATTERNS = dict(PII_PATTERNS_ORDERED)  # back-compat lookup

# Loose patterns used for *in-text* scrubbing within free-text columns
PII_INLINE = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "phone":       re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "ipv4":        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url":         re.compile(r"https?://[^\s]+"),
}

# Small deterministic name pool for `pii_columns={"col": "name"}`.
# Tiny on purpose — fake names should look plausible without ever mapping
# back to a real person. The pool size sets a uniqueness ceiling per fit.
_FAKE_FIRST = [
    "Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Quinn", "Avery",
    "Drew", "Sage", "Reese", "Rowan", "Hayden", "Parker", "Logan", "Emerson",
    "Skyler", "Finley", "Kendall", "Blake", "Cameron", "Dakota", "Elliot",
    "Frankie", "Gray", "Harper", "Indigo", "Jamie", "Kai", "Lane",
]
_FAKE_LAST = [
    "Stone", "Rivers", "Hayes", "Brooks", "Carter", "Reed", "Wells", "Mason",
    "Quinn", "Ellis", "Ward", "Pierce", "Vaughn", "Cole", "Park", "Hill",
    "Tate", "Marsh", "Lane", "Frost", "Chen", "Patel", "Khan", "Singh",
    "Garcia", "Lopez", "Nguyen", "Kim", "Okafor", "Diaz",
]


class DataMasker:
    """Deterministic, dtype-aware tabular data masker with PII support.

    Parameters
    ----------
    salt : str, optional
        Hex string used to key the HMAC. If omitted, a cryptographically
        random 128-bit salt is generated — this is what makes each fit
        produce a "unique mask for any dataset".
    preserve_dtypes : bool, default True
        If True, masked numerics stay numeric, datetimes stay datetime.
    numeric_strategy : {"tokenize", "shift_scale"}, default "tokenize"
        "tokenize"    — map each unique value to a deterministic float
                        in [0, 1). Destroys order/correlation but 1:1.
        "shift_scale" — deterministic affine a*x + b per column. Preserves
                        order and correlations (EDA-friendly).
    datetime_jitter_days : int, default 3650
        Half-range of per-column datetime shift, in days.
    passthrough_cols : list[str], optional
        Columns to leave unchanged, overrides detection.
    force_mask_cols : list[str], optional
        Columns to always mask, overrides categorical exclusion / text detection.
    exclude_all_categoricals : bool, default False
        Skip every Categorical / object / string column.

    PII-specific parameters (new)
    -----------------------------
    pii_detection : bool, default True
        Auto-detect PII type per string/categorical column by regex match rate.
    pii_strategy : {"format_preserve", "tokenize", "redact"}, default "format_preserve"
        "format_preserve" — keep structure (email@x.tld stays email@x.tld),
                            replace identifying parts deterministically.
        "tokenize"        — replace with `<EMAIL_a3f7b2>`-style tokens.
        "redact"          — replace with fixed `<EMAIL>` etc. WARNING:
                            destroys row-wise uniqueness for that column.
    pii_columns : dict[str, str], optional
        Explicit `{"col": "email" | "phone" | "ssn" | "credit_card" | "ipv4"
                  | "url" | "iban" | "zip_us" | "name" | "address"}`.
        Overrides auto-detection. Use for columns auto-detection misses
        (notably "name" and "address", which can't be reliably regex-detected).
    scrub_text_pii : bool, default True
        For columns kept as passthrough (natural-text), still scrub inline
        PII patterns (emails, phones, SSNs, cards, IPs, URLs).
    pii_detection_threshold : float, default 0.8
        Fraction of non-null cells in a column that must match a pattern
        for the column to be classified as that PII type.
    """

    _NUMERIC_STRATEGIES = {"tokenize", "shift_scale"}
    _PII_STRATEGIES = {"format_preserve", "tokenize", "redact"}
    _VALID_PII_TYPES = (set(PII_PATTERNS.keys()) | {"name", "address"})

    def __init__(
        self,
        *,
        salt: Optional[str] = None,
        preserve_dtypes: bool = True,
        numeric_strategy: str = "tokenize",
        datetime_jitter_days: int = 3650,
        passthrough_cols: Optional[list] = None,
        force_mask_cols: Optional[list] = None,
        exclude_all_categoricals: bool = False,
        # PII params
        pii_detection: bool = True,
        pii_strategy: str = "format_preserve",
        pii_columns: Optional[dict] = None,
        scrub_text_pii: bool = True,
        pii_detection_threshold: float = 0.8,
        # text-detection params (unchanged behaviour from prior version)
        text_uniqueness_ratio: float = 0.85,
        text_avg_length: float = 20.0,
        text_avg_words: float = 3.0,
    ):
        if numeric_strategy not in self._NUMERIC_STRATEGIES:
            raise ValueError(f"numeric_strategy must be in {self._NUMERIC_STRATEGIES}")
        if pii_strategy not in self._PII_STRATEGIES:
            raise ValueError(f"pii_strategy must be in {self._PII_STRATEGIES}")
        if pii_columns:
            bad = set(pii_columns.values()) - self._VALID_PII_TYPES
            if bad:
                raise ValueError(f"unknown pii types: {bad}. valid: {self._VALID_PII_TYPES}")

        self.salt = salt or secrets.token_hex(16)
        self.preserve_dtypes = preserve_dtypes
        self.numeric_strategy = numeric_strategy
        self.datetime_jitter_days = datetime_jitter_days
        self.passthrough_cols = set(passthrough_cols or [])
        self.force_mask_cols = set(force_mask_cols or [])
        self.exclude_all_categoricals = exclude_all_categoricals
        self.pii_detection = pii_detection
        self.pii_strategy = pii_strategy
        self.pii_columns = dict(pii_columns or {})
        self.scrub_text_pii = scrub_text_pii
        self.pii_detection_threshold = pii_detection_threshold
        self.text_uniqueness_ratio = text_uniqueness_ratio
        self.text_avg_length = text_avg_length
        self.text_avg_words = text_avg_words

        self._fitted = False
        self._plan: dict[str, dict] = {}  # column -> treatment dict

    # ============================================================
    # HMAC primitives (deterministic per salt + column)
    # ============================================================
    def _hmac_bytes(self, col_key: str, value) -> bytes:
        key = (self.salt + "::" + col_key).encode("utf-8")
        msg = repr(value).encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).digest()

    def _hmac_int(self, col_key: str, value, mod: int) -> int:
        return int.from_bytes(self._hmac_bytes(col_key, value)[:8], "big") % mod

    def _hmac_float_unit(self, col_key: str, value) -> float:
        """Deterministic float in [0, 1)."""
        return self._hmac_int(col_key, value, 2**53) / 2**53

    def _hmac_hex(self, col_key: str, value, n_chars: int = 12) -> str:
        return self._hmac_bytes(col_key, value).hex()[:n_chars]

    def _hmac_alphanum(self, col_key: str, value, length: int) -> str:
        """Deterministic alphanumeric string of given length."""
        b = self._hmac_bytes(col_key, value)
        alpha = string.ascii_lowercase + string.digits
        # Stretch by chaining if length exceeds 64 hex chars
        out, i = [], 0
        while len(out) < length:
            for byte in b:
                out.append(alpha[byte % len(alpha)])
                if len(out) >= length:
                    break
            i += 1
            b = hashlib.sha256(b + bytes([i])).digest()
        return "".join(out)

    def _hmac_digits(self, col_key: str, value, length: int) -> str:
        """Deterministic digit string of given length."""
        b = self._hmac_bytes(col_key, value)
        out, i = [], 0
        while len(out) < length:
            for byte in b:
                out.append(str(byte % 10))
                if len(out) >= length:
                    break
            i += 1
            b = hashlib.sha256(b + bytes([i])).digest()
        return "".join(out)

    # ============================================================
    # PII detection
    # ============================================================
    def _detect_pii_type(self, series: pd.Series) -> Optional[str]:
        """Return PII type name if >= threshold of non-null cells match.
        Patterns are tried in priority order (most specific first); the
        first one to clear the threshold wins."""
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return None
        for pii_type, pat in PII_PATTERNS_ORDERED:
            rate = non_null.str.match(pat).sum() / len(non_null)
            if rate >= self.pii_detection_threshold:
                return pii_type
        return None

    # ============================================================
    # PII maskers — each preserves 1:1 mapping under "format_preserve" and
    # "tokenize"; "redact" collapses to a constant per type (loses 1:1).
    # ============================================================
    def _mask_pii(self, col: str, value, pii_type: str):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        s = str(value)
        if self.pii_strategy == "redact":
            return f"<{pii_type.upper()}>"
        if self.pii_strategy == "tokenize":
            return f"<{pii_type.upper()}_{self._hmac_hex(col, s, 8)}>"
        # format_preserve
        fn = getattr(self, f"_fp_{pii_type}", self._fp_generic)
        return fn(col, s)

    def _fp_generic(self, col, s):
        # Fallback: replace letters/digits with deterministic alphanum,
        # keep separators
        out, idx = [], 0
        token = self._hmac_alphanum(col, s, sum(c.isalnum() for c in s))
        for ch in s:
            if ch.isalnum():
                out.append(token[idx])
                idx += 1
            else:
                out.append(ch)
        return "".join(out)

    def _fp_email(self, col, s):
        if "@" not in s:
            return self._fp_generic(col, s)
        local, _, domain = s.partition("@")
        local_m = self._hmac_alphanum(col + ":local", s, max(len(local), 3))
        if "." in domain:
            host, _, tld = domain.rpartition(".")
            host_m = self._hmac_alphanum(col + ":host", s, max(len(host), 3))
            return f"{local_m}@{host_m}.{tld}"
        return f"{local_m}@{self._hmac_alphanum(col + ':host', s, len(domain))}"

    def _fp_phone(self, col, s):
        digits = re.sub(r"\D", "", s)
        if not digits:
            return s
        mdigits = self._hmac_digits(col, s, len(digits))
        out, idx = [], 0
        for ch in s:
            if ch.isdigit():
                out.append(mdigits[idx])
                idx += 1
            else:
                out.append(ch)
        return "".join(out)

    def _fp_ssn(self, col, s):
        return self._fp_phone(col, s)

    def _fp_credit_card(self, col, s):
        # PCI-style: keep last 4, mask the rest deterministically
        digits = re.sub(r"\D", "", s)
        if len(digits) <= 4:
            return "X" * len(digits)
        masked = self._hmac_digits(col, s, len(digits) - 4) + digits[-4:]
        # NB: replacing prefix with deterministic digits keeps 1:1 across the
        # full string, but two cards sharing the same last-4 will share that
        # suffix. That matches PCI display conventions and the rest of the
        # number still differs. Use "tokenize" if strict 1:1 visibility needed.
        out, idx = [], 0
        for ch in s:
            if ch.isdigit():
                out.append(masked[idx])
                idx += 1
            else:
                out.append(ch)
        return "".join(out)

    def _fp_ipv4(self, col, s):
        parts = s.split(".")
        if len(parts) != 4:
            return self._fp_generic(col, s)
        # Deterministic octets in [1, 254]
        masked = [str(1 + self._hmac_int(col + f":oct{i}", s, 254)) for i in range(4)]
        return ".".join(masked)

    def _fp_url(self, col, s):
        # Keep scheme + tld structure, hash the rest
        m = re.match(r"^(https?://)([^/]+)(/.*)?$", s)
        if not m:
            return self._fp_generic(col, s)
        scheme, host, path = m.group(1), m.group(2), m.group(3) or ""
        if "." in host:
            sub, _, tld = host.rpartition(".")
            host_m = self._hmac_alphanum(col + ":host", s, max(len(sub), 4)) + "." + tld
        else:
            host_m = self._hmac_alphanum(col + ":host", s, len(host))
        path_m = ("/" + self._hmac_alphanum(col + ":path", s, max(len(path) - 1, 4))) if path else ""
        return scheme + host_m + path_m

    def _fp_iban(self, col, s):
        # Keep country code + 2-digit checksum, mask the BBAN
        if len(s) < 4:
            return self._fp_generic(col, s)
        country, check, bban = s[:2], s[2:4], s[4:]
        return country + check + self._hmac_alphanum(col + ":bban", s, len(bban)).upper()

    def _fp_zip_us(self, col, s):
        return self._fp_phone(col, s)

    def _fp_name(self, col, s):
        # Deterministic fake first + last from small pool
        first = _FAKE_FIRST[self._hmac_int(col + ":first", s, len(_FAKE_FIRST))]
        last = _FAKE_LAST[self._hmac_int(col + ":last", s, len(_FAKE_LAST))]
        # If original looked like single token, return just first
        if " " not in s.strip():
            return first
        return f"{first} {last}"

    def _fp_address(self, col, s):
        # Synthesize a plausible US-style address deterministically
        num = 100 + self._hmac_int(col + ":num", s, 9900)
        street_word = self._hmac_alphanum(col + ":street", s, 6).capitalize()
        suffix = ["St", "Ave", "Rd", "Blvd", "Ln", "Way"][self._hmac_int(col + ":sfx", s, 6)]
        return f"{num} {street_word} {suffix}"

    # ============================================================
    # Inline (within-text) PII scrubbing for natural-text columns
    # ============================================================
    def _scrub_inline(self, col: str, text):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return text
        s = str(text)
        for pii_type, pat in PII_INLINE.items():
            def _repl(match, _t=pii_type):
                return self._mask_pii(col + ":inline", match.group(0), _t)
            s = pat.sub(_repl, s)
        return s

    # ============================================================
    # Per-column maskers — generic (non-PII) string/numeric/datetime/bool
    # ============================================================
    def _mask_string_value(self, col, value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return value
        return self._hmac_hex(col, value, 12)

    def _mask_numeric_value(self, col, value, params):
        if pd.isna(value):
            return value
        if self.numeric_strategy == "tokenize":
            return self._hmac_float_unit(col, value)
        # shift_scale: a*x + b
        return params["scale"] * float(value) + params["shift"]

    def _mask_datetime_value(self, col, value, params):
        if pd.isna(value):
            return value
        return pd.Timestamp(value) + pd.Timedelta(days=params["shift_days"])

    def _mask_bool_value(self, col, value, params):
        if pd.isna(value):
            return value
        return bool(value) ^ params["flip"]

    # ============================================================
    # Natural-text detection (free-text passthrough heuristic)
    # ============================================================
    def _looks_like_natural_text(self, series: pd.Series) -> bool:
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return False
        n_unique = non_null.nunique()
        uniq_ratio = n_unique / len(non_null)
        avg_len = non_null.str.len().mean()
        avg_words = non_null.str.split().str.len().mean()
        return (
            uniq_ratio >= self.text_uniqueness_ratio
            and avg_len >= self.text_avg_length
            and avg_words >= self.text_avg_words
        )

    # ============================================================
    # Plan / fit
    # ============================================================
    def _is_categorical_like(self, series: pd.Series) -> bool:
        return (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

    def _plan_column(self, col: str, series: pd.Series) -> dict:
        # ---- explicit user overrides (precedence) ----
        if col in self.passthrough_cols:
            return {"kind": "passthrough", "reason": "user_specified"}

        if col in self.pii_columns:
            return {
                "kind": "pii",
                "pii_type": self.pii_columns[col],
                "reason": "user_specified",
            }

        if col in self.force_mask_cols:
            # Even when forced, still attempt PII detection for nicer output
            pii_type = self._detect_pii_type(series) if self.pii_detection and self._is_categorical_like(series) else None
            if pii_type:
                return {"kind": "pii", "pii_type": pii_type, "reason": "force_mask+autodetect"}
            return self._plan_by_dtype(col, series, forced=True)

        # ---- PII auto-detection on string/categorical columns ----
        if self.pii_detection and self._is_categorical_like(series):
            pii_type = self._detect_pii_type(series)
            if pii_type:
                return {"kind": "pii", "pii_type": pii_type, "reason": "auto_detected"}

        # ---- categorical exclusion flag ----
        if self.exclude_all_categoricals and self._is_categorical_like(series):
            return {"kind": "passthrough", "reason": "categorical_excluded",
                    "scrub_inline": self.scrub_text_pii}

        # ---- natural-text auto-detection ----
        if self._is_categorical_like(series) and self._looks_like_natural_text(series):
            return {"kind": "passthrough", "reason": "natural_text_detected",
                    "scrub_inline": self.scrub_text_pii}

        # ---- mask by dtype ----
        return self._plan_by_dtype(col, series)

    def _plan_by_dtype(self, col: str, series: pd.Series, forced: bool = False) -> dict:
        if pd.api.types.is_bool_dtype(series):
            return {"kind": "bool", "flip": bool(self._hmac_int(col, "flip", 2))}
        if pd.api.types.is_datetime64_any_dtype(series):
            half = self.datetime_jitter_days
            shift = self._hmac_int(col, "dshift", 2 * half + 1) - half
            return {"kind": "datetime", "shift_days": shift}
        if pd.api.types.is_numeric_dtype(series):
            # shift_scale params (only used if numeric_strategy=="shift_scale")
            scale = 0.5 + self._hmac_float_unit(col, "scale") * 1.5  # in [0.5, 2.0)
            shift = (self._hmac_float_unit(col, "shift") - 0.5) * 2.0 * float(
                series.abs().mean() if series.notna().any() else 1.0
            )
            return {"kind": "numeric", "scale": scale, "shift": shift}
        # default: string
        return {"kind": "string", "forced": forced}

    def fit(self, df: pd.DataFrame) -> "DataMasker":
        self._plan = {col: self._plan_column(col, df[col]) for col in df.columns}
        self._fitted = True
        return self

    # ============================================================
    # Transform
    # ============================================================
    def _apply_column(self, col: str, series: pd.Series) -> pd.Series:
        plan = self._plan[col]
        kind = plan["kind"]

        if kind == "passthrough":
            if plan.get("scrub_inline"):
                return series.map(lambda v: self._scrub_inline(col, v))
            return series.copy()

        if kind == "pii":
            pii_type = plan["pii_type"]
            return series.map(lambda v: self._mask_pii(col, v, pii_type))

        if kind == "bool":
            return series.map(lambda v: self._mask_bool_value(col, v, plan))

        if kind == "datetime":
            return series.map(lambda v: self._mask_datetime_value(col, v, plan))

        if kind == "numeric":
            out = series.map(lambda v: self._mask_numeric_value(col, v, plan))
            if self.preserve_dtypes:
                try:
                    out = out.astype(series.dtype)
                except (TypeError, ValueError):
                    pass
            return out

        # string
        return series.map(lambda v: self._mask_string_value(col, v))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_transform() before transform().")
        return pd.DataFrame(
            {col: self._apply_column(col, df[col]) for col in df.columns},
            index=df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ============================================================
    # Audit report
    # ============================================================
    def report(self) -> pd.DataFrame:
        rows = []
        for col, plan in self._plan.items():
            rows.append({
                "column": col,
                "treatment": plan["kind"],
                "pii_type": plan.get("pii_type", ""),
                "reason": plan.get("reason", ""),
                "inline_scrubbed": plan.get("scrub_inline", False),
            })
        return pd.DataFrame(rows)


# ============================================================
# Quick demo
# ============================================================
if __name__ == "__main__":
    df = pd.DataFrame({
        "customer_id":  ["C001", "C002", "C001", "C003", "C002"],
        "full_name":    ["Alice Wong", "Bob Patel", "Alice Wong",
                         "Carol Diaz", "Bob Patel"],
        "email":        ["alice.w@example.com", "bob.p@bigbank.io",
                         "alice.w@example.com", "carol@fintech.co",
                         "bob.p@bigbank.io"],
        "phone":        ["(555) 123-4567", "555.987.6543", "(555) 123-4567",
                         "555-222-1111", "555.987.6543"],
        "ssn":          ["123-45-6789", "987-65-4321", "123-45-6789",
                         "555-44-3322", "987-65-4321"],
        "card":         ["4532-1234-5678-9010", "5500-0000-0000-0004",
                         "4532-1234-5678-9010", "3782-822463-10005",
                         "5500-0000-0000-0004"],
        "ip":           ["192.168.1.10", "10.0.0.5", "192.168.1.10",
                         "172.16.4.22", "10.0.0.5"],
        "segment":      ["Retail", "Wholesale", "Retail", "DTC", "Wholesale"],
        "balance_usd":  [1250.50, 7800.00, 1250.50, 320.10, 7800.00],
        "opened_on":    pd.to_datetime(["2021-04-12", "2019-11-30",
                                        "2021-04-12", "2022-08-05",
                                        "2019-11-30"]),
        "is_active":    [True, False, True, True, False],
        "support_note": [
            "Customer Alice Wong at alice.w@example.com reported a wire delay this morning; called 555-123-4567 for confirmation.",
            "Bob Patel (SSN 987-65-4321) requested a statement reissue covering the previous billing cycle and asked about overdraft fees.",
            "Follow-up call: Alice Wong inquired about increasing her credit line ahead of an international trip booked for next quarter.",
            "Disputed ATM withdrawal originating from 172.16.4.22 was flagged by the fraud monitoring system and is under review.",
            "Bob Patel updated his mailing address after a recent relocation and confirmed the change via verification call.",
        ],
    })

    print("=" * 72)
    print("DEFAULT: PII auto-detect + format-preserve + inline scrub on notes")
    print("=" * 72)
    masker = DataMasker(
        pii_columns={"full_name": "name"},   # name can't be regex-detected
    )
    masked = masker.fit_transform(df)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\nORIGINAL:")
        print(df)
        print("\nMASKED:")
        print(masked)
        print("\nTREATMENT REPORT:")
        print(masker.report())

    print("\n" + "=" * 72)
    print("INVARIANT CHECKS")
    print("=" * 72)
    phone_sep = re.compile(r"[\(\-\.]")
    ssn_shape = re.compile(r"^\d{3}-\d{2}-\d{4}$")
    print(f"row uniqueness preserved : "
          f"{df.duplicated().sum() == masked.duplicated().sum()}")
    print(f"attribute relationship   : "
          f"{(masked.loc[df.customer_id == 'C001', 'email'].nunique() == 1)}")
    print(f"emails still look like emails: "
          f"{masked['email'].str.contains('@').all()}")
    print(f"phones keep separators   : "
          f"{masked['phone'].str.contains(phone_sep).all()}")
    print(f"SSNs keep dashes         : "
          f"{masked['ssn'].str.match(ssn_shape).all()}")
    print(f"cards keep last 4        : "
          f"{(masked['card'].str[-4:] == df['card'].str[-4:]).all()}")
    print(f"inline PII scrubbed      : "
          f"{not masked['support_note'].str.contains('alice.w@example.com').any()}")
    print(f"inline SSN scrubbed      : "
          f"{not masked['support_note'].str.contains('987-65-4321').any()}")

    print("\n" + "=" * 72)
    print("STRATEGY = tokenize  (tokens instead of format-preserving values)")
    print("=" * 72)
    m2 = DataMasker(
        pii_strategy="tokenize",
        pii_columns={"full_name": "name"},
    ).fit_transform(df)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(m2[["full_name", "email", "phone", "ssn", "card", "ip"]])

    print("\n" + "=" * 72)
    print("STRATEGY = redact  (fixed labels — destroys row uniqueness)")
    print("=" * 72)
    m3 = DataMasker(
        pii_strategy="redact",
        pii_columns={"full_name": "name"},
    ).fit_transform(df)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(m3[["full_name", "email", "phone", "ssn", "card", "ip"]])