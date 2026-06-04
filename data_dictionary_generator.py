"""
data_dictionary.py

Generate a data dictionary (source metadata catalog) from a tabular dataset,
with optional LLM-generated field descriptions.

Design notes
------------
* The LLM is injected as a plain callable ``(prompt: str) -> str``. This keeps
  the class decoupled from any specific SDK (Anthropic, OpenAI, local models,
  etc.). Adapters for common providers are shown at the bottom of the file.
* Profiling is provider-agnostic and works on a pandas DataFrame, which is the
  typical landing point for source data in an EDA pipeline.
* Descriptions are generated in a single batched call by default (cheaper and
  faster than one call per column), with a per-field fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import pandas as pd


# Type alias: anything that takes a prompt and returns text.
LLMCallable = Callable[[str], str]


@dataclass
class FieldProfile:
    """Metadata captured for a single source field/column."""

    name: str
    dtype: str
    inferred_semantic_type: str          # e.g. numeric, categorical, datetime, boolean, id, text
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any]
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean: Optional[float] = None
    description: Optional[str] = None     # filled in by the LLM (or left None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataDictionary:
    """
    Build a data dictionary from a DataFrame and (optionally) enrich it with
    LLM-generated field descriptions.

    Example
    -------
    >>> dd = DataDictionary(df, dataset_name="loan_applications",
    ...                     dataset_context="Consumer lending source extract from core banking system")
    >>> dd.profile()
    >>> dd.generate_descriptions(llm=my_llm)      # my_llm: Callable[[str], str]
    >>> dd.to_frame()                              # pandas DataFrame view
    >>> dd.to_json("loan_applications_dict.json")  # persist
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        dataset_context: str = "",
        sample_size: int = 5,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df
        self.dataset_name = dataset_name
        self.dataset_context = dataset_context
        self.sample_size = sample_size
        self.fields: list[FieldProfile] = []

    # ------------------------------------------------------------------ #
    # Profiling
    # ------------------------------------------------------------------ #
    def profile(self) -> "DataDictionary":
        """Compute metadata for every column. Returns self for chaining."""
        self.fields = [self._profile_column(col) for col in self.df.columns]
        return self

    def _profile_column(self, col: str) -> FieldProfile:
        s = self.df[col]
        n = len(s)
        non_null = int(s.count())
        nulls = n - non_null
        nunique = int(s.nunique(dropna=True))

        semantic = self._infer_semantic_type(s, nunique, n)

        # Pull a few non-null examples to give the LLM real context.
        samples = (
            s.dropna()
            .drop_duplicates()
            .head(self.sample_size)
            .tolist()
        )
        samples = [self._json_safe(v) for v in samples]

        min_v = max_v = mean_v = None
        if pd.api.types.is_numeric_dtype(s) and non_null > 0:
            min_v = self._json_safe(s.min())
            max_v = self._json_safe(s.max())
            mean_v = float(s.mean())
        elif pd.api.types.is_datetime64_any_dtype(s) and non_null > 0:
            min_v = self._json_safe(s.min())
            max_v = self._json_safe(s.max())

        return FieldProfile(
            name=col,
            dtype=str(s.dtype),
            inferred_semantic_type=semantic,
            non_null_count=non_null,
            null_count=nulls,
            null_pct=round(100 * nulls / n, 2) if n else 0.0,
            unique_count=nunique,
            unique_pct=round(100 * nunique / non_null, 2) if non_null else 0.0,
            sample_values=samples,
            min_value=min_v,
            max_value=max_v,
            mean=mean_v,
        )

    @staticmethod
    def _infer_semantic_type(s: pd.Series, nunique: int, n: int) -> str:
        if pd.api.types.is_bool_dtype(s):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if pd.api.types.is_numeric_dtype(s):
            # High-cardinality integer columns are often identifiers.
            name = str(s.name).lower()
            if nunique == n and ("id" in name or "key" in name or "number" in name):
                return "id"
            return "numeric"
        # Object / string types.
        if n and nunique / max(n, 1) > 0.5:
            return "text"
        return "categorical"

    @staticmethod
    def _json_safe(v: Any) -> Any:
        """Coerce numpy/pandas scalars into JSON-serializable Python types."""
        if pd.isna(v):
            return None
        if hasattr(v, "item"):          # numpy scalar
            return v.item()
        if isinstance(v, (pd.Timestamp,)):
            return v.isoformat()
        return v

    # ------------------------------------------------------------------ #
    # LLM-generated descriptions
    # ------------------------------------------------------------------ #
    def generate_descriptions(
        self,
        llm: LLMCallable,
        chunk_size: int = 25,
    ) -> "DataDictionary":
        """
        Use the provided LLM callable to generate a description of what each
        field is and how it is used.

        Fields are described in chunks: each call sends up to ``chunk_size``
        fields and asks the model for a name->description JSON map. Chunking
        keeps any single response small enough to parse reliably (the main
        failure mode on wide tables) while preserving cross-field context
        within each chunk, and bounds the number of API calls to
        ``ceil(n_fields / chunk_size)`` instead of one-per-field.

        Parameters
        ----------
        llm : Callable[[str], str]
            Any function that accepts a prompt string and returns the model's
            text response. See the adapters at the bottom of this file.
        chunk_size : int
            Maximum number of fields described per LLM call. Lower it for very
            wide tables or smaller-context models; raise it to reduce calls.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not self.fields:
            self.profile()

        for start in range(0, len(self.fields), chunk_size):
            chunk = self.fields[start : start + chunk_size]
            self._describe_chunk(llm, chunk)
        return self

    def _describe_chunk(self, llm: LLMCallable, chunk: list[FieldProfile]) -> None:
        field_payload = [
            {
                "name": fp.name,
                "dtype": fp.dtype,
                "semantic_type": fp.inferred_semantic_type,
                "null_pct": fp.null_pct,
                "unique_count": fp.unique_count,
                "sample_values": fp.sample_values,
            }
            for fp in chunk
        ]

        prompt = (
            "You are a data analyst documenting source data for a data dictionary.\n"
            f"Dataset name: {self.dataset_name}\n"
            f"Business context: {self.dataset_context or 'not provided'}\n\n"
            "For each field below, write a concise 1-2 sentence description covering "
            "what the field represents and how it is typically used. Base your answer "
            "on the field name, type, and sample values.\n\n"
            f"Fields:\n{json.dumps(field_payload, indent=2, default=str)}\n\n"
            "Respond with ONLY a JSON object mapping each field name to its description "
            "string. No markdown, no preamble.\n"
            'Example: {"customer_id": "Unique identifier for the customer..."}'
        )

        raw = llm(prompt)
        try:
            mapping = json.loads(self._strip_fences(raw))
        except json.JSONDecodeError as exc:
            names = [fp.name for fp in chunk]
            raise ValueError(
                f"Could not parse LLM response as JSON for fields {names}. "
                f"Consider lowering chunk_size. Raw response: {raw[:200]!r}"
            ) from exc

        for fp in chunk:
            if fp.name in mapping:
                fp.description = str(mapping[fp.name]).strip()

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ```json ... ``` fences a model might wrap output in."""
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[-1] if "\n" in t else t
            t = t.rsplit("```", 1)[0]
        return t.strip()

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def to_records(self) -> list[dict[str, Any]]:
        return [fp.to_dict() for fp in self.fields]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_records())

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        payload = {
            "dataset_name": self.dataset_name,
            "dataset_context": self.dataset_context,
            "row_count": int(len(self.df)),
            "field_count": len(self.fields),
            "fields": self.to_records(),
        }
        text = json.dumps(payload, indent=indent, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text


# ====================================================================== #
# Optional LLM adapters
# ----------------------------------------------------------------------
# Each returns a Callable[[str], str] suitable for `generate_descriptions`.
# ====================================================================== #
def anthropic_llm(model: str = "claude-sonnet-4-20250514", max_tokens: int = 2000) -> LLMCallable:
    """Adapter for the Anthropic SDK. Requires `pip install anthropic`."""
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    def _call(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    return _call


def openai_llm(model: str = "gpt-4o-mini") -> LLMCallable:
    """Adapter for the OpenAI SDK. Requires `pip install openai`."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from env

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return _call


# ---------------------------------------------------------------------- #
# Demo
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "account_id": [1001, 1002, 1003, 1004],
            "balance": [2540.10, 0.00, 18900.55, 430.00],
            "open_date": pd.to_datetime(["2021-03-01", "2022-07-15", "2020-11-30", "2023-01-05"]),
            "is_delinquent": [False, True, False, False],
            "product_type": ["checking", "savings", "checking", "credit"],
        }
    )

    dd = DataDictionary(
        demo,
        dataset_name="accounts",
        dataset_context="Retail banking account snapshot.",
    ).profile()

    # Without an LLM you still get full profiling:
    print(dd.to_frame()[["name", "dtype", "inferred_semantic_type", "null_pct", "unique_count"]])

    # With an LLM (uncomment after configuring credentials):
    # dd.generate_descriptions(llm=anthropic_llm())
    # print(dd.to_json())

    # A trivial mock to test the wiring without API calls:
    def mock_llm(prompt: str) -> str:
        return json.dumps({c: f"Auto description for {c}" for c in demo.columns})

    dd.generate_descriptions(llm=mock_llm)
    print(dd.to_frame()[["name", "description"]])