"""Tests for the dynamic name-pool expansion in DataMasker._fp_name.

Prior implementation used _FAKE_FIRST (30) × _FAKE_LAST (30) = 900 fixed
combinations, which caused collisions for datasets with >900 distinct names.

New implementation uses HMAC-based syllable synthesis whose output space
scales automatically with the column's unique-value count during fit(),
keeping collision probability negligible for any practical dataset size.
"""

import pandas as pd
import pytest

from data_pii_mask import DataMasker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _masker(salt="test-salt", **kw):
    return DataMasker(salt=salt, pii_columns={"name": "name"}, **kw)


# ---------------------------------------------------------------------------
# 1. Plan records metadata for name columns
# ---------------------------------------------------------------------------

class TestPlanMetadata:
    def test_n_unique_recorded(self):
        df = pd.DataFrame({"name": ["Alice Wong", "Bob Patel", "Carol Diaz"]})
        m = _masker().fit(df)
        assert m._plan["name"]["name_n_unique"] == 3

    def test_n_syllables_recorded(self):
        df = pd.DataFrame({"name": ["Alice Wong", "Bob Patel", "Carol Diaz"]})
        m = _masker().fit(df)
        assert isinstance(m._plan["name"]["name_n_syllables"], int)
        assert m._plan["name"]["name_n_syllables"] >= 2

    def test_syllable_count_grows_with_cardinality(self):
        """Higher n_unique must produce >= syllable count as lower n_unique."""
        small_df = pd.DataFrame({"name": [f"Name {i}" for i in range(10)]})
        large_df = pd.DataFrame({"name": [f"Name {i}" for i in range(5000)]})
        m_small = _masker(salt="same").fit(small_df)
        m_large = _masker(salt="same").fit(large_df)
        assert (
            m_large._plan["name"]["name_n_syllables"]
            >= m_small._plan["name"]["name_n_syllables"]
        )

    def test_other_pii_types_unaffected(self):
        """name_n_unique / name_n_syllables should NOT appear on non-name columns."""
        df = pd.DataFrame({"email": ["a@b.com", "c@d.net"]})
        m = DataMasker(salt="s", pii_columns={"email": "email"}).fit(df)
        assert "name_n_unique" not in m._plan["email"]
        assert "name_n_syllables" not in m._plan["email"]


# ---------------------------------------------------------------------------
# 2. _name_syllables_needed — static helper
# ---------------------------------------------------------------------------

class TestSyllablesNeeded:
    @pytest.mark.parametrize("n", [1, 5, 10, 28])
    def test_small_n_at_least_2(self, n):
        assert DataMasker._name_syllables_needed(n) >= 2

    @pytest.mark.parametrize("n", [100, 500, 1000, 5000, 50000])
    def test_output_space_exceeds_n_squared(self, n):
        k = DataMasker._name_syllables_needed(n)
        assert 90 ** k >= 10 * n ** 2

    def test_monotonic(self):
        prev = DataMasker._name_syllables_needed(1)
        for n in [10, 100, 1000, 10_000, 100_000]:
            cur = DataMasker._name_syllables_needed(n)
            assert cur >= prev, f"syllables decreased at n={n}"
            prev = cur


# ---------------------------------------------------------------------------
# 3. 1:1 uniqueness — the core guarantee
# ---------------------------------------------------------------------------

class TestUniqueness:
    def test_small_dataset_1to1(self):
        names = [f"Person {i} Last{i}" for i in range(100)]
        df = pd.DataFrame({"name": names})
        result = _masker().fit_transform(df)
        assert result["name"].nunique() == len(names)

    def test_medium_dataset_1to1(self):
        """n=1000 — well above the old 900-combination ceiling."""
        names = [f"First{i} Last{i}" for i in range(1000)]
        df = pd.DataFrame({"name": names})
        result = _masker().fit_transform(df)
        assert result["name"].nunique() == len(names)

    def test_large_dataset_1to1(self):
        """n=5000 — demonstrates the pool truly scales."""
        names = [f"First{i} Last{i}" for i in range(5000)]
        df = pd.DataFrame({"name": names})
        result = _masker().fit_transform(df)
        assert result["name"].nunique() == len(names)

    def test_single_token_names_1to1(self):
        names = [f"Name{i}" for i in range(200)]
        df = pd.DataFrame({"name": names})
        result = _masker().fit_transform(df)
        assert result["name"].nunique() == len(names)


# ---------------------------------------------------------------------------
# 4. Determinism — same salt + same data → same output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_salt_same_output(self):
        df = pd.DataFrame({"name": ["Alice Wong", "Bob Patel", "Alice Wong"]})
        r1 = DataMasker(salt="fixed", pii_columns={"name": "name"}).fit_transform(df)
        r2 = DataMasker(salt="fixed", pii_columns={"name": "name"}).fit_transform(df)
        assert (r1["name"] == r2["name"]).all()

    def test_repeated_value_same_masked_value(self):
        df = pd.DataFrame({"name": ["Alice Wong", "Bob Patel", "Alice Wong"]})
        result = _masker().fit_transform(df)
        assert result["name"].iloc[0] == result["name"].iloc[2]

    def test_different_salt_different_output(self):
        df = pd.DataFrame({"name": ["Alice Wong", "Bob Patel"]})
        r1 = DataMasker(salt="salt-a", pii_columns={"name": "name"}).fit_transform(df)
        r2 = DataMasker(salt="salt-b", pii_columns={"name": "name"}).fit_transform(df)
        assert not (r1["name"] == r2["name"]).all()


# ---------------------------------------------------------------------------
# 5. Format preservation — single vs multi-token names
# ---------------------------------------------------------------------------

class TestFormatPreservation:
    def test_single_token_stays_single_token(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
        result = _masker().fit_transform(df)
        for val in result["name"]:
            assert " " not in val, f"Unexpected space in single-token name: {val!r}"

    def test_multi_token_stays_multi_token(self):
        df = pd.DataFrame({"name": ["Alice Wong", "Bob Patel"]})
        result = _masker().fit_transform(df)
        for val in result["name"]:
            assert " " in val, f"Expected space in multi-token name: {val!r}"

    def test_output_is_capitalized(self):
        df = pd.DataFrame({"name": ["alice wong", "BOB PATEL", "carol diaz"]})
        result = _masker().fit_transform(df)
        for val in result["name"]:
            # Each word should start uppercase (capitalize() on each part)
            parts = val.split()
            for p in parts:
                assert p[0].isupper(), f"Expected capitalized part in {val!r}"

    def test_output_is_alphabetic(self):
        df = pd.DataFrame({"name": ["Alice Wong", "Bob Patel"]})
        result = _masker().fit_transform(df)
        for val in result["name"]:
            assert val.replace(" ", "").isalpha(), f"Non-alpha chars in {val!r}"


# ---------------------------------------------------------------------------
# 6. NaN pass-through
# ---------------------------------------------------------------------------

class TestNullHandling:
    def test_nan_passthrough(self):
        import numpy as np
        df = pd.DataFrame({"name": ["Alice Wong", None, float("nan"), "Bob Patel"]})
        result = _masker().fit_transform(df)
        assert pd.isna(result["name"].iloc[1])
        assert pd.isna(result["name"].iloc[2])
        assert result["name"].iloc[0] != result["name"].iloc[3]


# ---------------------------------------------------------------------------
# 7. Masking does not affect original DataFrame
# ---------------------------------------------------------------------------

class TestNoMutation:
    def test_original_unchanged(self):
        df = pd.DataFrame({"name": ["Alice Wong", "Bob Patel"]})
        orig = df.copy()
        _masker().fit_transform(df)
        pd.testing.assert_frame_equal(df, orig)


# ---------------------------------------------------------------------------
# 8. Integration — name column inside a wider DataFrame
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_name_col_in_mixed_dataframe(self):
        df = pd.DataFrame({
            "id":    [1, 2, 3, 1],
            "name":  ["Alice Wong", "Bob Patel", "Carol Diaz", "Alice Wong"],
            "email": ["a@x.com", "b@y.net", "c@z.io", "a@x.com"],
        })
        m = DataMasker(salt="s", pii_columns={"name": "name"})
        result = m.fit_transform(df)

        # name column is masked
        assert not (result["name"] == df["name"]).any()
        # repeated "Alice Wong" maps to same fake name
        assert result["name"].iloc[0] == result["name"].iloc[3]
        # email still has @
        assert result["email"].str.contains("@").all()
        # id column unchanged
        assert (result["id"] == df["id"]).all()
