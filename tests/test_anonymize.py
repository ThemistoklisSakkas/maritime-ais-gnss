import polars as pl

from maritime.anonymize import pseudonym, pseudonymize

KEY = bytes(32)


def test_pseudonym_is_stable_and_key_dependent():
    assert pseudonym(219000000, KEY) == pseudonym(219000000, KEY)
    assert pseudonym(219000000, KEY) != pseudonym(219000001, KEY)
    assert pseudonym(219000000, KEY) != pseudonym(219000000, b"\x01" * 32)
    assert pseudonym(219000000, KEY).startswith("V-")
    assert "219000000" not in pseudonym(219000000, KEY)


def test_pseudonymize_replaces_ids_and_drops_names():
    df = pl.DataFrame({"mmsi_a": [219000000], "name_a": ["SHIP A"], "mmsi_b": [265000000],
                       "name_b": ["SHIP B"], "duration_h": [2.0]})
    out = pseudonymize(df, KEY)
    assert out.columns == ["vessel_a", "vessel_b", "duration_h"]
    assert out["vessel_a"][0] == pseudonym(219000000, KEY)
