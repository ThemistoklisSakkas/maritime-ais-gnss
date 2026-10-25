import numpy as np
import polars as pl
import pytest

from maritime.zones import Zones, anchorage_cells, in_anchorage, load_wpi, near_port

RADIUS = {"Large": 8.0, "Medium": 5.0, "Small": 3.0, "Very Small": 2.0, "default": 2.0}
M_PER_DEG_LAT = 111_195.0


@pytest.fixture
def ports():
    return pl.DataFrame({"port": ["A"], "lat": [55.5], "lon": [11.5], "radius_km": [2.0]})


def test_load_wpi_filters_bbox_and_maps_radius(tmp_path):
    csv = tmp_path / "wpi.csv"
    csv.write_text(
        "Main Port Name,UN/LOCODE,Latitude,Longitude,Harbor Size\n"
        "Arhus,DK AAR,56.15,10.2167,Medium\n"
        "Bandholm,DK BDX,54.8333,11.5,\n"
        "Piraeus,GR PIR,37.94,23.63,Large\n",
        encoding="utf-8",
    )
    df = load_wpi(csv, (3.0, 53.5, 17.0, 58.5), RADIUS)
    assert df["port"].to_list() == ["Arhus", "Bandholm"]
    assert df["radius_km"].to_list() == [5.0, 2.0]   # empty size -> default


def test_near_port_radius(ports):
    lat = np.array([55.5 + 1000 / M_PER_DEG_LAT, 55.5 + 5000 / M_PER_DEG_LAT])  # 1 km, 5 km N
    lon = np.array([11.5, 11.5])
    assert near_port(lat, lon, ports).tolist() == [True, False]


def test_near_port_empty_inputs(ports):
    assert near_port(np.array([]), np.array([]), ports).size == 0
    assert not near_port(np.array([55.0]), np.array([11.0]), ports.clear()).any()


def points(mmsis, status, lat=55.701, lon=12.601):
    return pl.DataFrame({"mmsi": mmsis, "nav_status": [status] * len(mmsis),
                         "lat": [lat] * len(mmsis), "lon": [lon] * len(mmsis)})


def test_anchorage_needs_several_distinct_vessels(tmp_path):
    a = tmp_path / "b00.parquet"
    pl.concat([
        points([1, 2, 3], "At anchor"),                 # 3 vessels: anchorage
        points([4, 4, 5], "At anchor", lat=56.301),     # only 2 distinct: not one
        points([6, 7, 8], "Under way using engine", lat=57.001),
    ]).write_parquet(a)
    cells = anchorage_cells([a], cell_deg=0.02, min_vessels=3, statuses=["At anchor", "Moored"])
    assert cells.height == 1
    probe = pl.DataFrame({"lat": [55.705, 56.305], "lon": [12.605, 12.605]})
    assert in_anchorage(probe, cells, 0.02).tolist() == [True, False]


def test_zones_mask_combines_ports_and_anchorages(ports):
    cells = pl.DataFrame({"cx": [630], "cy": [2850]}, schema={"cx": pl.Int32, "cy": pl.Int32})
    z = Zones(ports, cells, 0.02)
    df = pl.DataFrame({"lat": [55.5, 57.005, 56.0], "lon": [11.5, 12.605, 14.0]})
    assert z.mask(df).tolist() == [True, True, False]
