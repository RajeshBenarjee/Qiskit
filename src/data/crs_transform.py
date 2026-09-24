"""
Coordinate Reference System (CRS) Normalization and Projection Utility for Andhra Pradesh.
Standardizes between EPSG:4326 (WGS 84 geographic) and EPSG:32644 (WGS 84 / UTM zone 44N projected).
UTM Zone 44N covers 78°E to 84°E (covering the entirety of coastal and central Andhra Pradesh).
"""

from typing import Tuple, List, Dict, Any
import math
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform

# Initialize pyproj transformers (always_xy=True expects (lon, lat) / (easting, northing))
_transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
_transformer_to_wgs = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)

def wgs_to_utm(lon: float, lat: float) -> Tuple[float, float]:
    """
    Transforms WGS84 (lon, lat) to UTM Zone 44N (easting, northing) in meters.
    """
    easting, northing = _transformer_to_utm.transform(lon, lat)
    return easting, northing

def utm_to_wgs(easting: float, northing: float) -> Tuple[float, float]:
    """
    Transforms UTM Zone 44N (easting, northing) in meters to WGS84 (lon, lat).
    """
    lon, lat = _transformer_to_wgs.transform(easting, northing)
    return lon, lat

def euclidean_metric_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculates Euclidean distance in meters between two projected UTM (easting, northing) points.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)

def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculates great-circle distance between two geographic coordinates in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def transform_geojson_geometry(geom_dict: Dict[str, Any], to_crs: str = "EPSG:32644") -> Dict[str, Any]:
    """
    Transforms a GeoJSON geometry dictionary between EPSG:4326 and EPSG:32644.
    """
    geom = shape(geom_dict)
    if to_crs == "EPSG:32644":
        projected = transform(_transformer_to_utm.transform, geom)
    elif to_crs == "EPSG:4326":
        projected = transform(_transformer_to_wgs.transform, geom)
    else:
        raise ValueError(f"Unsupported target CRS: {to_crs}")
    return mapping(projected)
