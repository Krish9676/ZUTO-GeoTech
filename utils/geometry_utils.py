"""Utility geometry helpers for field/regional inputs."""

from __future__ import annotations

from typing import Iterable, List, Tuple
import math


class GeometryUtils:
    """Geometry helper methods used by ingestion pipeline."""

    EARTH_RADIUS_KM = 6371.0

    @staticmethod
    def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance (Haversine) in km."""
        p = math.pi / 180.0
        dlat = (lat2 - lat1) * p
        dlon = (lon2 - lon1) * p
        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
        return 2 * GeometryUtils.EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))

    @staticmethod
    def calculate_bbox_from_point(lat: float, lon: float, area_ha: float = 1.0) -> List[float]:
        """Approximate square bbox from center point and area in hectares."""
        area_km2 = max(area_ha, 0.01) * 0.01
        half_side_km = math.sqrt(area_km2) / 2.0
        dlat = half_side_km / 111.0
        dlon = half_side_km / max(111.0 * math.cos(math.radians(lat)), 1e-6)
        return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

    @staticmethod
    def calculate_bbox_from_geometry(geometry: object) -> List[float]:
        """Extract bbox [min_lon, min_lat, max_lon, max_lat] from GeoJSON-like object."""
        if hasattr(geometry, 'bounds'):
            minx, miny, maxx, maxy = geometry.bounds
            return [float(minx), float(miny), float(maxx), float(maxy)]

        if isinstance(geometry, dict):
            if 'bbox' in geometry:
                bbox = geometry['bbox']
                if len(bbox) == 4:
                    return [float(b) for b in bbox]
            coords = geometry.get('coordinates')
            if coords:
                flat = GeometryUtils._flatten_coords(coords)
                lons = [p[0] for p in flat]
                lats = [p[1] for p in flat]
                return [min(lons), min(lats), max(lons), max(lats)]

        raise ValueError('Unsupported geometry format')

    @staticmethod
    def calculate_centroid_from_geometry(geometry: object) -> Tuple[float, float]:
        """Return centroid (lat, lon)."""
        if hasattr(geometry, 'centroid'):
            return float(geometry.centroid.y), float(geometry.centroid.x)

        bbox = GeometryUtils.calculate_bbox_from_geometry(geometry)
        min_lon, min_lat, max_lon, max_lat = bbox
        return (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0

    @staticmethod
    def calculate_area_from_geometry(geometry: object) -> float:
        """Approximate geometry area in hectares using bbox envelope."""
        min_lon, min_lat, max_lon, max_lat = GeometryUtils.calculate_bbox_from_geometry(geometry)
        width = GeometryUtils.calculate_distance_km(min_lat, min_lon, min_lat, max_lon)
        height = GeometryUtils.calculate_distance_km(min_lat, min_lon, max_lat, min_lon)
        return max(width * height * 100.0, 0.01)

    @staticmethod
    def validate_bbox(bbox: List[float]) -> bool:
        """Validate WGS84 bbox ordering and ranges."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        min_lon, min_lat, max_lon, max_lat = bbox
        return (-180 <= min_lon < max_lon <= 180) and (-90 <= min_lat < max_lat <= 90)

    @staticmethod
    def _flatten_coords(coords: Iterable) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        def rec(v):
            if isinstance(v, (list, tuple)) and len(v) >= 2 and isinstance(v[0], (int, float)):
                pts.append((float(v[0]), float(v[1])))
            elif isinstance(v, (list, tuple)):
                for x in v:
                    rec(x)
        rec(coords)
        if not pts:
            raise ValueError('No coordinates found in geometry')
        return pts
