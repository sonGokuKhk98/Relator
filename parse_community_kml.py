"""
Parse Municipality/Location/community.kml into a GeoJSON file
with real polygon boundaries for each Dubai community.
"""
import re
import json
from pathlib import Path

KML_PATH = Path(__file__).parent / "data" / "Municipality" / "Location" / "community.kml"
OUT_PATH = Path(__file__).parent / "data_variable" / "community_polygons.geojson"


def parse_kml():
    text = KML_PATH.read_text(encoding="utf-8")

    # Split by <Placemark> blocks
    placemarks = re.findall(r'<Placemark[^>]*>(.*?)</Placemark>', text, re.DOTALL)
    print(f"Found {len(placemarks)} placemarks")

    features = []

    for pm in placemarks:
        # Extract community name (English)
        cname_match = re.search(r'<SimpleData name="CNAME_E">(.*?)</SimpleData>', pm)
        if not cname_match:
            continue
        cname = cname_match.group(1).strip()

        # Extract community number
        comm_num_match = re.search(r'<SimpleData name="COMM_NUM">(.*?)</SimpleData>', pm)
        comm_num = comm_num_match.group(1).strip() if comm_num_match else ""

        # Extract Arabic name
        cname_ar_match = re.search(r'<SimpleData name="CNAME_A">(.*?)</SimpleData>', pm)
        cname_ar = cname_ar_match.group(1).strip() if cname_ar_match else ""

        # Extract coordinates from <Polygon> or <MultiGeometry>
        coord_blocks = re.findall(r'<coordinates>(.*?)</coordinates>', pm, re.DOTALL)
        if not coord_blocks:
            continue

        rings = []
        for block in coord_blocks:
            coords_raw = block.strip().split()
            ring = []
            for c in coords_raw:
                parts = c.split(",")
                if len(parts) >= 2:
                    try:
                        lng = float(parts[0])
                        lat = float(parts[1])
                        ring.append([lng, lat])
                    except ValueError:
                        continue
            if len(ring) >= 3:
                # Ensure ring is closed
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                rings.append(ring)

        if not rings:
            continue

        # Determine geometry type
        if len(rings) == 1:
            geometry = {"type": "Polygon", "coordinates": [rings[0]]}
        else:
            # Multiple rings — could be MultiPolygon or polygon with holes
            # Treat as MultiPolygon for safety
            geometry = {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}

        # Compute centroid for label placement
        all_lats = []
        all_lngs = []
        for ring in rings:
            for coord in ring:
                all_lngs.append(coord[0])
                all_lats.append(coord[1])
        centroid_lat = sum(all_lats) / len(all_lats)
        centroid_lng = sum(all_lngs) / len(all_lngs)

        feature = {
            "type": "Feature",
            "properties": {
                "name": cname,
                "name_ar": cname_ar,
                "comm_num": comm_num,
                "centroid_lat": round(centroid_lat, 6),
                "centroid_lng": round(centroid_lng, 6),
            },
            "geometry": geometry,
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(geojson, f)

    # Also compute file size
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Written {len(features)} community polygons to {OUT_PATH.name} ({size_kb:.0f} KB)")

    # Print some stats
    names = [f["properties"]["name"] for f in features]
    print(f"Communities: {', '.join(names[:10])}... ({len(names)} total)")


if __name__ == "__main__":
    parse_kml()
