import os
import csv
import json
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

SPECIES = "Craspedacusta sowerbii"


def main():
    map_path = os.path.join(os.path.dirname(__file__), CFG.get("map_output_path", "data/bloom_map.geojson"))
    out_path = os.path.join(os.path.dirname(__file__), CFG.get("sightings_output_path", "data/sightings_gbif.csv"))

    if not os.path.exists(map_path):
        print(f"no bloom map at {map_path}")
        return

    with open(map_path) as f:
        features = json.load(f).get("features", [])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scientificName",
            "decimalLatitude",
            "decimalLongitude",
            "individualCount",
            "occurrenceRemarks",
        ])
        for feature in features:
            lon, lat = feature["geometry"]["coordinates"]
            props = feature["properties"]
            writer.writerow([
                SPECIES,
                lat,
                lon,
                props.get("count", 0),
                f"bloom_severity {props.get('bloom_severity', 0)}",
            ])

    print(f"wrote {len(features)} sightings to {out_path}")


if __name__ == "__main__":
    main()
