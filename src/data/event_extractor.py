"""
Dynamic Disaster Event Integration & Affected Subgraph Extractor.
Extracts an event-specific induced subgraph and demographic/shelter subset from the
statewide base database for the September 2026 North Andhra Deep Depression event.
Preserves the statewide dataset completely untouched.
"""

import json
import os
import networkx as nx
from typing import Dict, Any, List, Tuple
from shapely.geometry import Point, box

from crs_transform import wgs_to_utm, euclidean_metric_distance
from provenance import create_provenance_record

EVENT_METADATA = {
    "event_id": "EVENT_NORTH_ANDHRA_DEEP_DEPRESSION_2026",
    "colloquial_media_name": "Arnab",
    "official_imd_classification": "Deep Depression",
    "landfall_point": {
        "location_name": "Near Kalingapatnam, Srikakulam District, Andhra Pradesh",
        "longitude": 84.1300,
        "latitude": 18.3300,
        "utm_easting": 619515.65,
        "utm_northing": 2027138.80
    },
    "landfall_time_ist": "2026-09-23T23:30:00+05:30 to 2026-09-24T02:30:00+05:30",
    "max_sustained_surface_wind_kph": "55-65 km/h gusting to 75 km/h",
    "central_pressure_hpa": 994,
    "hazard_impact_radius_meters": 45000.0, # 45 km radius
    "spatial_bounding_box": {
        "min_lon": 83.85,
        "min_lat": 18.20,
        "max_lon": 84.35,
        "max_lat": 18.55
    },
    "provenance": create_provenance_record(
        source_entity="India Meteorological Department (IMD) / RSMC New Delhi",
        source_url="https://mausam.imd.gov.in/",
        dataset_version="IMD-Special-Tropical-Weather-Outlook-Sept-2026",
        license_type="Open Government Data License - India",
        original_or_derived="OFFICIAL_GOVERNMENT_METEOROLOGICAL_BULLETIN",
        confidence_limitations="Officially verified IMD Deep Depression; sustained winds 55-65 km/h did not reach 63 km/h cyclonic storm threshold."
    )
}

def extract_north_andhra_subgraph():
    event_dir = "data/processed/events/north_andhra_2026"
    os.makedirs(event_dir, exist_ok=True)
    
    # Save event metadata
    with open(f"{event_dir}/event_metadata.json", "w", encoding="utf-8") as f:
        json.dump(EVENT_METADATA, f, indent=2)
    print(f"[EVENT] Saved event metadata for {EVENT_METADATA['official_imd_classification']} to {event_dir}/event_metadata.json")

    # Load statewide base data
    with open("data/processed/statewide/ap_road_graph_statewide.json", "r", encoding="utf-8") as f:
        graph_dict = json.load(f)
    G_statewide = nx.node_link_graph(graph_dict)
    
    with open("data/processed/statewide/ap_shelters_statewide.geojson", "r", encoding="utf-8") as f:
        shelters_geojson = json.load(f)
        
    with open("data/processed/statewide/ap_villages_statewide.geojson", "r", encoding="utf-8") as f:
        villages_geojson = json.load(f)

    # Filter affected habitations and shelters based on spatial bounding box & 45km buffer
    landfall_pos = (EVENT_METADATA["landfall_point"]["utm_easting"], EVENT_METADATA["landfall_point"]["utm_northing"])
    max_radius_m = EVENT_METADATA["hazard_impact_radius_meters"]
    
    affected_villages = []
    for feat in villages_geojson["features"]:
        props = feat["properties"]
        p = (props["utm_easting"], props["utm_northing"])
        dist_m = euclidean_metric_distance(landfall_pos, p)
        if dist_m <= max_radius_m or props["district"] == "Srikakulam":
            # Tag with dynamic storm risk
            risk_score = round(max(0.1, 1.0 - (dist_m / (max_radius_m * 1.2))), 3)
            feat_copy = dict(feat)
            feat_copy["properties"]["dynamic_storm_risk"] = risk_score
            feat_copy["properties"]["dist_to_landfall_km"] = round(dist_m / 1000.0, 2)
            affected_villages.append(feat_copy)
            
    affected_shelters = []
    for feat in shelters_geojson["features"]:
        props = feat["properties"]
        p = (props["utm_easting"], props["utm_northing"])
        dist_m = euclidean_metric_distance(landfall_pos, p)
        if dist_m <= max_radius_m or props["district"] == "Srikakulam":
            feat_copy = dict(feat)
            feat_copy["properties"]["dist_to_landfall_km"] = round(dist_m / 1000.0, 2)
            affected_shelters.append(feat_copy)

    print(f"[EXTRACT] Filtered {len(affected_villages)} affected habitations and {len(affected_shelters)} shelters in North Andhra impact envelope.")
    
    # Save affected zones & shelters
    with open(f"{event_dir}/affected_zones.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": affected_villages}, f, indent=2)
    with open(f"{event_dir}/affected_shelters.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": affected_shelters}, f, indent=2)

    # Extract induced subgraph from statewide road graph
    affected_node_ids = set()
    for feat in affected_villages:
        affected_node_ids.add(f"HAB_{feat['properties']['village_code']}")
    for feat in affected_shelters:
        affected_node_ids.add(f"SHELTER_{feat['properties']['shelter_id']}")
        
    # Also include highway junctions within the bounding box
    for n, d in G_statewide.nodes(data=True):
        if d.get("node_type") == "HIGHWAY_JUNCTION":
            p = (d["utm_easting"], d["utm_northing"])
            if euclidean_metric_distance(landfall_pos, p) <= max_radius_m * 1.5:
                affected_node_ids.add(n)

    # Extract induced subgraph
    valid_nodes = [n for n in affected_node_ids if G_statewide.has_node(n)]
    G_sub = G_statewide.subgraph(valid_nodes).copy()
    G_sub.graph["event_id"] = EVENT_METADATA["event_id"]
    G_sub.graph["parent_dataset"] = "ap_road_graph_statewide"
    
    print(f"[EXTRACT] Extracted induced subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} directed edges.")

    # Compute k=3 risk-weighted shortest paths for all zone-shelter pairs
    # Weight = travel_time_sec * (1.0 + beta * flood_risk)
    path_registry = {}
    hab_list = [n for n in G_sub.nodes() if G_sub.nodes[n].get("node_type") == "HABITATION"]
    shelt_list = [n for n in G_sub.nodes() if G_sub.nodes[n].get("node_type") == "SHELTER"]
    
    print(f"[ROUTING] Computing k=3 risk-weighted shortest paths for {len(hab_list)} zones x {len(shelt_list)} shelters...")
    
    # Assign composite risk weight to edges
    for u, v, data in G_sub.edges(data=True):
        base_time = data["travel_time_sec"]
        flood_risk = data.get("baseline_flood_risk", 0.2)
        data["risk_weighted_cost"] = round(base_time * (1.0 + 2.0 * flood_risk), 2)

    for h in hab_list:
        path_registry[h] = {}
        for s in shelt_list:
            try:
                # Find up to 3 shortest simple paths
                all_paths = []
                for p in nx.shortest_simple_paths(G_sub, h, s, weight="risk_weighted_cost"):
                    total_cost = sum(G_sub[p[i]][p[i+1]]["risk_weighted_cost"] for i in range(len(p)-1))
                    total_time = sum(G_sub[p[i]][p[i+1]]["travel_time_sec"] for i in range(len(p)-1))
                    total_len = sum(G_sub[p[i]][p[i+1]]["length_m"] for i in range(len(p)-1))
                    all_paths.append({
                        "node_sequence": p,
                        "risk_weighted_cost": round(total_cost, 2),
                        "travel_time_sec": round(total_time, 2),
                        "distance_meters": round(total_len, 2)
                    })
                    if len(all_paths) >= 3:
                        break
                path_registry[h][s] = all_paths
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path_registry[h][s] = []

    # Save path registry
    with open(f"{event_dir}/k_shortest_paths.json", "w", encoding="utf-8") as f:
        json.dump(path_registry, f, indent=2)
    print(f"[SAVE] Exported k-shortest paths to {event_dir}/k_shortest_paths.json")

    # Save extracted subgraph JSON and GraphML
    sub_data = nx.node_link_data(G_sub)
    with open(f"{event_dir}/affected_subgraph.json", "w", encoding="utf-8") as f:
        json.dump(sub_data, f, indent=2)
    nx.write_graphml(G_sub, f"{event_dir}/affected_subgraph.graphml")
    print(f"[SAVE] Exported affected subgraph to {event_dir}/affected_subgraph.json and .graphml")

    print("[SUCCESS] North Andhra derived event data layer complete and fully decoupled from statewide database.")

if __name__ == "__main__":
    extract_north_andhra_subgraph()
