"""
Dashboard Architecture & Data-Loading Layer for Q-EVAC.
Provides cached, verified data access for the Streamlit + Folium dashboard cockpit.
Strictly distinguishes REAL DATA, DERIVED FROM REAL DATA, CONTROLLED SIMULATION, and MODEL OUTPUT.
Enforces the approved designation: Validated Coastal Sample & Prototype Testbed.
"""

import json
import os
import networkx as nx
import pandas as pd
from typing import Dict, Any, List, Tuple

class QEvacDataLoader:
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.statewide_dir = os.path.join(base_dir, "data/processed/statewide")
        self.events_dir = os.path.join(base_dir, "data/processed/events")
        self.results_dir = os.path.join(base_dir, "results")

    # ---------------------------------------------------------
    # 1. STATEWIDE BASE LAYER (Validated Coastal Sample)
    # ---------------------------------------------------------
    def load_statewide_shelters(self) -> Dict[str, Any]:
        path = os.path.join(self.statewide_dir, "ap_shelters_statewide.geojson")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "data": data,
            "provenance_tag": "DERIVED FROM REAL DATA",
            "provenance_details": {
                "source": "APSAC GeoServer WMS (Andhra-CycloneShelters)",
                "designation": "Validated Coastal Sample & Prototype Testbed",
                "total_facilities": len(data.get("features", [])),
                "capacity_rule": "OFFICIAL_DESIGN_STANDARD (1,000 for MPCS; 500 for secondary relief centres)"
            }
        }

    def load_statewide_villages(self) -> Dict[str, Any]:
        path = os.path.join(self.statewide_dir, "ap_villages_statewide.geojson")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "data": data,
            "provenance_tag": "REAL DATA",
            "provenance_details": {
                "source": "Census of India 2011 Primary Census Abstract (PCA)",
                "designation": "Validated Coastal Sample & Prototype Testbed",
                "total_habitations": len(data.get("features", []))
            }
        }

    def load_statewide_road_graph(self) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        path = os.path.join(self.statewide_dir, "ap_road_graph_statewide.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
        return G, {
            "provenance_tag": "DERIVED FROM REAL DATA",
            "provenance_details": {
                "source": "OpenStreetMap / MoRTH Highway Corridors / APSAC",
                "designation": "Validated Coastal Sample & Prototype Testbed",
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "projection": "EPSG:32644 (UTM Zone 44N)"
            }
        }

    # ---------------------------------------------------------
    # 2. EVENT LAYER SELECTION
    # ---------------------------------------------------------
    def list_available_events(self) -> List[Dict[str, str]]:
        return [
            {
                "event_id": "EVENT_NORTH_ANDHRA_DEEP_DEPRESSION_2026",
                "display_name": "September 2026 North Andhra Deep Depression",
                "colloquial_name": "Arnab (Colloquial Media Reference)",
                "classification": "Deep Depression (IMD Official)",
                "status": "FIRST_VALIDATION_SCENARIO",
                "provenance_tag": "REAL DATA"
            }
        ]

    def load_event_metadata(self, event_id: str) -> Dict[str, Any]:
        event_path = os.path.join(self.events_dir, "north_andhra_2026/event_metadata.json")
        with open(event_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "metadata": meta,
            "provenance_tag": "REAL DATA",
            "provenance_details": meta.get("provenance", {})
        }

    def load_event_subgraph(self, event_id: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        event_path = os.path.join(self.events_dir, "north_andhra_2026/affected_subgraph.json")
        with open(event_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        G_sub = nx.node_link_graph(data)
        return G_sub, {
            "provenance_tag": "DERIVED FROM REAL DATA",
            "provenance_details": {
                "parent_dataset": "ap_road_graph_statewide",
                "filter": "45 km meteorological impact buffer centered on Kalingapatnam landfall",
                "nodes": G_sub.number_of_nodes(),
                "edges": G_sub.number_of_edges()
            }
        }

    # ---------------------------------------------------------
    # 3. CONTROLLED DISRUPTION & DYNAMIC REROUTING ARTIFACTS
    # ---------------------------------------------------------
    def load_rerouting_artifacts(self) -> Dict[str, Any]:
        reroute_dir = os.path.join(self.events_dir, "north_andhra_2026/rerouting")
        report_path = os.path.join(self.results_dir, "rerouting/rerouting_report.json")
        summary_csv = os.path.join(self.results_dir, "rerouting/rerouting_summary.csv")

        with open(os.path.join(reroute_dir, "disruption_definition.json"), "r", encoding="utf-8") as f:
            disruption_def = json.load(f)

        with open(os.path.join(reroute_dir, "routes_pre.json"), "r", encoding="utf-8") as f:
            routes_pre = json.load(f)

        with open(os.path.join(reroute_dir, "routes_post.json"), "r", encoding="utf-8") as f:
            routes_post = json.load(f)

        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        summary_df = pd.read_csv(summary_csv)

        return {
            "disruption_definition": disruption_def,
            "disruption_provenance_tag": "CONTROLLED SIMULATION",
            "disruption_disclaimer": "CONTROLLED DISRUPTION EXPERIMENT — NOT AN OBSERVED ROAD CLOSURE",
            "routes_pre": routes_pre,
            "routes_post": routes_post,
            "report_data": report_data,
            "summary_df": summary_df,
            "model_provenance_tag": "MODEL OUTPUT"
        }

    # ---------------------------------------------------------
    # 4. BENCHMARK LADDER & PENALTY SWEEPS
    # ---------------------------------------------------------
    def load_benchmark_summary(self) -> pd.DataFrame:
        csv_path = os.path.join(self.results_dir, "size_ladder/ladder_summary.csv")
        return pd.read_csv(csv_path)

    def load_detailed_benchmark_runs(self) -> pd.DataFrame:
        csv_path = os.path.join(self.results_dir, "size_ladder/ladder_runs.csv")
        return pd.read_csv(csv_path)

    def load_penalty_sweep(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.results_dir, "size_ladder/penalty_sweep.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
