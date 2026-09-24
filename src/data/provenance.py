"""
Provenance metadata management for Q-EVAC.
Ensures every dataset, spatial layer, and individual record maintains an immutable audit trail.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional

def create_provenance_record(
    source_entity: str,
    source_url: str,
    dataset_version: str,
    license_type: str,
    original_or_derived: str,
    confidence_limitations: str,
    capacity_provenance: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a standardized provenance metadata dictionary.
    
    capacity_provenance must be one of:
      - 'ACTUAL_OFFICIAL_CAPACITY'
      - 'OFFICIAL_DESIGN_STANDARD'
      - 'OFFICIAL_PLANNING_NORM'
      - 'ESTIMATED'
      - 'SYNTHETIC'
    """
    record = {
        "source_entity": source_entity,
        "source_url": source_url,
        "retrieval_date": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "license": license_type,
        "original_or_derived": original_or_derived,
        "confidence_limitations": confidence_limitations,
    }
    if capacity_provenance is not None:
        valid_provenance = [
            "ACTUAL_OFFICIAL_CAPACITY",
            "OFFICIAL_DESIGN_STANDARD",
            "OFFICIAL_PLANNING_NORM",
            "ESTIMATED",
            "SYNTHETIC"
        ]
        if capacity_provenance not in valid_provenance:
            raise ValueError(f"Invalid capacity_provenance '{capacity_provenance}'. Must be one of {valid_provenance}")
        record["capacity_provenance"] = capacity_provenance

    return record
