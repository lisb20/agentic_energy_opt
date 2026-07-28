import numpy as np


def Action(lon, lat, demand_profile, solar_profile, rooftop_area, closest_PP_id):
    """Return a non-negative PV deployment priority for one grid node."""
    demand = np.asarray(demand_profile, dtype=float)
    solar = np.asarray(solar_profile, dtype=float)
    alignment = np.dot(demand, solar) / (
        np.linalg.norm(demand) * np.linalg.norm(solar) + 1e-12
    )
    area_score = np.log1p(max(float(rooftop_area), 0.0))
    return float(max(alignment, 0.0) * area_score)
