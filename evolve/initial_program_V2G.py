import numpy as np

def Action(lon, lat, demand_profile, target_pattern, N_ev_node, closest_PP_id):
    """
    <Parameters>
    lon: longitude of the location
    lat: latitude of the location
    demand_profile: np.array, shape (96,), representing the demand profile (MW) of the location at 0.25h (15min) resolution.
    target_pattern: np.array, shape (96,), representing the V2G target (MW) of the whole city the location at 0.25h (15min) resolution.
    N_ev_node: np.array of int, shape (96,), representing the number of EV nodes in the location at 0.25h (15min) resolution.
    closest_PP_id: int [0-24], representing the id of the closest power plant to the location.
    <Returns>
    Reginal_V2G_target: np.array, shape (96,) of float number. The V2G target (MW) of the location at 0.25h (15min) resolution.
    """
    Reginal_V2G_target = target_pattern * np.mean(N_ev_node) / 250000

    return Reginal_V2G_target