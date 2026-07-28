# Data interface

The empirical network and mobility datasets used in the paper cannot be
distributed. To run the code on an authorized or replacement dataset, set
`DATA_DIR` to a directory containing files with the following interfaces:

| File | Expected content |
| --- | --- |
| `flow_config.yaml` | Power-flow configuration consumed by `Flow`; include the dataset CRS in `projection` |
| `topology.json` | Network topology and node geometry |
| `power_demand.npy` | Regional demand array with shape `(N, T)` |
| `solar_profile.npy` | One-day solar profile |
| `region_area.pkl` | Mapping from region identifiers to usable roof area |
| `closest_generator.json` | Mapping from region identifiers to generator identifiers |

V2G experiments additionally require two private mobility arrays. Their paths
are supplied separately through `V2G_DRIVING_POWER` and `V2G_LOCATION`.

No empirical data are included in this supplementary code package.
