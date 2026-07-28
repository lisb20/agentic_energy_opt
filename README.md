# Supplementary code: LLM-guided energy planning

This anonymized package contains the code used for LLM-guided photovoltaic
(PV) planning and vehicle-to-grid (V2G) coordination. It is intended for
double-blind supplementary review.

## Availability statement

The source code for the agent, simulation adapters, evaluator, and evolutionary
optimizer is included. The empirical electricity-network topology, regional
load inputs, roof-area mapping, and vehicle mobility trajectories cannot be
redistributed because of data-use and privacy restrictions. No empirical data,
model credentials, machine-specific paths, author identities, or experiment
logs are included.

The required private-data interfaces are documented in
[`data/README.md`](data/README.md). Researchers with an authorized dataset can
provide compatible files without changing the source code.

## Package layout

```text
agent/       Task parsing and simulation-agent orchestration
mcp/         MCP tools exposed to the agent
simulator/   Power-flow, PV, and V2G simulation adapters
evolve/      Evaluation logic and initial PV/V2G programs
openevolve/  Vendored OpenEvolve runtime (see its LICENSE)
data/        Interface documentation only; no empirical data
```

## Installation

Python 3.10 or newer is required.
The supplementary package was tested with Python 3.11 and the pinned direct
dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

Configure an OpenAI-compatible endpoint and the private-data location:

```bash
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1
export OPENAI_MODEL=YOUR_AGENT_MODEL
export OPENEVOLVE_MODEL=YOUR_EVOLUTION_MODEL
export DATA_DIR=/path/to/authorized/data
```

For V2G experiments, also provide:

```bash
export V2G_DRIVING_POWER=/path/to/driving_power.npy
export V2G_LOCATION=/path/to/vehicle_locations.npy
```

## Running the agent

Start the MCP simulation service:

```bash
python mcp/MCP.py
```

In another terminal, submit a task:

```bash
python agent/main.py \
  'Load a PV plan from "initial_plan.npy" and optimize it to the target capacity.'
```

The MCP port defaults to `13579` and can be changed with `MCP_PORT`.

## Running the evolutionary component

Normally the agent creates `cache/state_to_opt.pkl` and invokes the optimizer.
For an authorized pre-built state, the optimizer can also be started directly:

```bash
DOMAIN=PV bash evolve/run.sh
```

Set `DOMAIN=V2G` for V2G coordination. The default number of iterations comes
from `evolve/config.yaml`; `OPENEVOLVE_ITERATIONS` provides a temporary command-
line override.

## Third-party software

The `openevolve/` directory contains a reduced vendored copy of OpenEvolve.
Its original license and attribution files are retained in that directory.
