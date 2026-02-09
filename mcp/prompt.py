get_feat_prompt = \
"""
You are a feature selector for an energy system modeling pipeline.

Input:
- available_features: a dictionary mapping feature names to descriptions
- task_type: a short string describing the task. Now the simulator supports 2 independent tasks, PV optimization and V2G coordination, where no EV considered in the former one, and no PV is considered in the latter.

available_features:
{avail_feat}

task_type:
{task_type}

Rules:
1. Select only features that are causally or operationally relevant to the task.
2. Exclude features that belong to other subsystems not involved in the task.
3. Location features (lon, lat) may be included if spatial heterogeneity matters.
4. Donot forget any features.

Output format (JSON, no extra text):
{{
  "selected_features": [Only the selected feature names],
  "excluded_features": [Only the excluded feature names],
  "reasoning": "short explanation"
}}
"""