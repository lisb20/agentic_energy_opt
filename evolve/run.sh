export OPENAI_API_KEY="sk-xxx"

python ./openevolve-run.py \
  ./initial_program_v3.py \
  ./evaluator.py \
  --config ./config.yaml \
  --log-level INFO