import mlflow
import yaml
import tempfile
from dotenv import load_dotenv

load_dotenv("./.env", override=True)

client = mlflow.tracking.MlflowClient()
runs = mlflow.search_runs(experiment_ids=['61'])
runs = runs.sort_values('metrics.test/rMAE')

for _, row in runs.iterrows():
    run_id = row['run_id']
    run_name = row.get('tags.mlflow.runName', '')
    rmae = row.get('metrics.test/rMAE')
    print(f"\n=== {run_name} ({run_id[:8]}) — test/rMAE = {rmae} ===")

    try:
        # log_config utilise mlflow.log_dict(cfg, "config.yaml") -> artefact à la racine
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = client.download_artifacts(run_id, "config.yaml", tmp_dir)
            with open(local_path) as f:
                cfg = yaml.safe_load(f)
            print(yaml.dump(cfg, default_flow_style=False))
    except Exception as e:
        print(f"  Erreur: {e}")