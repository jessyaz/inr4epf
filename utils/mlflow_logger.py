import mlflow
from dotenv import load_dotenv
from omegaconf import OmegaConf

class MLflowLogger:
    def __init__(self, cfg, tester_flag=False):
        load_dotenv("./.env", override=True)

        self.cfg = cfg

        self.experiment_name = self.cfg.mlflow.experiment_name
        self.run_name = self.cfg.mlflow.run_name

        self.tester_flag = tester_flag

    def __enter__(self):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else mlflow.create_experiment(self.experiment_name)
        )


        mlflow.end_run()
        mlflow.start_run(experiment_id=experiment_id, run_name=self.run_name)

        self.log_config(self.cfg)

        return self



    def __exit__(self, *args):
        status = "FINISHED" if self.tester_flag else "FAILED"
        mlflow.end_run(status)

    def log_config(self, cfg):
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")


    def log_checkpoint(self, path):
        mlflow.log_artifact(path)


    def log_plot(self, fig, artifact_path="plots"):
        mlflow.log_figure(fig, artifact_path)


    def log_metrics(self, loss_dict, epoch, prefix="train"):
        metrics_to_log = {
            f"{prefix}/{key}": value for key, value in loss_dict.items()
        }
        mlflow.log_metrics(metrics_to_log, step=epoch)