import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
from pathlib import Path
import datetime
import torch



from utils.mlflow_logger import MLflowLogger
from datasets.loader import load_market_dataloader
from utils.trainer import train as trainer
from utils.tester import test as tester

from models.test_model import Model as test_model
from models.lstm_3008 import Model as lstm_3008
from models.naif.naive_model import Model as naive_model

#Git models
from models.git_interfaces.epf_transformer import Model as epf_transformer
from models.git_interfaces.timexer_interface import Model as timexer

from models.git_interfaces.timeseries_lib_interface import Model as timeseries_lib



MODEL_REGISTRY = {
    "test_model": test_model,
    "lstm_3008":lstm_3008,
    "naive_model": naive_model,
    "epf_transformer": epf_transformer,
    "timexer": timexer,


    "patchtst": timeseries_lib,
    "dlinear": timeseries_lib,
    "itransformer": timeseries_lib,
}


def instantiate_model(cfg):
    model_name = cfg.registry

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"registry inconnu : '{model_name}'. Disponibles : {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](cfg)


# uv run runner.py registry=test_model --config-name link2conf

@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):

    model_uid = uuid.uuid4().hex[:8]
    cfg.model_uid = model_uid

    print(f"[runner] model_uid = {model_uid} - [experiment] {cfg.mlflow.experiment_name}")
    run_dir = Path("runs") / f"{model_uid}_{cfg.registry}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.run_dir = run_dir.as_posix()

    cfg.mlflow.run_name = cfg.registry + "_" + model_uid

    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    print(" ---------- Using device : ", device, " ----------")

    try:
        dataset_name = cfg.dataset.name
    except:
        raise ValueError(f"Dataset error : '{cfg.dataset}'")

    train_loader, val_loader, test_loader, scaler = load_market_dataloader(cfg.dataset.name, batch_size=cfg.dataset.batch_size, missing_rate=cfg.dataset.missing_rate, seed=cfg.seed)
    for x, y in train_loader:
        print('Shape X_exog:', x.shape)
        print('Shape Y_target:', y.shape)
        break

    model = instantiate_model(cfg).to(device)

    with MLflowLogger(cfg) as logger:

        print("Training ...")
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)

        loaders = {'train_loader':train_loader ,'val_loader':val_loader}
        loss_dict = trainer(model, loaders, optimizer, device, logger)

        print(f"Training finish with : {loss_dict}")
        torch.save(model.state_dict(), run_dir / "model.pth")
        logger.log_checkpoint(str(run_dir / "model.pth"))

        print("test en cours...")
        loss_dict_test = tester(model, test_loader, scaler, device, logger)

        print(f"Finish : {loss_dict_test}")

        logger.tester_flag = True

    return loss_dict["val_loss"]["MAE"]


if __name__ == "__main__":

    main()