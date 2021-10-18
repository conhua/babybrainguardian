from src.encoders.graph_attention import GraphAttnEncoder
from src.trainers.contrastive import ContrastiveTrainer
from src.utils import create_contrastive_config
import yaml
import os
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, default="config_files/contrastive_config.yaml")


if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file, "r"))
    save_dir = config["MetaConfig"]["save_dir"]
    name = config["MetaConfig"]["name"]
    os.makedirs(os.path.join(save_dir, name), exist_ok=True)
    shutil.copy(args.config_file, os.path.join(save_dir, name, "config.yaml"))
    base_model_config, trainer_config = create_contrastive_config(config)
    model = GraphAttnEncoder(base_model_config)
    contrastive_trainer = ContrastiveTrainer(model, trainer_config)
    contrastive_trainer.train()
