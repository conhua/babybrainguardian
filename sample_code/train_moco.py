from src.encoders.graph_attention import GraphAttnEncoder
from src.trainers.moco import MoCo, MoCoTrainer
from src.utils import create_moco_configs
import yaml
import os
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, default="config_files/moco_config.yaml")


if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file, "r"))
    save_dir = config["MetaConfig"]["save_dir"]
    name = config["MetaConfig"]["name"]
    os.makedirs(os.path.join(save_dir, name), exist_ok=True)
    shutil.copy(args.config_file, os.path.join(save_dir, name, "config.yaml"))
    base_model_config, moco_config, trainer_config = create_moco_configs(config)
    encoder_q = GraphAttnEncoder(base_model_config)
    encoder_k = GraphAttnEncoder(base_model_config)
    moco_model = MoCo(encoder_q, encoder_k, moco_config)
    moco_trainer = MoCoTrainer(moco_model, trainer_config)
    moco_trainer.train()
