import pickle

from src.encoders.graph_attention import GraphAttnEncoder
from src.trainers.contrastive import ContrastiveTrainer
from src.trainers.moco import MoCo, MoCoTrainer
from src.utils import create_contrastive_config, create_moco_configs
import yaml
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, default="")
parser.add_argument("--method", type=str, default="contrastive")
parser.add_argument("--model-ckpt", type=str, default="best_ckpt.pt")
parser.add_argument("--save-dir", type=str, default="features/20210512/contrastive")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.method == "moco":
        config = yaml.safe_load(open(args.config_file, "r"))
        base_model_config, moco_config, trainer_config = create_moco_configs(config)
        encoder_q = GraphAttnEncoder(base_model_config)
        encoder_k = GraphAttnEncoder(base_model_config)
        moco = MoCo(encoder_q, encoder_k, moco_config)
        trainer = MoCoTrainer(moco, trainer_config)
        trainer.load_checkpoint(args.model_ckpt)
        train_reps = trainer.inference(trainer.train_cls_set)
        test_reps = trainer.inference(trainer.test_cls_set)
        save_dir = os.path.join(args.save_dir, config["MetaConfig"]["name"])
        os.makedirs(save_dir, exist_ok=True)
        features = {
            "train": {
                "features": train_reps,
                "labels": trainer.train_cls_set.targets
            },
            "test": {
                "features": test_reps,
                "labels": trainer.test_cls_set.targets
            }
        }
        with open(os.path.join(save_dir, "features.pkl"), "wb") as f:
            pickle.dump(features, f)
    elif args.method == "contrastive":
        config = yaml.safe_load(open(args.config_file, "r"))
        base_model_config, trainer_config = create_contrastive_config(config)
        trainer_config.device = "cuda:1"
        encoder = GraphAttnEncoder(base_model_config)
        trainer = ContrastiveTrainer(encoder, trainer_config)
        trainer.load_checkpoint(args.model_ckpt)
        train_reps = trainer.inference(trainer.train_cls_set)
        test_reps = trainer.inference(trainer.test_cls_set)
        save_dir = os.path.join(args.save_dir, config["MetaConfig"]["name"])
        os.makedirs(save_dir, exist_ok=True)
        features = {
            "train": {
                "features": train_reps,
                "labels": trainer.train_cls_set.targets
            },
            "test": {
                "features": test_reps,
                "labels": trainer.test_cls_set.targets
            }
        }
        with open(os.path.join(save_dir, "features.pkl"), "wb") as f:
            pickle.dump(features, f)
