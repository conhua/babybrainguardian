def create_config_class(cls_name, cls_base, attrs):
    def parse_dict(obj, configs):
        for attr in obj.properties:
            setattr(obj, attr, configs.get(attr, None))

    def __str__(obj):
        res = ""
        sep = ", "
        end = "."
        for attr in obj.properties:
            res += attr + "=" + str(getattr(obj, attr)) + sep
        return res.strip(sep) + end
    cls_attrs = {attr: None for attr in attrs}
    cls_attrs["properties"] = attrs
    cls_attrs["parse_dict"] = parse_dict
    cls_attrs["__str__"] = __str__
    cls_attrs["__repr__"] = __str__
    ConfigClass = type(cls_name, cls_base, cls_attrs)
    return ConfigClass


_base_model_config = ["data_dim", "data_length", "input_size", "output_size", "hidden_size"]
_moco_config = ["queue_size", "momentum", "temperature", "feature_dim", "normalize"]
_moco_trainer_config = ["device", "lr", "batch_size", "start_epoch", "epochs", "save_freq",
                        "print_freq", "dataset_path", "name", "momentum", "weight_decay", "cos",
                        "schedule", "save_dir", "resized_output_size", "crop_scale"]
_contrastive_trainer_config = _moco_trainer_config + ["warm", "warm_epochs", "warmup_from", "warmup_to",
                                                      "lr_decay_rate"]

BaseModelConfig = create_config_class("BaseModelConfig", (), _base_model_config)
MoCoConfig = create_config_class("MoCoConfig", (), _moco_config)
MoCoTrainerConfig = create_config_class("MoCoTrainerConfig", (), _moco_trainer_config)
ContrastiveTrainerConfig = create_config_class("ContrastiveTrainerConfig", (), _contrastive_trainer_config)

__all__ = ["BaseModelConfig", "MoCoConfig", "MoCoTrainerConfig", "ContrastiveTrainerConfig"]


if __name__ == "__main__":
    config_base = BaseModelConfig()
    config_moco = MoCoConfig()
    config_trainer = MoCoTrainerConfig()
    print(config_base)
    print(config_moco)
    print(config_trainer)
