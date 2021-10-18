from src.configs import BaseModelConfig, MoCoConfig, MoCoTrainerConfig, ContrastiveTrainerConfig


class AverageMeter:
    def __init__(self, name, format=".2f"):
        self.name = name
        self.format = format
        self.sum = None
        self.num = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0
        self.avg = 0
        self.val = 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val * num
        self.num += num
        self.avg = self.sum / self.num if self.num != 0 else 0

    def __str__(self):
        return f"{self.name} : {self.val: {self.format}} (avg: {self.avg: {self.format}})"


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def create_moco_configs(config):
    meta_config = config["MetaConfig"]
    base_model_config = BaseModelConfig()
    moco_config = MoCoConfig()
    trainer_config = MoCoTrainerConfig()
    base_model_config.parse_dict(config["BaseModelConfig"])
    moco_config.parse_dict(config["MoCoConfig"])
    trainer_config.parse_dict(config["TrainerConfig"])
    base_model_config.output_size = meta_config["feature_dim"]
    moco_config.feature_dim = meta_config["feature_dim"]
    trainer_config.name = meta_config["name"]
    return base_model_config, moco_config, trainer_config


def create_contrastive_config(config):
    meta_config = config["MetaConfig"]
    base_model_config = BaseModelConfig()
    trainer_config = ContrastiveTrainerConfig()
    base_model_config.parse_dict(config["BaseModelConfig"])
    trainer_config.parse_dict(config["TrainerConfig"])
    base_model_config.output_size = meta_config["feature_dim"]
    trainer_config.name = meta_config["name"]
    return base_model_config, trainer_config
