import torch.nn as nn
import torch
import time
import math
from torch.utils.data import DataLoader
from src.pbsf_dataset import PBSFDataset
from src.utils import AverageMeter, ProgressMeter
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import os
from src import transforms


class MoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k, moco_config):
        super(MoCo, self).__init__()
        self.queue_size = moco_config.queue_size
        self.momentum = moco_config.momentum
        self.temperature = moco_config.temperature
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.feature_dim = moco_config.feature_dim
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(self.feature_dim, self.queue_size))
        if moco_config.normalize:
            self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        if self.queue_size % batch_size != 0:  # for simplicity
            pass
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def encode(self, inputs):
        return self.encoder_q(inputs)

    def forward(self, input_q, input_k):
        q = self.encoder_q(input_q)  # N x C
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(input_k)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.size(0)).to(logits)
        labels = labels.type(torch.long)
        self._dequeue_and_enqueue(k)
        return logits, labels


class MoCoTrainer:
    def __init__(self, model, config):
        super(MoCoTrainer, self).__init__()
        self.model = model
        self.config = config
        self.train_set, self.train_cls_set, self.test_cls_set = self.create_dataset()
        self.optimizer = self.create_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.device = config.device
        self.model.to(config.device)
        self.save_dir = os.path.join(config.save_dir, config.name)
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        train_loader = DataLoader(
            self.train_set, batch_size=self.config.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True
        )
        epoch_time = AverageMeter("Epoch Time")
        best_auc = 0
        for epoch in range(self.config.start_epoch, self.config.epochs):
            start = time.time()
            self.adjust_learning_rate(epoch)
            self.train_epoch(train_loader, epoch)
            classifier = self.train_classifier()
            preds, proba = self.test_classifier(classifier)
            targets = self.test_cls_set.targets
            auc = roc_auc_score(targets, proba)
            if auc > best_auc:
                self.save_checkpoint("best_ckpt.pt")
                best_auc = auc
            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(f"epoch_{epoch}_ckpt.pt")
            epoch_time.update(time.time() - start)
            print(f"[Epoch {epoch}] | {str(epoch_time)} | AUC: {auc} | Best AUC: {best_auc}")
        self.save_checkpoint("final.pt")

    def train_epoch(self, train_loader, epoch):
        batch_time = AverageMeter("Batch Time", ".2f")
        data_time = AverageMeter("Data Time", ".2f")
        losses = AverageMeter("Loss", ".4f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix=f"Epoch: [{epoch}]"
        )

        self.model.train()
        end = time.time()

        for i, (inputs, _) in enumerate(train_loader):
            input_q = inputs[0]
            input_k = inputs[1]
            data_time.update(time.time() - end)
            input_q = input_q.to(self.device)
            input_k = input_k.to(self.device)
            output, target = self.model(input_q, input_k)
            loss = self.criterion(output, target)

            losses.update(loss.item(), input_q.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.config.print_freq == 0:
                progress.display(i)

    def train_classifier(self):
        classifier = lgb.LGBMClassifier(num_leaves=5, n_estimators=150, learning_rate=0.05)
        representations = self.inference(self.train_cls_set)
        targets = self.train_cls_set.targets
        classifier.fit(representations, targets)
        return classifier

    def test_classifier(self, classifier):
        representations = self.inference(self.test_cls_set)
        proba = classifier.predict_proba(representations)
        preds = classifier.predict(representations)
        return preds, proba[:, 1]

    def inference(self, dataset):
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False
        )
        self.model.eval()
        representations = []
        with torch.no_grad():
            for _, (inputs, _) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                reps = self.model.encode(inputs)
                representations.append(reps.cpu().numpy())
        return np.concatenate(representations)

    def create_dataset(self):
        train_data, test_data = self.load_data(self.config.dataset_path, self.config.name)
        rep_transform = transforms.TwoCropTransform(
            transforms.RandomResizedCrop(self.config.resized_output_size, self.config.crop_scale)
        )
        cls_transform = transforms.ToTensor()
        train_set = PBSFDataset(train_data, rep_transform)
        train_cls_set = PBSFDataset(train_data, cls_transform)
        test_cls_set = PBSFDataset(test_data, cls_transform)
        return train_set, train_cls_set, test_cls_set

    def create_optimizer(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.config.lr,
                                    momentum=self.config.momentum,
                                    weight_decay=self.config.weight_decay)
        return optimizer

    def adjust_learning_rate(self, epoch):
        lr = self.config.lr
        if self.config.cos:
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / self.config.epochs))
        else:
            for m in self.config.schedule:
                lr *= 0.1 if epoch >= m else 1.
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def load_data(path, name):
        with open(path, "rb") as f:
            all_data = pickle.load(f)
        train_data = [x for x in all_data if x["name"] != name]
        test_data = [x for x in all_data if x["name"] == name]
        return train_data, test_data

    def save_checkpoint(self, fname):
        save_dir = os.path.join(self.save_dir, fname)
        checkpoints = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(checkpoints, save_dir)

    def load_checkpoint(self, fname):
        save_dir = os.path.join(self.save_dir, fname)
        checkpoints = torch.load(save_dir)
        self.model.load_state_dict(checkpoints["state_dict"])
        self.optimizer.load_state_dict(checkpoints["optimizer"])
