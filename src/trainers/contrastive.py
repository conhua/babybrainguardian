import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from src.utils import AverageMeter,ProgressMeter
from sklearn.metrics import roc_auc_score
import time
import pickle
import src.transforms as transforms
from src.pbsf_dataset import PBSFDataset
import lightgbm as lgb
import numpy as np
import math


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07, device="cpu"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        self.device = device

    def forward(self, features, labels=None, mask=None):
        batch_size = features.size(0)
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not equal to number of features.")
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.size(1)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode {self.contrast_mode}")

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-30)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveTrainer:
    def __init__(self, model, config):
        super(ContrastiveTrainer, self).__init__()
        self.model = model
        self.config = config
        self.train_set, self.train_cls_set, self.test_cls_set = self.create_dataset()
        self.optimizer = self.create_optimizer()
        self.device = config.device
        self.criterion = SupConLoss(device=config.device)
        self.model.to(config.device)
        self.save_dir = os.path.join(config.save_dir, config.name)

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

        for i, (inputs, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            batch_size = labels.size(0)
            inputs = torch.cat([inputs[0], inputs[1]], dim=0)
            inputs = inputs.to(self.device)
            # self.warmup_learning_rate(epoch, i, len(train_loader))
            features = self.model(inputs)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            f = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = self.criterion(f, labels)
            losses.update(loss.item(), labels.size(0))
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
                reps = self.model(inputs)
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
            eta_min = lr * (self.config.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / self.config.epochs)
            ) / 2
        else:
            steps = np.sum(epoch > np.asarray(self.config.schedule))
            if steps > 0:
                lr = lr * (self.config.lr_decay_rate ** steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def warmup_learning_rate(self, epoch, batch_id, total_batches):
        lr = self.config.lr
        if self.config.warm and epoch <= self.config.warm_epochs:
            p = (batch_id + (epoch - 1) * total_batches) / \
                (self.config.warm_epochs * total_batches)
            lr = self.config.warmup_from + p * (self.config.warmup_to - self.config.warmup_from)

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


if __name__=="__main__":
    pass
