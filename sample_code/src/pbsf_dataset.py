from torch.utils.data import Dataset


class PBSFDataset(Dataset):

    def __init__(self, data, transform):
        self.values = [x["values"] for x in data]
        self.targets = [x["target"] for x in data]
        self.transform = transform

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        x = self.values[idx]
        x[x > 23] = 23
        x[x < -23] = -23
        return self.transform(x), self.targets[idx]
