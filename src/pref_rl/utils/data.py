import torch
from torch.utils.data import Dataset


def segment_collate_fn(batch):
    segments, preferences, weights, segment_metas, mask = zip(*batch)

    segments = torch.stack(segments, 0)
    preferences = torch.stack(preferences, 0) 
    weights = torch.stack(weights, 0)
    mask = torch.tensor(mask)

    return segments, preferences, weights, segment_metas, mask


class SegmentDataset:
    def __init__(self, segments, preferences, weights, segment_metas):
        self.segments = segments
        self.preferences = preferences
        self.weights = weights
        self.segment_metas = segment_metas

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx], self.preferences[idx], self.weights[idx], self.segment_metas[idx]


class MaskedDataset(Dataset):
    def __init__(self, dataset, mask_value):
        self.dataset = dataset
        self.mask_value = mask_value
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        if isinstance(item, (tuple, list)):
            item = (*item, self.mask_value)
        else:
            item = (item, self.mask_value)
        
        return item
