from torch.utils.data import Dataset


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
