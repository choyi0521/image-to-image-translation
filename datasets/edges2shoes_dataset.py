from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Edges2ShoesDataset(Dataset):
    def __init__(self, args, is_test=False):
        super().__init__()
        self.args = args

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return

    def __getitem__(self, idx):
        return