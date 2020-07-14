from torch.utils.data import Dataset
from os import listdir
from PIL import Image
from torchvision import transforms


class Edges2ShoesDataset(Dataset):
    def __init__(self, args, is_test=False):
        super().__init__()
        image_dir = args.dataset + ('/val' if is_test else '/train')
        self.fnames = [image_dir+'/'+fname for fname in listdir(image_dir)]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx])
        img = self.transform(img)
        x, y = img[:, :, :256], img[:, :, 256:]
        return x, y

    def __len__(self):
        return len(self.fnames)
