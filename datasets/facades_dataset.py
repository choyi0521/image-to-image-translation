from torch.utils.data import Dataset
from os import listdir
from PIL import Image
from torchvision import transforms


class FacadesDataset(Dataset):
    def __init__(self, args, dataset_type):
        super().__init__()
        if dataset_type not in ['train', 'val', 'test']:
            raise Exception('No such dataset_type')
        image_dir = args.dataset + '/' + dataset_type
        self.fnames = [image_dir+'/'+fname for fname in listdir(image_dir)]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        img = Image.open(self.fnames[idx]).convert('RGB')
        img = self.transform(img)
        x, y = img[:, :, 256:], img[:, :, :256]
        return x, y

    def __len__(self):
        return len(self.fnames)
