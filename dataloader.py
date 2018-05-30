import glob
import os

from PIL import Image
from torch.utils import data
from torchvision import transforms

from torch.utils import data
from torch.utils.data.dataset import random_split


def pil_loader(f):
    with open(f, 'rb') as fp:
        img = Image.open(fp)
        return img.convert('RGB')


class DogsVsCats(data.Dataset):
    train_dir = 'train'

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.samples = []
        for path in glob.glob(os.path.join(self.root, self.train_dir, '*.jpg')):
            class_index = 0 if 'cat' in path else 1
            self.samples.append((path, class_index))

    def __getitem__(self, index):
        path, class_index = self.samples[index]
        img = pil_loader(path)

        if self.transform:
            img = self.transform(img)

        return img, class_index

    def __len__(self):
        return len(self.samples)


def get_dataloader(size, root, batch_size, valid_ratio=0.2, num_workers=0):
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(size),
         transforms.ToTensor()])

    dataset = DogsVsCats(root, transform=transform)

    # split dataset
    n_samples = len(dataset)
    n_valid_samples = int(len(dataset) * valid_ratio)
    train_set, valid_set = random_split(
        dataset, [n_samples - n_valid_samples, n_valid_samples])

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_loader, valid_loader


def test():
    train_loader, _ = get_dataloader(224, 'data', 128)
    x, y = next(iter(train_loader))
    print(x.size())
    print(y.size())


if __name__ == '__main__':
    test()
