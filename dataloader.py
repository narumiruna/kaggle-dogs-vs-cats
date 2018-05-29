import glob
import os

from PIL import Image
from torch.utils import data
from torchvision import transforms


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


def test():
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.ToTensor()])
    dataset = DogsVsCats('data', transform=transform)
    loader = data.DataLoader(dataset, batch_size=128, shuffle=True)
    for x, y in loader:
        print(x.size())
        print(y.size())
        break


if __name__ == '__main__':
    test()
