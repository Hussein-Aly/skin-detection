from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Lambda, ToTensor, Compose

from utils.data_augmentation import pad_tensor


class BaseDataset(Dataset):
    def __init__(self, file_paths, target):
        self.image_paths = file_paths
        self.target = target
        self.transform = Compose([
            ToTensor(),
            Lambda(lambda tensor: pad_tensor(tensor))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        return image, self.target


class SkinDataset(BaseDataset):
    def __init__(self, skin_file_paths):
        super().__init__(skin_file_paths, target=1)


class NonSkinDataset(BaseDataset):
    def __init__(self, non_skin_file_paths):
        super().__init__(non_skin_file_paths, target=0)
