import torch.utils.data as data
import albumentations as albu
import albumentations.pytorch as albupt
import cv2

class BaseDataset(data.Dataset):
    def __init__(self, root, split='train'):
        super(BaseDataset, self).__init__()
        self.img_base = None
        self.label_base = None
        self.imgs_path = None # or self.labels_path = None, get the list which stores the paths of all images/labels
        self.split = split

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label_path = None
        # use cv2 to read images so that the datatype is unit8, and can use albumentation library
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = None

        sample = {'image':img, 'mask':label}
        if self.split == 'train':
            return self.transform_train(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_test(sample)


    def __len__(self):
        return len(self.imgs_path)

    def transform_train(self, sample):
        composed_transform = albu.Compose([
            albu.VerticalFlip(),
            albu.HorizontalFlip(),
            albu.Normalize(), # different form torchvision.transforms, dtype shoud be uint8, /255 -mean /std
            albupt.ToTensorV2()
        ])
        augmented = composed_transform(image=sample['image'], mask=sample['mask'])
        return augmented['image'], augmented['mask']

    def transform_val(self, sample):
        """No random operation so that validation set always keeps the same."""
        composed_transform = albu.Compose([
            # albu.resize(512, 1024, p=1),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albupt.ToTensorV2()
        ])
        transformed = composed_transform(image=sample['image'], mask=sample['mask'])
        return transformed['image'], transformed['mask']

    def transform_test(self, sample):
        """No random operation so that test set always keeps the same."""
        composed_transform = albu.Compose([
            # albu.resize(512, 1024, p=1),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albupt.ToTensorV2()
        ])
        transformed = composed_transform(image=sample['image'], mask=sample['mask'])
        return transformed['image'], transformed['mask']
