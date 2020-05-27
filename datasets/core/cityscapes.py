import torch.utils.data as data
import os
from datasets.utils import recursive_glob
import cv2
import albumentations as albu
import albumentations.pytorch as albupt


class CityScapes(data.Dataset):
    def __init__(self, root='/usr/xtmp/vision/datasets/Cityscapes', split='train'):
        super(CityScapes, self).__init__()
        self.img_base = os.path.join(root, 'leftImg8bit_trainvaltest', 'leftImg8bit', split)
        self.label_base = os.path.join(root, 'gtFine_trainvaltest', 'gtFine', split)
        self.imgs_path = recursive_glob(rootdir=self.img_base, suffix='.png')
        # see labels: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.split = split

    def __getitem__(self, index):
        # image
        img_path = self.imgs_path[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # label
        img_name = '_'.join(img_path.split('_')[:-1]).split(os.sep)[-1]
        label_path = os.path.join(self.label_base, img_path.split(os.sep)[-2], img_name + '_gtFine_labelIds.png')
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = self.encode_segmap(np.array(label))

        if self.split == 'train':
            return self.transform_train(img, label)
        elif self.split == 'val':
            return self.transform_val(img, label)
        elif self.split == 'test':
            return self.transform_test(img, label)

    def __len__(self):
        return len(self.imgs_path)

    def encode_segmap(self, mask):
        """Encode mask ID to trainID"""
        for id in self.void_classes:
            mask[mask == id] = 255
        for new_id, id in enumerate(self.valid_classes):
            mask[mask == id] = new_id
        return mask

    def transform_train(self, img, label):
        composed_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.RandomScale(scale_limit=0.2, p=1),
            albu.RandomCrop(512, 512, p=1),
            albu.GaussianBlur(p=0.5),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albupt.ToTensorV2()
        ])
        augmented = composed_transform(image=img, mask=label)
        return augmented['image'], augmented['mask']

    def transform_val(self, img, mask):
        # no random operation so that validation set always keeps the same
        composed_transform = albu.Compose([
            albu.Resize(512, 1024),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albupt.ToTensorV2()
        ])
        transformed = composed_transform(image=img, mask=mask)
        return transformed['image'], transformed['mask']

    def transform_test(self, img, mask):
        composed_transform = albu.Compose([
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albupt.ToTensorV2()
        ])
        transformed = composed_transform(image=img, mask=mask)
        return transformed['image'], transformed['mask']

if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    cityscapes = CityScapes(split='val')
    train_loader = DataLoader(cityscapes, shuffle=False, batch_size=4, num_workers=2)

    fig, axes = plt.subplots(2, 4)
    for i, sample in enumerate(train_loader):
        images = sample[0].numpy()
        masks = sample[1].numpy()
        for j in range(images.shape[0]):
            img = images[j,:,:,:].transpose(1,2,0)
            img = img * (0.229, 0.224, 0.225)
            img = img + (0.485, 0.456, 0.406)
            mask = masks[j,:,:] * 10
            axes[0,j].imshow(img)
            axes[1,j].imshow(mask)
        # save the current figure
        plt.savefig("/home/home1/xw176/work/frameworks/DeepLab_v3+/test_output/{}.png".format(i))
        print("save {}".format(i))


