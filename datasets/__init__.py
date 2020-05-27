from datasets.core.cityscapes import CityScapes

def make_dataset(dataset, root, split):
    if dataset == 'cityscapes':
        return CityScapes(root, split)