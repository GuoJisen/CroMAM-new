from torchvision import transforms


def get_transformation(mean=None, std=None):
    """
    Get data augmentation for different dataset
    """
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                # transforms.RandomRotation(90),
                # transforms.ColorJitter(brightness=0.35,
                #                        contrast=0.5,
                #                        saturation=0.1,
                #                        hue=0.16),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]),
        'val':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
    }
    return data_transforms
