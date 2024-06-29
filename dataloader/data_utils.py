from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def transforms_for_pretrain(resize=224, mean_std=None):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.CenterCrop(resize),
                                         transforms.RandomRotation(degrees=360),
                                         transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def transforms_for_finetune(resize=224, mean_std=None, p=0.2):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.CenterCrop(resize), transforms.RandomHorizontalFlip(),
                                         transforms.RandomGrayscale(p), transforms.RandomRotation(degrees=360),
                                         transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms

