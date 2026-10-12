import logging
import torchvision.transforms as T


def get_augmentation_transform(aug_list: list, image_res: int):
    """
    Builds a torchvision.transforms.Compose object from a list of augmentation names.
    """
    transforms_list = []
    if not aug_list:
        return T.Compose([])

    logging.info("Initializing Augmentations")
    for aug_name in aug_list:
        if aug_name == "hflip":
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
            logging.info("Added augmentation: RandomHorizontalFlip(p=0.5)")
        elif aug_name == "vflip":
            transforms_list.append(T.RandomVerticalFlip(p=0.5))
            logging.info("Added augmentation: RandomVerticalFlip(p=0.5)")
        elif aug_name == "rotate":
            # 加上 fill=(255, 255, 255) 保证数据增强旋转后也是白底
            transforms_list.append(T.RandomRotation(degrees=(0, 345), fill=(255, 255, 255)))
            logging.info("Added augmentation: RandomRotation(degrees=(0, 345), fill=(255, 255, 255))")
        elif aug_name == "color_jitter":
            transforms_list.append(
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )
            logging.info(
                "Added augmentation: ColorJitter(brightness=0.2, contrast=0.2, ...)"
            )
        elif aug_name == "affine":
            transforms_list.append(
                T.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10)
            )
        else:
            logging.warning(f"Unknown augmentation '{aug_name}' requested. Ignoring.")

    if not transforms_list:
        logging.warning(
            "Augmentation requested but no valid augmentations found in aug_list."
        )
        return T.Compose([])

    return T.Compose(transforms_list)
