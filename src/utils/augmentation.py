import cv2
import albumentations as A

class CigaretteLighterTransform(object):
    def __init__(self, size=64, mode='generate'):
        self.mode = mode
        self.size = size

    def __call__(self, image):
        data_transform = {
            'generate': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Resize(
                    height = self.size, 
                    width = self.size,
                    p = 1
                ),
                A.ShiftScaleRotate(
                    shift_limit=(-0.05, 0.05), 
                    scale_limit=(0, 0), 
                    rotate_limit=(-45, 45)
                )
            ])
        }

        return data_transform[self.mode](image=image)