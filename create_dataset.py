import os
import shutil
import glob
import random
import PIL


def split_data():
    classes = os.listdir('./dataset/train')

    for cls in classes:
        imgs = glob.glob(os.path.join('./dataset/train', cls, '*'))
        num_val = round(len(imgs) * 0.2)
        val_imgs = random.sample(imgs, num_val)
        dst_dir = os.path.join('./dataset/val', cls)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        for img in val_imgs:
            shutil.move(img, dst_dir)


if __name__ == '__main__':
    split_data()
