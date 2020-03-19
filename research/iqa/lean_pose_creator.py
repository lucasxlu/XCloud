import random
import argparse

import Augmentor

parser = argparse.ArgumentParser()
parser.add_argument('-img_dir', type=str,
                    default='/Users/lucasx/Documents/Dataset/frontal')
parser.add_argument('--sample_num', type=int, default=10)
args = vars(parser.parse_args())

print('-' * 100)
for key, value in args.items():
    print('%s = %s' % (key, value))
print('-' * 100)


def generate_lean_samples():
    """
    generate lean shelf pose samples from frontal shelf pose images
    :return:
    """
    p = Augmentor.Pipeline(args['img_dir'])
    # p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
    p.skew_left_right(probability=1, magnitude=random.randint(7, 10) / 10)
    # p.skew_top_bottom(probability=1, magnitude=random.randint(6, 10) / 10)
    # p.random_contrast(0.5, min_factor=0.5, max_factor=1.05)
    # p.random_distortion(probability=0.2, grid_width=2, grid_height=2, magnitude=2)

    # p.sample(args['sample_num'])
    p.process()


if __name__ == '__main__':
    generate_lean_samples()
