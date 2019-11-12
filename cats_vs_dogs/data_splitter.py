
import os
import sys
import shutil


def copy_and_split_data(original_dataset_dir, base_dir):
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)

    print 'Copying train cats...'
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    print 'Done'

    print 'Copying validation cats...'
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    print 'Done'

    print 'Copying test cats...'
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    print 'Done'

    print 'Copying train dogs...'
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    print 'Done'

    print 'Copying validation dogs...'
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    print 'Done'

    print 'Copying test dogs...'
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
    print 'Done'

    print 'total training cat images:', len(os.listdir(train_cats_dir))
    print 'total training dog images:', len(os.listdir(train_dogs_dir))
    print 'total validation cat images:', len(os.listdir(validation_cats_dir))
    print 'total validation dog images:', len(os.listdir(validation_dogs_dir))
    print 'total test cat images:', len(os.listdir(test_cats_dir))
    print 'total test dog images:', len(os.listdir(test_dogs_dir))


if __name__ == '__main__':
    assert len(sys.argv) >= 2, 'Data source directory missing; ' \
                               'Data target directory missing'
    assert len(sys.argv) > 2, 'Data target directory missing'
    copy_and_split_data(sys.argv[1], sys.argv[2])
