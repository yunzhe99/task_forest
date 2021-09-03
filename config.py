class config:
    training_set = '/mnt/disk/TJU-DHD/dhd_traffic/train.txt'  # the dir of the whole training set
    training_dir = '/mnt/disk/TJU-DHD/dhd_traffic/trainset/images/'

    min_train = 500  # the minimun sample size that a tree can be trained

    nc = 5  # the number of classes for detection
    names = [ 'Pedestrian', 'Cyclist', 'Car', 'Truck', 'Van']  # the name of classes for detection
    val = '/mnt/disk/TJU-DHD/dhd_traffic/val.txt'  # the path of the dataset for val

    # for yolov5 training
    batch_size = 96  # the size of a training batch
    device = 2  # the device selected for training
    epoch_num = 40  # the number of training epoch
