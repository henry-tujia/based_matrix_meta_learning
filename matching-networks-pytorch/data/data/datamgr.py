# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide =100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader




class OmniglotNShotDataset():
    def __init__(self, batch_size, classes_per_set=20, samples_per_class=1, seed=2017, shuffle=True, use_cache=True):
        """
        Construct N-shot dataset
        :param batch_size:  Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        :param seed: seed for random function
        :param shuffle: if shuffle the dataset
        :param use_cache: if true,cache dataset to memory.It can speedup the train but require larger memory
        """
        np.random.seed(seed)
        self.x = np.load('data/data.npy')
        self.x = np.reshape(self.x, newshape=(self.x.shape[0], self.x.shape[1], 28, 28, 1))
        if shuffle:
            np.random.shuffle(self.x)
        self.x_train, self.x_val, self.x_test = self.x[:1200], self.x[1200:1411], self.x[1411:]
        # self.mean = np.mean(list(self.x_train) + list(self.x_val))
        self.x_train = self.processes_batch(self.x_train, np.mean(self.x_train), np.std(self.x_train))
        self.x_test = self.processes_batch(self.x_test, np.mean(self.x_test), np.std(self.x_test))
        self.x_val = self.processes_batch(self.x_val, np.mean(self.x_val), np.std(self.x_val))
        # self.std = np.std(list(self.x_train) + list(self.x_val))
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datatset = {"train": self.x_train, "val": self.x_val, "test": self.x_test}
        self.use_cache = use_cache
        if self.use_cache:
            self.cached_datatset = {"train": self.load_data_cache(self.x_train),
                                    "val": self.load_data_cache(self.x_val),
                                    "test": self.load_data_cache(self.x_test)}

    def processes_batch(self, x_batch, mean, std):
        """
        Normalizes a batch images
        :param x_batch: a batch images
        :return: normalized images
        """
        return (x_batch - mean) / std

    def _sample_new_batch(self, data_pack):
        """
        Collect 1000 batches data for N-shot learning
        :param data_pack: one of(train,test,val) dataset shape[classes_num,20,28,28,1]
        :return: A list with [support_set_x,support_set_y,target_x,target_y] ready to be fed to our networks
        """
        support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                  data_pack.shape[3], data_pack.shape[4]), np.float32)

        support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
        target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]), np.float32)
        target_y = np.zeros((self.batch_size, 1), np.int32)

        for i in range(self.batch_size):
            classes_idx = np.arange(data_pack.shape[0])
            samples_idx = np.arange(data_pack.shape[1])
            choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
            choose_label = np.random.choice(self.classes_per_set, size=1)
            choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

            x_temp = data_pack[choose_classes]
            x_temp = x_temp[:, choose_samples]
            y_temp = np.arange(self.classes_per_set)
            support_set_x[i] = x_temp[:, :-1]
            support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
            target_x[i] = x_temp[choose_label, -1]
            target_y[i] = y_temp[choose_label]

        return support_set_x, support_set_y, target_x, target_y

    def _rotate_data(self, image, k):
        """
        Rotates one image by self.k * 90 degrees counter-clockwise
        :param image: Image to rotate
        :return: Rotated Image
        """
        return np.rot90(image, k)

    def _rotate_batch(self, batch_images, k):
        """
        Rotates a whole image batch
        :param batch_images: A batch of images
        :param k: integer degree of rotation counter-clockwise
        :return: The rotated batch of images
        """
        batch_size = batch_images.shape[0]
        for i in np.arange(batch_size):
            batch_images[i] = self._rotate_data(batch_images[i], k)
        return batch_images

    def _get_batch(self, dataset_name, augment=False):
        """
        Get next batch from the dataset with name.
        :param dataset_name: The name of dataset(one of "train","val","test")
        :param augment: if rotate the images
        :return: a batch images
        """
        if self.use_cache:
            support_set_x, support_set_y, target_x, target_y = self._get_batch_from_cache(dataset_name)
        else:
            support_set_x, support_set_y, target_x, target_y = self._sample_new_batch(self.datatset[dataset_name])
        support_set_x = support_set_x.reshape((support_set_x.shape[0], support_set_x.shape[1] * support_set_x.shape[2],
                                               support_set_x.shape[3], support_set_x.shape[4], support_set_x.shape[5]))
        support_set_y = support_set_y.reshape(support_set_y.shape[0], support_set_y.shape[1] * support_set_y.shape[2])
        return support_set_x, support_set_y, target_x, target_y

    def get_train_batch(self, augment=False):
        return self._get_batch("train", augment)

    def get_val_batch(self, augment=False):
        return self._get_batch("val", augment)

    def get_test_batch(self, augment=False):
        return self._get_batch("test", augment)

    def load_data_cache(self, data_pack, argument=True):
        """
        cache the dataset in memory
        :param data_pack: shape[classes_num,20,28,28,1]
        :return:
        """
        cached_dataset = []
        classes_idx = np.arange(data_pack.shape[0])
        samples_idx = np.arange(data_pack.shape[1])
        for _ in range(1000):
            support_set_x = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class, data_pack.shape[2],
                                      data_pack.shape[3], data_pack.shape[4]), np.float32)

            support_set_y = np.zeros((self.batch_size, self.classes_per_set, self.samples_per_class), np.int32)
            target_x = np.zeros((self.batch_size, data_pack.shape[2], data_pack.shape[3], data_pack.shape[4]),
                                np.float32)
            target_y = np.zeros((self.batch_size, 1), np.int32)
            for i in range(self.batch_size):
                choose_classes = np.random.choice(classes_idx, size=self.classes_per_set, replace=False)
                choose_label = np.random.choice(self.classes_per_set, size=1)
                choose_samples = np.random.choice(samples_idx, size=self.samples_per_class + 1, replace=False)

                x_temp = data_pack[choose_classes]
                x_temp = x_temp[:, choose_samples]
                y_temp = np.arange(self.classes_per_set)
                support_set_x[i] = x_temp[:, :-1]
                support_set_y[i] = np.expand_dims(y_temp[:], axis=1)
                target_x[i] = x_temp[choose_label, -1]
                target_y[i] = y_temp[choose_label]
            cached_dataset.append([support_set_x, support_set_y, target_x, target_y])
        return cached_dataset

    def _get_batch_from_cache(self, dataset_name):
        """

        :param dataset_name:
        :return:
        """
        if self.indexes[dataset_name] >= len(self.cached_datatset[dataset_name]):
            self.indexes[dataset_name] = 0
            self.cached_datatset[dataset_name] = self.load_data_cache(self.datatset[dataset_name])
        next_batch = self.cached_datatset[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target
