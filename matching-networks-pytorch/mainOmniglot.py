# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: BoyuanJiang
# College of Information Science & Electronic Engineering,ZheJiang University
# Email: ginger188@gmail.com
# Copyright (c) 2017

# @Time    :17-8-29 22:26
# @FILE    :mainOmniglot.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from data_loader import OmniglotNShotDataset,CUBNShotDataset
from OmniglotBuilder import OmniglotBuilder
import tqdm

# Experiment setup
batch_size = 5
fce = True
classes_per_set = 5
samples_per_class = 1
channels = 1
# Training setup
total_epochs = 100
total_train_batches = 1000
total_val_batches = 250
total_test_batches = 500
best_val_acc = 0.0

# base_file = "/mnt/data/th/BSNet/filelists/CUB/base.json"
# val_file   = "/mnt/data/th/BSNet/filelists/CUB/val.json"
base_file = "/mnt/data/th/BSNet/filelists/UCMerced_LandUse/base.json"
val_file = "/mnt/data/th/BSNet/filelists/UCMerced_LandUse/val.json"
# data = OmniglotNShotDataset(batch_size=batch_size, classes_per_set=classes_per_set,
#                             samples_per_class=samples_per_class, seed=2017, shuffle=True, use_cache=False)
data = CUBNShotDataset(batch_size=batch_size, classes_per_set=classes_per_set,
                            samples_per_class=samples_per_class, seed=2017, shuffle=True, use_cache=False)
data.read_file(base_file,"train")
data.read_file(val_file,"val")

obj_oneShotBuilder = OmniglotBuilder(data)
obj_oneShotBuilder.build_experiment(batch_size=batch_size, num_channels=3, lr=1e-3, image_size=84, classes_per_set=classes_per_set,
                                    samples_per_class=samples_per_class, keep_prob=0.0, fce=False, optim="adam", weight_decay=0,
                                    use_cuda=True)

with tqdm.tqdm(total=total_train_batches) as pbar_e:
    for e in range(total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches)
        print("Epoch {}: train_loss:{:.3f} train_accuracy:{:.3f}".format(e, total_c_loss, total_accuracy))
        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_val_epoch(total_val_batches)
        print("Epoch {}: val_loss:{:.3f} val_accuracy:{:.3f}".format(e, total_val_c_loss, total_val_accuracy))
        if total_val_accuracy>best_val_acc:
            best_val_acc = total_val_accuracy
            # total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_test_epoch(total_test_batches)
            # print("Epoch {}: test_loss:{} test_accuracy:{}".format(e, total_test_c_loss, total_test_accuracy))
        pbar_e.update(1)
print("Best val_accuracy:{:.3f}".format(best_val_acc))