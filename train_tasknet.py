
import argparse
import copy
import numpy as np
import json
import os
import torch
from utils import RunningStatistics, adjust_learning_rate
import losses
from nets import GazeHeadResNet
from dataset import HDFDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from argparse import Namespace
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Train a gaze/headpose estimation model.')
parser.add_argument('--config_json', type=str, help='Path to config in JSON format')
args = parser.parse_args()

#####################################################
# load configurations
assert os.path.isfile(args.config_json)
logging.info('Loading ' + args.config_json)
config = json.load(open(args.config_json))
config = Namespace(**config)
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)

config.lr = config.batch_size*config.base_learning_rate

#####################################################
# save configurations
target_dir = config.save_path + '/configs'
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
fpath = os.path.relpath(target_dir + '/params.json')
with open(fpath, 'w') as f:
    json.dump(vars(config), f, indent=4)
    logging.info('Written %s' % fpath)

#####################################################
# load datasets
with open('./gazecapture_split.json', 'r') as f:
    all_gc_prefixes = json.load(f)


train_prefixes = all_gc_prefixes['train']
train_dataset = HDFDataset(hdf_file_path=config.gazecapture_file,
                                 key_data='',
                                 prefixes=train_prefixes)

val_prefixes = all_gc_prefixes['val']
val_dataset = HDFDataset(hdf_file_path=config.gazecapture_file,
                               key_data='',
                               prefixes=val_prefixes)


test_prefixes = all_gc_prefixes['test']
test_gc_dataset = HDFDataset(hdf_file_path=config.gazecapture_file,
                                key_data='',
                                prefixes=test_prefixes)

mpii_dataset = HDFDataset(hdf_file_path=config.mpiigaze_file, key_data='', prefixes=None)

train_dataloader = DataLoader(train_dataset,
                              batch_size=int(config.batch_size),
                              shuffle=True,
                              drop_last=True,
                              num_workers=config.num_data_loaders,
                              pin_memory=True,
                              )


val_dataloader = DataLoader(val_dataset,
                              batch_size=int(config.batch_size),
                              shuffle=False,
                              drop_last=False,
                              num_workers=config.num_data_loaders,
                              pin_memory=True,
                              )

test_gc_dataloader = DataLoader(test_gc_dataset,
                              batch_size=int(config.batch_size),
                              shuffle=False,
                              drop_last=False,
                              num_workers=config.num_data_loaders,
                              pin_memory=True,
                              )

test_mpii_dataloader = DataLoader(mpii_dataset,
                              batch_size=int(config.batch_size),
                              shuffle=False,
                              drop_last=False,
                              num_workers=config.num_data_loaders,
                              pin_memory=True,
                              )


# logging data stats.
logging.info('')
logging.info("Train datset size: %s" % len(train_dataset))
logging.info("Val dataset size: %s" % (len(val_dataset)))

#####################################################
# create network
network = GazeHeadResNet(norm_layer='instance').to(device)

# Transfer on the GPU before constructing and optimizer
if torch.cuda.device_count() >= 1:
    logging.info('Using %d GPUs!' % torch.cuda.device_count())
    network = nn.DataParallel(network)

net_optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=config.l2_reg)

#####################################################
# single training step

def execute_training_step(current_step):
    global train_data_iterator
    try:
        train_input = next(train_data_iterator)
    except StopIteration:
        torch.cuda.empty_cache()
        global train_dataloader
        train_data_iterator = iter(train_dataloader)
        train_input = next(train_data_iterator)

    network.train()

    gaze_hat, head_hat = network(train_input['_image'].to(device))

    g_loss = losses.gaze_angular_loss(y=train_input['_gaze'].to(device), y_hat=gaze_hat)
    h_loss = losses.gaze_angular_loss(y=train_input['_head'].to(device), y_hat=head_hat)

    loss = g_loss + h_loss
    net_optimizer.zero_grad()
    loss.backward()
    net_optimizer.step()

    running_losses.add('train_gaze_loss', g_loss.detach().cpu().numpy())
    running_losses.add('train_head_loss', h_loss.detach().cpu().numpy())

#####################################################

# single val/test step
def execute_test(test_data):
    test_losses = RunningStatistics()
    with torch.no_grad():
        network.eval()
        for idx, data_dict in enumerate(test_data):
            gaze_hat, head_hat = network(data_dict['_image'].to(device))
            gaze_loss = losses.gaze_angular_loss(data_dict['_gaze'].to(device), gaze_hat)
            head_loss = losses.gaze_angular_loss(data_dict['_head'].to(device), head_hat)

            test_losses.add('val_gaze_loss', gaze_loss.detach().cpu().numpy())
            test_losses.add('val_head_loss', head_loss.detach().cpu().numpy())
    test_loss_means = test_losses.means()
    logging.info('Test Losses at [%7d]: %s' %
                 (current_step, ', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))

    return np.sum([v for k, v in test_loss_means.items()])

#####################################################

logging.info('Training')
running_losses = RunningStatistics()
train_data_iterator = iter(train_dataloader)

val_best_acc = float('inf')

best_model = copy.deepcopy(network)
# main training loop
for current_step in range(0, config.num_training_steps):
    # lr decay
    if (current_step % config.decay_steps == 0):
        lr = adjust_learning_rate([net_optimizer], config.decay, int(current_step /config.decay_steps), config.lr)
    # Testing loop: every specified iterations compute the test statistics
    if current_step % config.print_freq_test == 0 and current_step != 0:
        network.eval()
        torch.cuda.empty_cache()
        # test
        val_loss = execute_test(val_dataloader)
        if val_loss < val_best_acc:
            val_best_acc = val_loss
            torch.save(network.state_dict(), os.path.join(config.save_path, str(current_step) + '.pth.tar'))
            best_model = copy.deepcopy(network)
        torch.cuda.empty_cache()
    # Training step
    execute_training_step(current_step)
    # Print training loss
    if current_step != 0 and (current_step % config.print_freq_train == 0):
        running_loss_means = running_losses.means()
        logging.info('Losses at [%7d]: %s' %
                     (current_step,
                      ', '.join(['%s: %.5f' % v
                                 for v in running_loss_means.items()])))
        running_losses.reset()
logging.info('Finished Training')


###################################
# TESTING
## on MPIIGaze
test_losses = RunningStatistics()
with torch.no_grad():
    best_model.eval()
    for idx, data_dict in enumerate(test_mpii_dataloader):
        gaze_hat, head_hat = best_model(data_dict['_image'].to(device))
        gaze_loss = losses.gaze_angular_loss(data_dict['_gaze'].to(device), gaze_hat)
        head_loss = losses.gaze_angular_loss(data_dict['_head'].to(device), head_hat)

        test_losses.add('val_gaze_loss', gaze_loss.detach().cpu().numpy())
        test_losses.add('val_head_loss', head_loss.detach().cpu().numpy())
test_loss_means = test_losses.means()
logging.info('MPIIGaze Test Losses: %s' % ', '.join(['%s: %.6f' % v for v in test_loss_means.items()]))

## on GC Test
test_losses = RunningStatistics()
with torch.no_grad():
    best_model.eval()
    for idx, data_dict in enumerate(test_gc_dataloader):
        gaze_hat, head_hat = best_model(data_dict['_image'].to(device))
        gaze_loss = losses.gaze_angular_loss(data_dict['_gaze'].to(device), gaze_hat)
        head_loss = losses.gaze_angular_loss(data_dict['_head'].to(device), head_hat)

        test_losses.add('val_gaze_loss', gaze_loss.detach().cpu().numpy())
        test_losses.add('val_head_loss', head_loss.detach().cpu().numpy())
test_loss_means = test_losses.means()
logging.info('GC Test Losses: %s' % ', '.join(['%s: %.6f' % v for v in test_loss_means.items()]))
