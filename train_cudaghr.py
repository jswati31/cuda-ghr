
import numpy as np
from collections import OrderedDict
import json
import argparse
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import losses
from dataset import HDFDataset
from nets import Controller
import torch.nn.functional as F
from utils import adjust_learning_rate, RunningStatistics, send_data_dict_to_gpu, recover_images, build_image_matrix
from argparse import Namespace
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Train a CUDA-GHR controller.')
parser.add_argument('--config_json', type=str, help='Path to config in JSON format')
parser.add_argument('--columbia', action='store_true', help='train on columbia if true')
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
# load datasets
with open('./gazecapture_split.json', 'r') as f:
    all_gc_prefixes = json.load(f)

train_prefixes = all_gc_prefixes['train']
source_train_dataset = HDFDataset(hdf_file_path=config.gazecapture_file,
                                  key_data='source',
                                  prefixes=train_prefixes)

if args.columbia:
    target_train_prefixes = ['{:04d}'.format(i) for i in range(1, 51)]
    target_train_dataset = HDFDataset(hdf_file_path=config.columbia_file,
                                      key_data='target',
                                      prefixes=target_train_prefixes)

else:
    target_train_prefixes = ['p{:02d}'.format(i) for i in range(11)]
    target_train_dataset = HDFDataset(hdf_file_path=config.mpiigaze_file,
                                      key_data='target',
                                      prefixes=target_train_prefixes)


source_train_dataloader = DataLoader(source_train_dataset,
                                      batch_size=int(config.batch_size),
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.num_data_loaders,
                                      pin_memory=True,
                                      )

target_train_dataloader = DataLoader(target_train_dataset,
                                      batch_size=int(config.batch_size),
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.num_data_loaders,
                                      pin_memory=True,
                                      )


# logging data stats.
logging.info('')
logging.info("Source datset size: %s" % len(source_train_dataset))
logging.info("Target dataset size: %s " % len(target_train_dataset))

#####################################################
# create network

network = Controller(config, device)
network = network.to(device)

if torch.cuda.device_count() >= 1:
    logging.info('Using %d GPUs!' % torch.cuda.device_count())
    network.push_modules_to_multi_gpu()


if config.load_step != 0:
    logging.info('Loading available model')
    network.load_model(os.path.join(config.save_path, "checkpoints", str(config.load_step) + '.pt'))

if 'pretrained' in config:
    logging.info('Loading pretrained model')
    network.load_model(config.pretrained)


gen_params = list(network.image_encoder.parameters()) + \
             list(network.generator.parameters()) + \
             list(network.gaze_latent_encoder.parameters())

gen_optimizer = torch.optim.Adam(gen_params, lr=config.lr, weight_decay=config.l2_reg)

disc_params = list(network.target_discrim.parameters()) + \
              list(network.source_discrim.parameters()) + \
              list(network.latent_discriminator.parameters())

disc_optimizer = torch.optim.Adam(disc_params, lr=config.lr, weight_decay=config.l2_reg)

optimizers = [gen_optimizer, disc_optimizer]

perceptual_loss = losses.PerceptualLoss(device=device)

latent_disc_loss = nn.BCEWithLogitsLoss(reduction='mean')

#####################################################

# single training step
def execute_training_step(current_step):

    network.train_mode_on()

    global source_train_data_iterator
    global target_train_data_iterator
    try:
        source_input = next(source_train_data_iterator)
        target_input = next(target_train_data_iterator)
    except StopIteration:
        torch.cuda.empty_cache()
        global source_train_dataloader
        global target_train_dataloader
        source_train_data_iterator = iter(source_train_dataloader)
        target_train_data_iterator = iter(target_train_dataloader)
        source_input = next(source_train_data_iterator)
        target_input = next(target_train_data_iterator)

    input_dict = send_data_dict_to_gpu(source_input, target_input, device)

    ############### DISCRIMINATOR ############

    for param in network.target_discrim.parameters():
        param.requires_grad = True
    for param in network.source_discrim.parameters():
        param.requires_grad = True
    for param in network.latent_discriminator.parameters():
        param.requires_grad = True
    for param in network.image_encoder.parameters():
        param.requires_grad = False
    for param in network.generator.parameters():
        param.requires_grad = False
    for param in network.gaze_latent_encoder.parameters():
        param.requires_grad = False

    ## source data
    s_app_embedding = network.image_encoder(input_dict['source_image'])
    s_gaze_embedding = network.gaze_latent_encoder(input_dict['source_gaze'])
    source_emdedding = torch.cat((s_app_embedding, s_gaze_embedding), dim=-1)
    source_gen_image = network.generator([source_emdedding, F.pad(input=input_dict['source_head'], pad=(0, 1),
                                                                  mode='constant', value=0)])[0]

    real = network.source_discrim(input_dict['source_image'])
    fake = network.source_discrim(source_gen_image.detach())
    disc_loss_D = losses.discriminator_loss(real=real, fake=fake)

    ## target data
    gaze_pred_, head_pred_ = network.task_net(input_dict['target_image'])
    t_app_embedding = network.image_encoder(input_dict['target_image'])
    t_gaze_embedding = network.gaze_latent_encoder(gaze_pred_)
    target_embedding = torch.cat((t_app_embedding, t_gaze_embedding), dim=-1)
    target_gen_image = network.generator([target_embedding, F.pad(input=head_pred_, pad=(0, 1),
                                                           mode='constant', value=0)])[0]

    real = network.target_discrim(input_dict['target_image'])
    fake = network.target_discrim(target_gen_image.detach())
    disc_loss_D += losses.discriminator_loss(real=real, fake=fake)

    ## latents
    real = network.latent_discriminator(target_embedding)
    fake = network.latent_discriminator(source_emdedding)
    disc_loss_D += losses.discriminator_loss(real=real, fake=fake)

    disc_optimizer.zero_grad()
    disc_loss_D.backward()
    disc_optimizer.step()

    ############### GENERATOR ############

    for param in network.target_discrim.parameters():
        param.requires_grad = False
    for param in network.source_discrim.parameters():
        param.requires_grad = False
    for param in network.latent_discriminator.parameters():
        param.requires_grad = False
    for param in network.image_encoder.parameters():
        param.requires_grad = True
    for param in network.generator.parameters():
        param.requires_grad = True
    for param in network.gaze_latent_encoder.parameters():
        param.requires_grad = True

    # source data
    s_app_embedding = network.image_encoder(input_dict['source_image'])
    s_gaze_embedding = network.gaze_latent_encoder(input_dict['source_gaze'])
    source_emdedding = torch.cat((s_app_embedding, s_gaze_embedding), dim=-1)
    source_gen_image = network.generator([source_emdedding, F.pad(input=input_dict['source_head'], pad=(0, 1),
                                                                  mode='constant', value=0)])[0]
    # target data
    gaze_pred_orig_t, head_pred_orig_t = network.task_net(input_dict['target_image'])
    t_app_embedding = network.image_encoder(input_dict['target_image'])
    t_gaze_embedding = network.gaze_latent_encoder(gaze_pred_orig_t)
    target_embedding = torch.cat((t_app_embedding, t_gaze_embedding), dim=-1)
    target_gen_image = network.generator([target_embedding, F.pad(input=head_pred_orig_t, pad=(0, 1),
                                                           mode='constant', value=0)])[0]

    ## reconstruction loss and perceptual loss
    rloss = torch.nn.L1Loss()(source_gen_image, input_dict['source_image']) + \
            torch.nn.L1Loss()(target_gen_image, input_dict['target_image'])

    percep_loss = perceptual_loss.loss(source_gen_image, input_dict['source_image']) \
                  + perceptual_loss.loss(target_gen_image, input_dict['target_image'])

    ## GAN loss
    fake = network.source_discrim(source_gen_image)
    disc_loss_G = losses.generator_loss(fake=fake)

    fake = network.target_discrim(target_gen_image)
    disc_loss_G += losses.generator_loss(fake=fake)

    # L_feat
    fake = network.latent_discriminator(target_embedding)
    real = network.latent_discriminator(source_emdedding)
    real_size = list(real.size())
    fake_size = list(fake.size())
    real_label = torch.zeros(real_size, dtype=torch.float32).to(device)
    fake_label = torch.ones(fake_size, dtype=torch.float32).to(device)
    disc_loss_G += (latent_disc_loss(fake, fake_label) + latent_disc_loss(real, real_label)) / 2

    ## label consistency loss
    gaze_pred_orig_s, head_pred_orig_s = network.task_net(input_dict['source_image'])
    gaze_pred, head_pred = network.task_net(source_gen_image)
    task_loss = losses.gaze_angular_loss(y=gaze_pred_orig_s, y_hat=gaze_pred)
    task_loss += losses.gaze_angular_loss(y=head_pred_orig_s, y_hat=head_pred)

    gaze_pred, head_pred = network.task_net(target_gen_image)
    task_loss += losses.gaze_angular_loss(y=gaze_pred_orig_t, y_hat=gaze_pred)
    task_loss += losses.gaze_angular_loss(y=head_pred_orig_t, y_hat=head_pred)

    ## redirected consistency loss
    gaze_swapped_image = network.generator([torch.cat((t_app_embedding, s_gaze_embedding), dim=-1),
                                            F.pad(input=head_pred_orig_t, pad=(0, 1),
                                                  mode='constant', value=0)])[0]
    head_swapped_image = network.generator([torch.cat((t_app_embedding, t_gaze_embedding), dim=-1),
                                            F.pad(input=input_dict['source_head'], pad=(0, 1),
                                                  mode='constant', value=0)])[0]

    gaze_swapped_gaze_pred, gaze_swapped_head_pred = network.task_net(gaze_swapped_image)
    head_swapped_gaze_pred, head_swapped_head_pred = network.task_net(head_swapped_image)
    task_loss += losses.gaze_angular_loss(y=gaze_pred_orig_s, y_hat=gaze_swapped_gaze_pred)
    task_loss += losses.gaze_angular_loss(y=head_pred_orig_t, y_hat=gaze_swapped_head_pred)
    task_loss += losses.gaze_angular_loss(y=gaze_pred_orig_t, y_hat=head_swapped_gaze_pred)
    task_loss += losses.gaze_angular_loss(y=head_pred_orig_s, y_hat=head_swapped_head_pred)

    loss = config.coeff_l1_loss*rloss + \
           config.coeff_latent_discriminator_loss*disc_loss_G + \
           config.coeff_perc_loss*percep_loss + \
           config.coeff_gaze_loss*task_loss

    gen_optimizer.zero_grad()
    loss.backward()
    gen_optimizer.step()

    # save training samples in tensorboard
    if config.use_tensorboard and current_step % config.save_freq_images == 0 and current_step != 0:
        image_index = 0
        tensorboard.add_image('train/source_input_image',
                              torch.clamp((input_dict['source_image'][image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/target_input_image',
                              torch.clamp((input_dict['target_image'][image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/source_generated_image',
                              torch.clamp((source_gen_image[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/target_generated_image',
                              torch.clamp((target_gen_image[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/target_generated_gaze_swap_image',
                              torch.clamp((gaze_swapped_image[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)
        tensorboard.add_image('train/target_generated_head_swap_image',
                              torch.clamp((head_swapped_image[image_index] + 1) * (255.0 / 2.0), 0, 255).type(
                                  torch.cuda.ByteTensor), current_step)

    return rloss.item(), disc_loss_D.item(), disc_loss_G.item(), percep_loss.item(), task_loss.item()

#####################################################

# single test/visualize step
def execute_visualize(data, current_step):
    test_losses = RunningStatistics()
    output_dict = OrderedDict()
    fid_dict = OrderedDict()
    with torch.no_grad():
        network.eval_mode_on()
        # source image
        s_app_embedding = network.image_encoder(data['source_image'])
        s_gaze_embedding = network.gaze_latent_encoder(data['source_gaze'])
        source_emdedding = torch.cat((s_app_embedding, s_gaze_embedding), dim=-1)
        source_gen_image = network.generator([source_emdedding, F.pad(input=data['source_head'], pad=(0, 1),
                                                                      mode='constant', value=0)])[0]
        # target data
        gaze_head_v, head_pred_v = network.task_net(data['target_image'])
        t_app_embedding = network.image_encoder(data['target_image'])
        t_gaze_embedding = network.gaze_latent_encoder(gaze_head_v)
        target_embedding = torch.cat((t_app_embedding, t_gaze_embedding), dim=-1)
        target_gen_image = network.generator([target_embedding, F.pad(input=head_pred_v, pad=(0, 1),
                                                                      mode='constant', value=0)])[0]

        rloss = torch.nn.L1Loss()(source_gen_image, data['source_image']) + \
                torch.nn.L1Loss()(target_gen_image, data['target_image'])
        test_losses.add('l1_loss', rloss.detach().cpu().numpy())

        gaze_swapped_image = network.generator([torch.cat((t_app_embedding, s_gaze_embedding), dim=-1),
                                                F.pad(input=head_pred_v, pad=(0, 1),
                                                      mode='constant', value=0)])[0]
        head_swapped_image = network.generator([torch.cat((t_app_embedding, t_gaze_embedding), dim=-1),
                                                F.pad(input=data['source_head'], pad=(0, 1),
                                                      mode='constant', value=0)])[0]

        gaze_to_head_g_pred, gaze_to_head_h_pred = network.task_net(gaze_swapped_image)
        head_to_gaze_g_pred, head_to_gaze_h_pred = network.task_net(head_swapped_image)

        gaze_to_head_loss = losses.gaze_angular_loss(y=head_pred_v, y_hat=gaze_to_head_h_pred)
        gaze_redir_loss = losses.gaze_angular_loss(y=data['source_gaze'], y_hat=gaze_to_head_g_pred)
        head_to_gaze_loss = losses.gaze_angular_loss(y=gaze_head_v, y_hat=head_to_gaze_g_pred)
        head_redir_loss = losses.gaze_angular_loss(y=data['source_head'], y_hat=head_to_gaze_h_pred)

        test_losses.add('gaze_to_head_loss', gaze_to_head_loss.detach().cpu().numpy())
        test_losses.add('gaze_redir_loss', gaze_redir_loss.detach().cpu().numpy())
        test_losses.add('head_to_gaze_loss', head_to_gaze_loss.detach().cpu().numpy())
        test_losses.add('head_redir_loss', head_redir_loss.detach().cpu().numpy())

        output_dict['source_image'] = data['source_image']
        output_dict['target_image'] = data['target_image']
        output_dict['gaze_swap_target_image_hat'] = gaze_swapped_image
        output_dict['head_swap_target_image_hat'] = head_swapped_image
        output_dict['target_image_hat'] = target_gen_image
        output_dict['source_image_hat'] = source_gen_image

    test_loss_means = test_losses.means()
    logging.info('Test Losses at [%7d]: %s' %
                 (current_step, ', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))
    if config.use_tensorboard:
        for k, v in test_loss_means.items():
            tensorboard.add_scalar('test/%s' % (k), v, current_step)

    path = os.path.join(config.save_path, 'samples')
    if not os.path.exists(path):
        os.makedirs(path)

    col1 = recover_images(output_dict['source_image'])
    col2 = recover_images(output_dict['source_image_hat'])
    col3 = recover_images(output_dict['target_image'])
    col4 = recover_images(output_dict['target_image_hat'])
    col5 = recover_images(output_dict['gaze_swap_target_image_hat'])
    col6 = recover_images(output_dict['head_swap_target_image_hat'])

    images = np.vstack((col1[:4], col2[:4], col3[:4], col4[:4], col5[:4], col6[:4]))

    gen_img = build_image_matrix(images, 6, min(config.batch_size, 4))

    cv2.imwrite(os.path.join(path, 'generated_' + str(current_step) + '.png'), gen_img)


#####################################################
# initializing tensorboard

if config.use_tensorboard:
    from tensorboardX import SummaryWriter
    tensorboard = SummaryWriter(logdir=config.save_path)

#####################################################

logging.info('Training')
running_losses = RunningStatistics()
source_train_data_iterator = iter(source_train_dataloader)
target_train_data_iterator = iter(target_train_dataloader)

# fixing test samples
target_test_data_dict = next(target_train_data_iterator)
source_test_data_dict = next(source_train_data_iterator)

test_data_dict = send_data_dict_to_gpu(source_test_data_dict, target_test_data_dict, device)

if config.train_mode:

    #####################################################
    # save configurations only if training
    target_dir = config.save_path + '/configs'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    fpath = os.path.relpath(target_dir + '/params.json')
    with open(fpath, 'w') as f:
        json.dump(vars(config), f, indent=4)
        logging.info('Written %s' % fpath)

    #####################################################

    # main training loop
    for current_step in range(config.load_step, config.num_training_steps):
        # Save model
        if current_step % config.save_interval == 0 and current_step != config.load_step:
            network.save_model(current_step)

        # lr decay
        if (current_step % config.decay_steps == 0) or current_step == config.load_step:
            lr = adjust_learning_rate(optimizers, config.decay, int(current_step / config.decay_steps), config.lr)
            if config.use_tensorboard:
                tensorboard.add_scalar('train/lr', lr, current_step)

        # Testing loop: every specified iterations compute the test statistics
        if current_step % config.print_freq_test == 0:
            torch.cuda.empty_cache()
            # test
            execute_visualize(test_data_dict, current_step)
            torch.cuda.empty_cache()

        # Training step
        tr_loss = execute_training_step(current_step)
        # Print training loss
        if current_step != 0 and (current_step % config.print_freq_train == 0):
            logging.info('Losses at [%7d]: %s %s %s %s %s' %
                         (current_step, tr_loss[0], tr_loss[1], tr_loss[2], tr_loss[3], tr_loss[4]))
            if config.use_tensorboard:
                tensorboard.add_scalar('train/l1_loss', tr_loss[0], current_step)
                tensorboard.add_scalar('train/discD', tr_loss[1], current_step)
                tensorboard.add_scalar('train/discG', tr_loss[2], current_step)
                tensorboard.add_scalar('train/perc_loss', tr_loss[3], current_step)
                tensorboard.add_scalar('train/task_loss', tr_loss[4], current_step)

    logging.info('Finished Training')

    # save final model
    network.save_model(config.num_training_steps)

#####################################################

# generate dataset for test data for training task net
if config.store_task_evaluation_dataset:
    import h5py
    network.load_model(os.path.join(config.save_path, "checkpoints", str(config.num_training_steps) + '.pt'))
    network.eval_mode_on()

    # generate dataset for target data for training task net
    current_person_id = None
    all_person_data = {}
    current_person_data = {}
    ofpath = os.path.join(config.save_path, 'Redirected_samples_{}_each.h5'.format(config.num_samples_generation))
    ofdir = os.path.dirname(ofpath)
    if not os.path.isdir(ofdir):
        os.makedirs(ofdir)

    h5f = h5py.File(ofpath, 'a')

    def store_person_predictions():
        global current_person_data
        if len(current_person_data) > 0:
            if current_person_id not in h5f:
                g = h5f.create_group(current_person_id)
                for key, data in current_person_data.items():
                    g.create_dataset(key, data=data, chunks=True, compression='lzf', dtype=np.float32,
                                     maxshape=tuple([None] + list(np.asarray(data).shape[1:])))
            else:
                for key, data in current_person_data.items():
                    data = np.array(data).astype(np.float32)
                    h5f[current_person_id][key].resize((h5f[current_person_id][key].shape[0] + data.shape[0]), axis=0)
                    h5f[current_person_id][key][-data.shape[0]:] = data

        current_person_data = {}


    if args.columbia:
        target_train_prefixes = ['{:04d}'.format(i) for i in range(1, 57)]
    else:
        target_train_prefixes = ['p{:02d}'.format(i) for i in range(15)]

    for current_person_id in target_train_prefixes:
        # load dataset for which to create data samples
        if args.columbia:
            target_dataset = HDFDataset(hdf_file_path=config.columbia_file, key_data='target',
                                              prefixes=[current_person_id])

        else:
            target_dataset = HDFDataset(hdf_file_path=config.mpiigaze_file, key_data='target',
                                              prefixes=[current_person_id])

        target_dataloader = DataLoader(target_dataset,
                                      batch_size=int(config.batch_size),
                                      shuffle=False,
                                      drop_last=True,
                                      num_workers=config.num_data_loaders,
                                      pin_memory=True
                                          )

        source_data_iterator = iter(source_train_dataloader)
        target_data_iterator = iter(target_dataloader)

        num_iter = int(config.num_samples_generation // config.batch_size)

        for i in range(num_iter):
            torch.cuda.empty_cache()

            try:
                source_input = next(source_data_iterator)
                target_input = next(target_data_iterator)
            except StopIteration:
                source_data_iterator = iter(source_train_dataloader)
                target_data_iterator = iter(target_dataloader)
                source_input = next(source_data_iterator)
                target_input = next(target_data_iterator)

            input_dict = send_data_dict_to_gpu(source_input, target_input, device)

            with torch.no_grad():
                network.eval_mode_on()

                # source image
                s_app_embedding = network.image_encoder(input_dict['source_image'])
                s_gaze_embedding = network.gaze_latent_encoder(input_dict['source_gaze'])
                source_emdedding = torch.cat((s_app_embedding, s_gaze_embedding), dim=-1)
                source_gen_image = network.generator([source_emdedding, F.pad(input=input_dict['source_head'],
                                                                              pad=(0, 1),
                                                                              mode='constant', value=0)])[0]
                # target data
                gaze_head_v, head_pred_v = network.task_net(input_dict['target_image'])
                t_app_embedding = network.image_encoder(input_dict['target_image'])
                t_gaze_embedding = network.gaze_latent_encoder(gaze_head_v)
                target_embedding = torch.cat((t_app_embedding, t_gaze_embedding), dim=-1)
                target_gen_image = network.generator([target_embedding, F.pad(input=head_pred_v, pad=(0, 1),
                                                                              mode='constant', value=0)])[0]

                gaze_swapped_image = network.generator([torch.cat((t_app_embedding, s_gaze_embedding), dim=-1),
                                                        F.pad(input=head_pred_v, pad=(0, 1), mode='constant',
                                                              value=0)])[0]
                head_swapped_image = network.generator([torch.cat((t_app_embedding, t_gaze_embedding), dim=-1),
                                                        F.pad(input=input_dict['source_head'], pad=(0, 1),
                                                              mode='constant',
                                                              value=0)])[0]

                zipped_data = zip(
                    target_input['target_key'],
                    gaze_swapped_image.cpu().numpy().astype(np.float32),
                    head_swapped_image.cpu().numpy().astype(np.float32),
                    input_dict['source_gaze'].cpu().numpy().astype(np.float32),
                    input_dict['source_head'].cpu().numpy().astype(np.float32),
                )

                for (person_id, image_gaze_sw, image_head_sw, gaze_b_r, head_b_r) in zipped_data:
                    # Store predictions if moved on to next person
                    if person_id != current_person_id:
                        store_person_predictions()
                        current_person_id = person_id
                    # Now write it
                    to_write = {
                        'gaze_sw_image': image_gaze_sw,
                        'head_sw_image': image_head_sw,
                        'source_gaze': gaze_b_r,
                        'source_head': head_b_r,
                    }
                    for k, v in to_write.items():
                        if k not in current_person_data:
                            current_person_data[k] = []
                        current_person_data[k].append(v)

                logging.info('processed batch [%s/%04d/%04d].' % (current_person_id, i + 1, num_iter))
        store_person_predictions()
    logging.info('Completed processing')
    logging.info('Done')
