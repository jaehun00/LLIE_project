import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import pickle
import argparse
from PIL import Image
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm
from model import *

from multi_read_data2 import MemoryFriendlyLoader
import shutil

parser = argparse.ArgumentParser("SCI")
########## Train
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=1000, help='epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--num_workers', type=int, default=4, help='num workers')

########## Dataset
parser.add_argument('--train_set', type=str, default='./data/our485/low', help='location of the train dataset')
parser.add_argument('--val_set', type=str, default='./data/eval15/low', help='location of the validation dataset')
parser.add_argument('--test_set', type=str, default='./data/darkface', help='location of the test dataset')

########## Save
parser.add_argument('--save', type=str, default='./Final/Train_both', help='location of the data corpus')
parser.add_argument('--img_process', type=int, default=50, help='save image epoch')
parser.add_argument('--input_op', type=bool, default=False, help='save input_op image')
parser.add_argument('--att', type=bool, default=False, help='save att image')
parser.add_argument('--illu', type=bool, default=False, help='save illu image')

args = parser.parse_args()

weight_path = args.save + '/' + 'weights'
os.makedirs(weight_path, exist_ok = True)
model_path = args.save + '/model_epochs'
os.makedirs(model_path, exist_ok = True)
image_path = args.save + '/image_epochs'
os.makedirs(image_path, exist_ok=True)

if args.input_op == True:
    in_image_path = image_path + '/input_op'
    os.makedirs(in_image_path, exist_ok=True)
    
if args.att == True:
    att_image_path = image_path + '/att'
    os.makedirs(att_image_path, exist_ok=True)

if args.illu == True:
    illu_image_path = image_path + '/illu'
    os.makedirs(illu_image_path, exist_ok=True)

train_loss_path = args.save + '/train_losses.pkl'
val_loss_path = args.save + '/val_losses.pkl'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('\nDevice : ', DEVICE)

def load_model(model, latest_epoch):
    print("Load_model")
    model_file = os.path.join(model_path, f'weights_{latest_epoch}.pt')
    if os.path.exists(model_file):
        # load latest model
        base_weights = torch.load(model_file)
        pretrained_dict = base_weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = model.to(DEVICE)

        latest_epoch += 1
        
        return model, latest_epoch
    else:
        print("No model files found.")
        sys.exit(1)
        
def make_model(model):
    print("Make New Model.")
    print("weight_init")
    # init weight
    model.enhance.in_conv.apply(model.weights_init)
    print("init en_in_conv")
    model.enhance.conv.apply(model.weights_init)
    print("init en_conv")
    model.enhance.out_conv.apply(model.weights_init)
    print("init en_out_conv")
    model.enhance.myblock.apply(model.weights_init)
    print("init en_myblock")
    
    model.calibrate.in_conv.apply(model.weights_init)
    print("init cal_in_conv")
    model.calibrate.convs.apply(model.weights_init)
    print("init cal_conv")
    model.calibrate.out_conv.apply(model.weights_init)
    print("init cal_out_conv")
    model.calibrate.myblock.apply(model.weights_init)
    print("init cal_myblock")
    
    model = model.to(DEVICE)  
    print("COMPLETE INIT MODEL")
    return model
 
def save_init():
    init = []
    save_losses_list(init, train_loss_path)
    save_losses_list(init, val_loss_path)

# save list
def save_losses_list(losses_list, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(losses_list, f)

# save loss
def backup_loss(loss, path):
    # backup train_losses pickle file
    loss_backup = []
    loss_backup.append(loss)

    losses_list = load_losses_list(path)
    losses_list.extend(loss_backup)
    
    save_losses_list(losses_list, path)

# load list
def load_losses_list(file_path):
    try:
        with open(file_path, 'rb') as f:
            losses_list = pickle.load(f)
        return losses_list
    except FileNotFoundError:
        return []

########################################################################################
############################            TRAIN            ###############################
########################################################################################             
def train():
    if not torch.cuda.is_available():
        print("no gpu device available")
        sys.exit(1)

    ########## Model Init #########
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    model = Network(stage=args.stage)

    latest_epoch = utils.find_latest_epoch(model_path)
    print(f"Latest epoch: {latest_epoch}")
    if latest_epoch is not None:
        model, latest_epoch = load_model(model, latest_epoch)
    else:
        latest_epoch = 1
        model = make_model(model)
    #############################

    ########## Optimizer init ############
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)

    MB = utils.count_parameters_in_MB(model)
    print('model size = ', MB, 'MB')
    ######################################

    ############ DataLoader ############
    train_low_data_names = args.train_set
    train_high_data_names = "./data/our485/high"
    TrainDataset = MemoryFriendlyLoader(low_img_dir=train_low_data_names, task='train', gt_img_dir=train_high_data_names)

    val_low_data_names = args.val_set
    val_high_data_names = "./data/eval15/high"
    ValDataset = MemoryFriendlyLoader(low_img_dir=val_low_data_names, task='validation', gt_img_dir=val_high_data_names)

    test_low_data_names = args.test_set
    TestDataset = MemoryFriendlyLoader(low_img_dir=test_low_data_names, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers, shuffle=True)

    val_queue = torch.utils.data.DataLoader(
        ValDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers, shuffle=True)

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=args.num_workers, shuffle=True)
    #####################################

    ############ pickle file ############
    if os.path.exists(train_loss_path) and os.path.exists(val_loss_path):
        print(f"{train_loss_path} exists in the directory.")
        print(f"{val_loss_path} exists in the directory.")
    else:
        print(f"{train_loss_path} does not exist in the directory.")
        print(f"{val_loss_path} does not exist in the directory.")
        
        save_init()
        print("new save .pkl")
    #####################################

    ########### Train ###########
    for epoch in range(latest_epoch, args.epochs+1):
        ############################## Train mode ####################################
        model.train()

        train_losses = []
        val_losses = []

        # Train
        for inputs, targets in tqdm(train_queue, desc=f'Training Epoch {epoch}/{args.epochs}', unit='batch'):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            loss = model._loss(inputs, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_losses.append(loss.item())
        
        ############################### Validation mode #################################
        model.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(val_queue, desc=f'Validation Epoch {epoch}/{args.epochs}', unit='batch'):
                inputs = inputs.to(DEVICE)
                targets = inputs.to(DEVICE)

                loss = model._loss(inputs, targets)
                val_losses.append(loss.item())

            if epoch % args.img_process == 0:
                for inputs, labels in  tqdm(test_queue, desc=f'imaging Epoch {epoch}/{args.epochs}', unit='batch'):
                    inputs = inputs.to(DEVICE)
                    image_name = labels
                    image_name = image_name[0].split('/')[-1].split('.')[0]
                    illu_list, ref_list, input_list, att_list= model(inputs)
                    u_name = '%s.png' % (image_name + '_' + str(epoch))
                    u_path = image_path + '/' + u_name
                    utils.save_images(ref_list[0], u_path)

                    for i in range(3):
                        if args.input_op == True:
                            in_name = '%s.png'%(image_name + '_' + str(epoch) + '_' + str(i))
                            in_path = in_image_path + '/' + in_name
                            utils.save_images(input_list[i], in_path)
                        if args.att == True:
                            att_name = '%s.png'%(image_name + '_' + str(epoch) + '_' + str(i) + 'att')
                            att_path = att_image_path + '/' + att_name
                            utils.save_images(att_list[i], att_path)
                        if args.illu == True:
                            illu_name = '%s.png'%(image_name + '_' + str(epoch) + '_' + str(i) + 'illu') 
                            illu_path = illu_image_path + '/' + illu_name
                            utils.save_images(illu_list[i], illu_path)

        ########################### backup ##############################################################
        # backup Model
        torch.save(model.state_dict(), os.path.join(model_path, 'weights_%d.pt' % epoch))

        # backup train_losses pickle file
        train_loss = round(np.average(train_losses), 4)
        backup_loss(train_loss, train_loss_path)

        # backup val_losses pickle file
        val_loss = round(np.average(val_losses), 4)
        backup_loss(val_loss, val_loss_path)
        ##################################################################################################

        val_losses_list = load_losses_list(val_loss_path)
        # save best_model
        if val_loss <= min(val_losses_list):
            is_best1 = True
        else:
            is_best1 = False

        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  Train Loss: {np.average(train_losses):.4f}")
        print(f"  Validation Loss: {np.average(val_losses):.4f}")
        utils.save_best(model, is_best1, weight_path)

if __name__ == '__main__':
    train()