import time
import torch
import os
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
import argparse
#############
# utilities #
#############
from utils.metrics import get_accuracy
from utils.read_mnist_data import read_mnist_data
###########
# dataset #
###########
from dataset.mnist_dataset import MnistDataset
##########
# models #
##########
from models.mlp_classifier import MLPClassifier
from models.lenet5_classifier import LeNet5Classifier
from models.can_classifier import CANClassifier

def train_one_epoch(model, optimizer, loss_func, train_loader, mode, device):
    model.train()
    losses = []
    accs = []
    for i, sample in enumerate(train_loader):
        sample = [s.to(device) for s in sample]
        imgs, labels = sample
        
        if mode == MODE_MLP:
            B, H, W = imgs.size()
            imgs = imgs.reshape((B, H*W))
        else:
            imgs = torch.unsqueeze(imgs, dim=1) # to add a dummy channel dimension
        
        out = model(imgs)
        loss = loss_func(out, labels)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred_labels = torch.argmax(out, dim=1)
        accs.append(get_accuracy(labels, pred_labels))
        
        losses.append(loss.item())
            
    return np.mean(losses), np.mean(accs)

def validation(model, loss_func, test_loader, mode, device):
    model.eval()
    test_losses = []
    test_accs = []
    
    for i, sample in enumerate(test_loader):
        sample = [s.to(device) for s in sample]
        
        imgs, labels = sample
        
        if mode == MODE_MLP:
            B, H, W = imgs.size()
            imgs = imgs.reshape((B, H*W))
        else:
            imgs = torch.unsqueeze(imgs, dim=1)
        
        out = model(imgs)
        pred_labels = torch.argmax(out, dim=1)
        test_accs.append(get_accuracy(labels, pred_labels))
        
        loss = loss_func(out, labels)
        test_losses.append(loss.item())
    
    return np.mean(test_losses), np.mean(test_accs)
        
MODE_KNN = "KNN"
MODE_CNN = "CNN"
MODE_MLP = "MLP"
MODE_CAN = "CAN"

parser = argparse.ArgumentParser(prog = 'ProgramName')
parser.add_argument("--mode", type=str, default="MLP")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--num_neurons", type=int, default=16)
parser.add_argument("--feat_dim", type=int, default=16)
parser.add_argument("--exp_name", type=str, default="default")
parser.add_argument("--num_neighbors", type=int, default=1)
parser.add_argument("--act", type=str, default="tanh")
parser.add_argument("--batch_size", type=int, default=16)

def main():
    ###############
    # user params #
    ###############
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mode = args.mode
    epoch = args.epoch
    lr = args.lr
    num_neurons = args.num_neurons
    feat_dim = args.feat_dim
    exp_name = args.exp_name
    act = args.act
    num_neighbors = args.num_neighbors
    bs = args.batch_size
    print("\tUSER PARAMETERS: \n \
            \tDEVICE {} \n \
            \tMODE {} \n \
            \tEPOCH {} \n \
            \tLEARNING RATE {:.7f} \n \
            \tEXPERIMENT NAME {} \n \
            \tBATCH SIZE {} \
            ".format(device, mode, epoch, lr, exp_name, bs))
    if mode == MODE_MLP:
        print("\t\tNUM_NEURONS {}".format(num_neurons))
    elif mode == MODE_CNN:
        print("\t\tACTIVATION {}".format(act))
    elif mode == MODE_CAN:
        print("\t\tFEATURE DIMENSION {}".format(feat_dim))
    elif mode == MODE_KNN:
        print("\t\tNUM NEIGHBORS {}".format(num_neighbors))
    
    ################
    # prepare data #
    ################
    data_path = "./data"
    train_images_path = "train-images.idx3-ubyte"
    train_labels_path = "train-labels.idx1-ubyte"
    test_images_path = "t10k-images.idx3-ubyte"
    test_labels_path = "t10k-labels.idx1-ubyte"
    train_set = MnistDataset(os.path.join(data_path, train_images_path), os.path.join(data_path, train_labels_path))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_set = MnistDataset(os.path.join(data_path, test_images_path), os.path.join(data_path, test_labels_path))
    test_loader = DataLoader(test_set, batch_size=50, shuffle=True)
    
    #####################################################
    # locate proper experiment results & logs directory #
    #####################################################
    exp_res_path = "./exp_results/" + exp_name
    new_exp_res_path = exp_res_path
    cnt = 0
    while True:
        cnt += 1
        if os.path.exists(new_exp_res_path):
            if os.path.exists(os.path.join(new_exp_res_path, "result.txt")):
                new_exp_res_path = exp_res_path + "_iter" + str(cnt)
            else:
                break
        else:
            break
    
    exp_res_path = new_exp_res_path
    log_path = os.path.join(exp_res_path, "logs")
    
    os.makedirs(exp_res_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    print("\tSTORING EXPERIMENT RESULTS TO {}".format(exp_res_path))
    print("\tSTORING EXPERIMENT LOGS TO {}".format(log_path))
    
    writer = SummaryWriter(log_path)
    
    ################
    # create model #
    ################
    model = None
    if mode == MODE_MLP:
        model = MLPClassifier(num_neurons, 10)
    elif mode == MODE_CNN:
        model = LeNet5Classifier(act)
    elif mode == MODE_CAN:
        model = CANClassifier(feat_dim)
    elif mode == MODE_KNN:
        model = KNeighborsClassifier(n_neighbors=num_neighbors, metric="l1")
        
    if model is None:
        raise Exception("Unrecognized training mode, received {}".format(mode))
    
    if mode != MODE_KNN:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.to(device)
        loss_func = torch.nn.CrossEntropyLoss()
        model_state_dict = {"epoch":0, "model_state_dict":model.state_dict(), "loss":0}
    
    #################
    # training loop #
    #################
    best_val = -100000
    if mode == MODE_KNN:
        train_imgs, train_labels = read_mnist_data(os.path.join(data_path, train_images_path), os.path.join(data_path, train_labels_path))
        test_imgs, test_labels = read_mnist_data(os.path.join(data_path, test_images_path), os.path.join(data_path, test_labels_path))
        
        def flatten_all_imgs(imgs):
            flattened = []
            for i, im in enumerate(imgs):
                flattened.append(im.flatten())
            return np.array(flattened)

        train_imgs = flatten_all_imgs(train_imgs)
        test_imgs = flatten_all_imgs(test_imgs)
        
        model.fit(train_imgs, train_labels)
        
        start = time.time()
        pred_labels = model.predict(test_imgs)
        dt = time.time()-start
        print("\tPREDICTION TOOK {} SECONDS".format(dt))
        acc = np.sum(pred_labels==test_labels)
        best_val = acc / len(test_imgs)
        print("\tTEST ACCURACY: {}".format(best_val))
    else:
        for e in range(epoch):
            train_loss, train_acc = train_one_epoch(model, optimizer, loss_func, train_loader, mode, device)
            print("\tEpoch {}, Train Loss {:.5f}, Training Accuracy {:.5f}".format(e, train_loss, train_acc))
            writer.add_scalar("Loss/Train", train_loss, e)
            writer.add_scalar("Accuracy/Train", train_acc, e)
            
            if e % 5 == 0:
                validation_loss, validation_acc = validation(model, loss_func, test_loader, mode, device)
                print("\tEpoch {}, Validation Loss {:.5f}, Validation Accuracy {:.5f}".format(e, validation_loss, validation_acc))
                writer.add_scalar("Loss/Validation", validation_loss, e)
                writer.add_scalar("Accuracy/Validation", validation_acc, e)
                
                if validation_acc > best_val and e > epoch * 0.1:
                    best_val = validation_acc
                    model_state_dict["epoch"] = e
                    model_state_dict["loss"] = best_val
                    model_state_dict["model_state_dict"] = model.state_dict()
                    torch.save(model_state_dict, os.path.join(exp_res_path, "best.pt"))
                
    ###################################
    # write experiment results to txt #
    ###################################
    res_file = os.path.join(exp_res_path, "result.txt")
    
    with open(res_file, "w+") as f:
        if mode != MODE_KNN:
            f.write("lr: {}\nepoch: {}\nbest_val: {}\nbatch_size: {}\n".format(lr, epoch, best_val, bs))
        
        if mode == MODE_MLP:
            f.write("num_neurons: {}\n".format(num_neurons))    
        elif mode == MODE_CNN:
            f.write("act: {}\n".format(act)) 
        elif mode == MODE_CAN:
            f.write("feat_dim: {}\n".format(feat_dim))
        elif mode == MODE_KNN:
            f.write("best_val: {}\nnum_neighbors: {}\ninference_time: {}".format(best_val, num_neighbors, dt))
            
        f.close()
        
if __name__ == "__main__":
    main()