import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import random
import torch.optim as optim
import torch.nn as nn
import copy
from sklearn import preprocessing
import torchvision
from torchvision import transforms
import itertools
from utils import *
from options import args_parser
import os

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, dim_out)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        x = self.activation(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        x = self.activation(x)
        return x
    
class CNNFashion(nn.Module):
    def __init__(self, num_classes):
        super(CNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) #in-ch, out-ch, kernel_size, 28x28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3) 
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.output = nn.Linear(64, num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #28x28 -> 26x26 -> 13x13
        x = self.pool(F.relu(self.conv2(x))) #13x13 -> 11x11 -> 5x5
        #x = self.pool(F.relu(self.conv3(x))) #5x5 -> 3x3 -> 1x1
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        x = self.activation(x)
        return x
    
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, transform):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform:
            image = self.transform(image)
        return image, label


class ClientUpdate(object):
    def __init__(self, train_set, idxs_train, idxs_val, criterion, lr, device, batch_size, rot_deg, num_users, dataset):
        self.device = device
        self.criterion = criterion
        self.lr = lr
        rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))
        self.train_set = DatasetSplit(train_set,idxs_train,rot_transform)
        if(idxs_val):
            self.val_set = DatasetSplit(train_set,idxs_val,rot_transform)
            self.ldr_val = DataLoader(self.val_set, batch_size = 1, shuffle=False)
        
        self.ldr_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        
        if(args.dataset=='cifar10'):
            self.local_model = CNN(num_classes=10).to(self.device)
        elif(args.dataset=='fashion-mnist'):
            self.local_model = CNNFashion(num_classes=10).to(self.device)
            
        self.best_model = copy.deepcopy(self.local_model)
        
        self.received_models = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.n_received = 0
        self.n_sampled = np.zeros(num_users)
        self.n_sampled_prev = np.zeros(num_users)
        self.n_selected = np.zeros(num_users)
        self.best_val_loss = np.inf
        self.best_val_acc = -np.inf
        self.count = 0
        self.stopped_early = False

        self.priors = np.zeros(num_users)
        self.priors_minmax = np.zeros(num_users)
        self.priors_norm = np.zeros(num_users)
        self.similarity_scores = np.zeros(num_users)
        self.neighbour_list = []
        
    def train(self,n_epochs):
        self.local_model.train()
        optimizer = torch.optim.Adam(self.local_model.parameters(),lr=self.lr)
        
        epoch_loss = []
        
        for iter in range(n_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = self.local_model(images.float())
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        self.train_loss_list.append(epoch_loss[-1])
        val_loss, val_acc = self.validate(self.local_model, train_set = False)
        self.val_loss_list.append(val_loss)
        self.val_acc_list.append(val_acc)
        
        if(val_loss < self.best_val_loss):
            self.count = 0
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_model.load_state_dict(self.local_model.state_dict())
        else:
            self.count += 1
            
        return self.best_model, epoch_loss[-1], self.best_val_loss, self.best_val_acc
    
    def validate(self,model,train_set):
        if(train_set):
            ldr = self.ldr_train
        else:
            ldr = self.ldr_val
            
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(ldr):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                log_probs = model(inputs)
                _, predicted = torch.max(log_probs.data, 1)
                                         
                loss = self.criterion(log_probs,labels)                
                batch_loss.append(loss.item())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            val_loss = sum(batch_loss)/len(batch_loss)

        return val_loss, val_acc
    
    
def create_test_data(cluster_id,rot_deg,dataset):
    if(dataset=='fashion-mnist'):
        rot_transform = transforms.RandomRotation(degrees=(rot_deg[cluster_id],rot_deg[cluster_id]))
        trans_fashion_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),rot_transform])
    
        test_dataset = torchvision.datasets.FashionMNIST('.', train=False, download=True, transform=trans_fashion_test)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
    elif(dataset=='cifar10'):
        rot_transform = transforms.RandomRotation(degrees=(rot_deg[cluster_id],rot_deg[cluster_id]))
        trans_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),rot_transform])

        test_dataset = torchvision.datasets.CIFAR10('.', train=False, download=True, transform=trans_cifar_test)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
    return test_loader


def test(model,criterion,test_loader,device):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(model(vec).shape)
            log_probs = model(inputs).view(1,10)

            _, predicted = torch.max(log_probs.data, 1)
            test_loss += criterion(log_probs, labels.long()).item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = 100 * correct / total
    return test_loss, test_acc

def test_labelshift(model,criterion,test_loader,device,group_labels):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if(labels.item() in group_labels):
                log_probs = model(inputs).view(1,10)

                _, predicted = torch.max(log_probs.data, 1)
                test_loss += criterion(log_probs, labels.long()).item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
    test_loss /= len(test_loader.dataset)
    test_acc = 100 * correct / total
    return test_loss, test_acc

if __name__ == '__main__':   
    args = args_parser()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    filename = 'results'
    filexist = os.path.isfile('./save/'+filename) 
    if(not filexist):
        with open('./save/'+filename,'a') as f1:

            f1.write('n_rounds;n_rouns_pens;num_clients;local_ep;bs;lr;n_clusters;pens;DAC;DAC_var;oracle;random;top_m;n_sampled;n_data_train;n_data_val;tau;test_acc0;test_acc1;test_acc2;test_acc3;dataset')

            f1.write('\n')
            
    cuda_no = "0"
    device = torch.device("cuda:"+cuda_no)
    criterion = nn.NLLLoss()
    
    num_clients = args.num_clients
    
    if(args.dataset=='fashion-mnist'):
        trans_fashion = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST('.', train=True, download=True, transform=trans_fashion)
    elif(args.dataset=='cifar10'):
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10('.', train=True, download=True, transform=trans_cifar)
    
    #assign data samples to clients
    
    if(args.iid == 'label'):
        if(args.dataset == 'fashion-mnist'):
            print("Error: Dataset Fahsion-MNIST not yet implemented for label shift. Try covariate-shift or change to CIFAR-10")
            exit()
        dict_users, dict_users_val = sample_cifargroups(train_dataset, num_clients, args.n_data_train, args.n_data_val) 
        
    elif(args.iid == 'covariate'):
        dict_users, dict_users_val = sample_labels_iid(train_dataset, num_clients, args.n_data_train, args.n_data_val)
        cluster_idx = np.zeros(num_clients,dtype='int')
        cluster_list = []
        if(args.n_clusters==4):
            #rot_deg = [0,90,180,270] #cluster rotations
            rot_deg = [0, 180, 10, 350]

            #hard-coded as of now
            cluster_idx[0:int(0.7*num_clients)] = np.zeros(int(0.7*num_clients),dtype='int')
            cluster_idx[int(0.7*num_clients):int(0.9*num_clients)] = 1*np.ones(int(0.2*num_clients),dtype='int')
            cluster_idx[int(0.9*num_clients):int(0.95*num_clients)] = 2*np.ones(int(0.05*num_clients),dtype='int')
            cluster_idx[int(0.95*num_clients):] = 3*np.ones(int(0.05*num_clients),dtype='int')

            cluster_0 = np.where(cluster_idx==0)[0]
            cluster_1 = np.where(cluster_idx==1)[0]
            cluster_2 = np.where(cluster_idx==2)[0]
            cluster_3 = np.where(cluster_idx==3)[0]
        elif(args.n_clusters == 2):
            rot_deg = [0, 180]
            cluster_0 = np.where(cluster_idx==0)[0]
            cluster_1 = np.where(cluster_idx==1)[0]
                    

    clients = []
    #start the training
    for idx in range(args.num_clients):
        if(args.iid == 'label'):
            rot_deg_i = 0
            
        elif(args.iid == 'covariate'):
            rot_deg_i = rot_deg[cluster_idx[idx]]
            cluster_list.append(rot_deg_i)

        client = ClientUpdate(train_dataset, dict_users[idx], 
                              dict_users_val[idx], criterion, args.lr, device, args.bs,
                              rot_deg_i, args.num_clients, args.dataset)
        clients.append(client)
                    
    sample_frac = 1.0
    if(args.pens):
        clients = train_pens(args.n_rounds_pens, args.local_ep, clients, sample_frac, args.n_sampled, args.top_m)
                    
        for i in range(len(clients)):
            expected_samples = (args.top_m/args.num_clients) * args.n_rounds_pens
            clients[i].neighbours = np.where(clients[i].n_selected > expected_samples)[0]
        
        clients = gossip(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.pens) 
        
    elif(args.DAC or args.DAC_var): 
        clients = train_DAC(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.tau, args.DAC_var)
    
    elif(args.random):
        for i in range(len(clients)):
            clients[i].neighbours = list(set(np.arange(num_clients))-set([i]))
        clients = gossip(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.pens)
    
    elif(args.oracle):
        for i in range(len(clients)):
            if(args.iid == 'covariate'):
                #here we choose neighbours using an oracle
                clients[i].neighbours = list(set(np.where(cluster_idx==cluster_idx[i])[0])-set([i]))
            elif(args.iid == 'label'):
                if(idx<40):
                    clients[i].neighbours = list(set(np.arange(0,40))-set([idx]))
                else:
                    clients[i].neighbours = list(set(np.arange(40,100))-set([idx]))
                
        clients = gossip(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.pens)
     
    
    client_heatmap = np.zeros((len(clients),len(clients)))
    for j in range(len(clients)):
        client_heatmap[:,j] = clients[j].n_sampled
    
    if(args.DAC or args.DAC_var):
        plt.figure(figsize=(12,8))
        sns.heatmap(client_heatmap)
        if(args.DAC):
            method_name = 'DAC'
        elif(args.DAC_var):
            method_name = 'DAC_var'
        plt.savefig(f"./save/figures/heatmap_{method_name}_{args.tau}_nsampled_{args.n_sampled}.eps")
    else:
        if(args.oracle):
            method_name = 'Oracle'
        elif(args.pens):
            method_name = 'PENS'
        elif(args.random):
            method_name = 'Random'
        plt.figure(figsize=(12,8))
        sns.heatmap(client_heatmap)
        plt.savefig(f"./save/figures/heatmap_{method_name}_nsampled_{args.n_sampled}.eps")
    
    acc_list0 = []
    acc_list1 = []
    acc_list2 = []
    acc_list3 = []

    if(args.iid == 'covariate'):
        test_loaders = []
        for nc in range(args.n_clusters):
            test_loader = create_test_data(nc,rot_deg,args.dataset)
            test_loaders.append(test_loader)

        for i in range(num_clients):
            if(i in cluster_0):
                _, acc = test(clients[i].best_model, criterion, test_loaders[0], device)
                acc_list0.append(acc)
            elif(i in cluster_1):
                _, acc = test(clients[i].best_model, criterion, test_loaders[1], device)
                acc_list1.append(acc)
            elif(i in cluster_2):
                _, acc = test(clients[i].best_model, criterion, test_loaders[2], device)
                acc_list2.append(acc)
            elif(i in cluster_3):
                _, acc = test(clients[i].best_model, criterion, test_loaders[3], device)
                acc_list3.append(acc)
    elif(args.iid == 'label'):
        if(args.dataset == 'cifar10'):
            trans_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            test_dataset = torchvision.datasets.CIFAR10('.', train=False, download=True, transform=trans_cifar_test)
            test_loader = DataLoader(test_dataset, batch_size=1)
        #elif(args.dataset == 'fashion-mnist'): #not yet implemented
        
        for i in range(args.num_clients):
            if(i<40):
                group_labels = np.array([0,1,8,9]) #vehicles
                _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group_labels)
                acc_list0.append(acc)
            else:
                group_labels = np.array([2,3,4,5,6,7]) #animals
                _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group_labels)
                acc_list1.append(acc)

    test_acc_0 = np.mean(acc_list0)
    test_acc_1 = np.mean(acc_list1)
    test_acc_2 = np.mean(acc_list2)
    test_acc_3 = np.mean(acc_list3)
                    
    with open('./save/'+filename,'a') as f1:
        f1.write(f'{args.n_rounds};{args.n_rounds_pens};{args.num_clients};{args.local_ep};{args.bs};{args.lr};{args.n_clusters};{args.pens};{args.DAC};{args.DAC_var};{args.oracle};{args.random};{args.top_m};{args.n_sampled};{args.n_data_train};{args.n_data_val};{args.tau};{test_acc_0};{test_acc_1};{test_acc_2};{test_acc_3};{args.dataset}')
        f1.write("\n")