import numpy as np
import copy

def sample_labels_iid(dataset, num_users, n_data_train, n_data_val):
    """
    Sample I.I.D. (labels) client data from MNIST/CIFAR10/FASHION-MNIST datasets
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_val = {}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, int(n_data_train), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        dict_users_val[i] = set(np.random.choice(all_idxs, int(n_data_val), replace=False))
        all_idxs = list(set(all_idxs) - dict_users_val[i])
        
    return dict_users, dict_users_val


def sample_cifargroups(dataset, num_users, n_data_train, n_data_val):

    group1 = np.array([0,1,8,9]) #vehicles
    group2 = np.array([2,3,4,5,6,7]) #animals
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    idxs1, idxs2 = np.array([]), np.array([])
    idxs1 = idxs1.astype(int)
    idxs2 = idxs1.astype(int)
    for x in group1:
        idxs1 = np.append(idxs1, idxs[x == labels[idxs]])
    
    for x in group2:
        idxs2 = np.append(idxs2, idxs[x == labels[idxs]])
        
    print(len(idxs1))
    print(len(idxs2))
    
    for i in range(num_users):
        if(i<int(num_users*0.4)): #vehicles
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
            
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
        else: #animals
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))
            
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))
        
    return dict_users, dict_users_val

def FedAvg(w,alpha):
    w_avg = copy.deepcopy(w[0])
    n_clients = len(w)
    alpha = alpha/np.sum(alpha)
    for l in w_avg.keys():
        w_avg[l] = w_avg[l].float() - w_avg[l]

    for l, layer in enumerate(w_avg.keys()): #for each layer
        for k in range(0,n_clients): #for each client
            w_avg[layer] += alpha[k]*w[k][layer]
    return w_avg

def min_max_scale_ab(x, a, b):
    x_new = a + ( (x - np.min(x) ) * (b-a) ) / (np.max(x)-np.min(x))
    return x_new

def softmax_scale(x, tau):
    x_new = np.exp(x*tau)/sum(np.exp(x*tau))
    return x_new

def tau_function(x,a,b):
    tau = 2*a/(1+np.exp(-b*x)) - a +1
    return tau

def train_DAC(n_rounds, n_local_epochs, clients, sample_frac, n_sampled, tau, new_method_var):
    
    for i in range(len(clients)):
        _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
        print(f"Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
        
    for round in range(n_rounds):
        idxs = np.random.choice(range(len(clients)), int(sample_frac*len(clients)), replace=False)
        for i in idxs:
            if(not clients[i].stopped_early):
                neighbour_list = list(range(len(clients)))
                if(round==0):
                    probas = np.ones(len(clients))/len(clients)
                    probas[i] = 0.0
                    probas = probas/np.sum(probas)
                else:
                    probas = clients[i].priors_norm
                #print(i,len(probas))
                neighbour_sampled = np.random.choice(neighbour_list, n_sampled, replace=False, p=probas)

                neighbour_stats = []
                for j in neighbour_sampled: #request models from n_sampled neighbours

                    model_j = clients[j].local_model
                    n_train = len(clients[j].train_set)
                    train_loss_ij, _ = clients[i].validate(model_j, train_set = True)
                    #train_loss_ii, _ = clients[i].validate(clients[i].local_model, train_set = True)
                    train_loss_ii = clients[i].train_loss_list[-1]
                    neighbour_stats.append((model_j.state_dict(), train_loss_ij, n_train, j))

                    clients[i].n_sampled[j] += 1

                w_avg = [neighbour_stats[k][0] for k in range(n_sampled)]
                w_avg.append(clients[i].local_model.state_dict())

                alpha = [neighbour_stats[k][2] for k in range(n_sampled)]
                alpha.append(len(clients[i].train_set))

                ik_loss = [neighbour_stats[k][1] for k in range(n_sampled)]
                
                ik_similarity = [1/(neighbour_stats[k][1]) for k in range(n_sampled)]
                
                neighbour_idx = [neighbour_stats[k][3] for k in range(n_sampled)]
                
                most_similar_k = neighbour_idx[np.argmax(ik_similarity)]
                
                for k in range(n_sampled):
                    clients[i].priors[neighbour_idx[k]] = ik_similarity[k] #train_loss_ii / ik_loss[k]
                
                neighbour_list = np.arange(len(clients))
                new_neighbours = []
                for k in neighbour_idx:
                    new_neighbours += list(set(neighbour_list[clients[k].priors > 0]) - set(neighbour_list[clients[i].n_sampled > 0]) - set([i]))
                
                new_neighbours = np.unique(new_neighbours)
                #print(new_neighbours)
                
                for j in new_neighbours:
                    score_kj_array = np.zeros(n_sampled)
                    ki_scores = []
                    for k in range(n_sampled):
                        score_kj = clients[neighbour_idx[k]].priors[j]
                        score_kj_array[k] = score_kj
                        if(score_kj>0):
                            #save tuple (ki_score, k)
                            ki_scores.append( (clients[i].priors[neighbour_idx[k]], k ))
                    
                    ki_scores.sort(key=lambda x:x[0]) #sort tuple
                    ki_max = ki_scores[-1][1] #choose k with max similarity to i
                    ki_max_score = ki_scores[-1][0]
                    
                    
                    #thresh = np.median(clients[i].priors[clients[i].n_sampled>0]) #of the ones i've communicated with?
                    #if(ki_max_score>thresh):
                    clients[i].priors[j] = score_kj_array[ki_max]
                    #else:
                    #    clients[i].priors[j] = 0
                
                not_i_idx = np.arange(len(clients[i].priors))!=i
                
                if(new_method_var):
                    tau_new = tau_function(round,tau,0.2)
                else:
                    tau_new = tau
                    
                clients[i].priors_norm[not_i_idx] = softmax_scale(clients[i].priors[not_i_idx], tau_new)
                
                w_new = FedAvg(w_avg,alpha)
                clients[i].local_model.load_state_dict(w_new)
                _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)

                print(f"Round {round} | Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
                #print(f"{np.round(clients[i].priors_norm,4)}")
    return clients


def train_pens(n_rounds, n_local_epochs ,clients,sample_frac, n_sampled, top_m):
    
    for i in range(len(clients)):
        _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
        print(f"Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
        
    for round in range(n_rounds):
        idxs = np.random.choice(range(len(clients)), int(sample_frac*len(clients)), replace=False)
        for i in idxs:
            if(not clients[i].stopped_early):
                neighbour_list = list(set(range(len(clients))) - set([i]))
                neighbour_sampled = np.random.choice(neighbour_list, n_sampled, replace=False)

                neighbour_stats = []
                for j in neighbour_sampled: #request models from n_sampled neighbours

                    model_j = clients[j].local_model
                    n_train = len(clients[j].train_set)
                    train_loss_ij, _ = clients[i].validate(model_j,train_set = True)
                    neighbour_stats.append((model_j.state_dict(), train_loss_ij, n_train, j))

                    clients[i].n_sampled[j] += 1

                neighbour_stats.sort(key=lambda x:x[1])
                w_avg = [neighbour_stats[k][0] for k in range(top_m)]
                w_avg.append(clients[i].local_model.state_dict())

                alpha = [neighbour_stats[k][2] for k in range(top_m)]
                alpha.append(len(clients[i].train_set))

                selected_neighbours = [neighbour_stats[k][3] for k in range(top_m)]

                for k in selected_neighbours:
                    clients[i].n_selected[k] += 1

                w_new = FedAvg(w_avg,alpha)
                clients[i].local_model.load_state_dict(w_new)
                _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)

                print(f"Round {round} | Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
            
    return clients

def gossip(n_rounds, n_local_epochs, clients, sample_frac, n_sampled, pens):
    
    for i in range(len(clients)):
        clients[i].count = 0
        clients[i].stopped_early = False
        if(not pens):
            _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
            print(f"Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
        
    for round in range(n_rounds):
        idxs = np.random.choice(range(len(clients)),int(sample_frac*len(clients)),replace=False)
        for i in idxs:
            if(not clients[i].stopped_early):
                if(n_sampled > len(clients[i].neighbours)):
                    n_sampled_neighbours = len(clients[i].neighbours)
                else:
                    n_sampled_neighbours = n_sampled

                neighbour_sampled = np.random.choice(clients[i].neighbours, n_sampled_neighbours, replace=False)
                neighbour_stats = []
                for j in neighbour_sampled: #request models from n_sampled neighbours

                    model_j = clients[j].local_model
                    n_train = len(clients[j].train_set)
                    neighbour_stats.append((model_j.state_dict(), n_train, j))
                    clients[i].n_sampled[j] += 1

                w_avg = [neighbour_stats[k][0] for k in range(n_sampled_neighbours)]
                w_avg.append(clients[i].local_model.state_dict())

                alpha = [neighbour_stats[k][1] for k in range(n_sampled_neighbours)]
                alpha.append(len(clients[i].train_set))
                w_new = FedAvg(w_avg,alpha)
                clients[i].local_model.load_state_dict(w_new)
                
                _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
                print(f"Round {round} | Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
            
    return clients

def train_pens2(n_rounds, n_local_epochs,clients,sample_frac,n_sampled,top_m):
    
    #for i in range(len(clients)):
    #    _, train_loss = clients[i].train(n_local_epochs)
    #    print(f"Client {i} | Train loss {np.round(train_loss,2)}")
        
    for round in range(n_rounds):
        idxs = np.random.choice(range(len(clients)),int(sample_frac*len(clients)),replace=False)
        for i in idxs:
            
            model_i = clients[i].local_model
            
            neighbour_list = list(set(range(len(clients)))-set([i]))
            neighbour_sampled = np.random.choice(neighbour_list,n_sampled,replace=False)
            
            neighbour_stats = []
            for j in neighbour_sampled: #request models from n_sampled neighbours
                
                model_j = clients[j].local_model
                n_train = len(clients[j].train_set)
                train_loss_ij = clients[i].validate_trainloss(model_j)
                neighbour_stats.append((model_j.state_dict(), train_loss_ij, n_train, j))
                
                clients[i].n_sampled[j] += 1
                
            neighbour_stats.sort(key=lambda x:x[1])
            w_avg = [neighbour_stats[k][0] for k in range(top_m)]
            w_avg.append(clients[i].local_model.state_dict())
            
            alpha = [neighbour_stats[k][2] for k in range(top_m)]
            alpha.append(len(clients[i].train_set))
            
            selected_neighbours = [neighbour_stats[k][3] for k in range(top_m)]
            
            for k in selected_neighbours:
                clients[i].n_selected[k] += 1
                clients[k].n_selected[i] += 1 #share information both ways
                
                #clients[i].n_selected += clients[k].n_selected #somethine like this?
            
            w_new = FedAvg(w_avg,alpha)
            clients[i].local_model.load_state_dict(w_new)
            _, train_loss = clients[i].train(n_local_epochs)
            print(f"Round {round} | Client {i} | Train loss {np.round(train_loss,2)}")
            
    return clients
