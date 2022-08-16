import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rounds', type=int, default=10, help="rounds of training")
    parser.add_argument('--n_rounds_pens', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--bs', type=int, default=8, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    
    parser.add_argument('--a', type=float, default='0', help="a value in minmax scale")
    parser.add_argument('--b', type=float, default='1', help="b value in minmax scale")
    parser.add_argument('--tau', type=float, default='1', help="temperature in softmax")
    
    parser.add_argument('--n_sampled', type=int, default=5)
    parser.add_argument('--top_m', type=int, default=2)
    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--pens', action='store_true')
    parser.add_argument('--DAC', action='store_true')
    parser.add_argument('--DAC_var', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--random', action='store_true')
    
    parser.add_argument('--n_data_train', type=int, default=100, help="train size")
    parser.add_argument('--n_data_val', type=int, default=100, help="validation size")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--iid', type=str, default='covariate', help="covariate or label (type of shift)")
    #parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args