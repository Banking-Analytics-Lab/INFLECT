import os 
from copy import deepcopy
from statistics import mean 
import numpy as np
from tqdm import tqdm 
from pprint import pprint
import pandas as pd
import torch 
from torch.optim import Adam
import argparse
from utils.utils import get_loss,get_metrics,get_window,evauluate, print_log, write_log, profit_calculation, Loss, focal_loss
import wandb
from models.models import get_model 
import numpy as np
import random
torch.manual_seed(1)
np.random.seed(0)
import pickle
from carbontracker.tracker import CarbonTracker

def get_var_name(variable):
    names = []
    for v in variable:
        for name, value in globals().items():
            if value is v:
                names.append(name)
    return names


device = torch.device('cuda:0')
# device = torch.device('cpu')
print(device)

parser = argparse.ArgumentParser('Training execution')
parser.add_argument('--GNN', type=str, help='GAT, GIN',default='')
parser.add_argument('--RNN', type=str, help='RNN, GRU/LSTM',default = '')
parser.add_argument('--DECODER', type=str, help='decoder',default = 'LIN')
parser.add_argument('--gnn_input_dim', type=int, help='input dim for GNN, defaults to number of features',default = 31)
parser.add_argument('--gnn_output_dim', type=int, help='output dim of gnn, matches input dim of RNN in combined models',default =200)
parser.add_argument('--embedding_dim', type=int, help='hidden dimension for GNN layers',default =200)
parser.add_argument('--heads', type=int, help='attention heads',default =2)
parser.add_argument('--dropout_rate', default = 0.2, help='Dropout rate',type = float)
parser.add_argument('--RNN_hidden_dim', type=int, help='hidden dim for RNNs',default =200)
parser.add_argument('--RNN_layers', type=int, help='layers for RNN',default =1)
parser.add_argument('--GNN_layers', type=int, help='layers for gnn',default =2)
parser.add_argument('--upsample_rate', type=float, help='upsample rate for embedding upsampleing',default =0)
parser.add_argument('--page_rank', action='store_true', help='if PageRank feature is included')
parser.add_argument('--epochs', type=int, help='epochs',default =5)
parser.add_argument('--lr', type=float, help='learning rate',default =.0001)
parser.add_argument('--eps', type=float, help='epsilon for GIN',default =0)
parser.add_argument('--boot_sample', type=int, help='sample size for bootstrap',default =10000)
parser.add_argument('--data_name', type=str, help='data name',default = 'TGN_paper')
parser.add_argument('--wandb_name', type=str, help='wandb folder name',default = 'TEST')
parser.add_argument('--output_name', type=str, help='model output folder name',default = 'output')
parser.add_argument('--loss', type=str, help='loss name',default = 'bce')
parser.add_argument('--full_windows', action='store_true', help='Use full windows or not')
parser.add_argument('--train_eps', action='store_true', help='Train eps for GIN')
parser.add_argument('--search_depth_SAGE', type=int, help='Depth for SAGE search',default =2)
parser.add_argument('--run_name', type=str, help='Name with combination of hps being run',default = '')
parser.add_argument('--log_file', type=str, help='Log file',default = 'logs.csv')
parser.add_argument('--fl_gamma', type=float, help='gamma for focal loss',default = 2.0)
parser.add_argument('--fl_beta', type=float, help='beta for focal loss',default = 0.999)
args = parser.parse_args()

RNN = args.RNN
GNN = args.GNN
fl_gamma = args.fl_gamma
fl_beta = args.fl_beta
page_rank = args.page_rank
boots = args.boot_sample
loss_type = args.loss
wandb_name = args.wandb_name
output_name = args.output_name

if page_rank:
    GNN_type = 'page_rank'
elif not GNN: 
    GNN_type = 'GCN'
else: 
    if GNN == 'GIN' or GNN == 'GAT' or GNN == 'SAGE': 
        GNN_type = 'GAT'
        GNN = 'GAT'
    else: 
        GNN
        GNN_type = 'GCN'

with open('./data/time_map.pkl', 'rb') as f:
    time_map = pickle.load(f)

data_path = f'./data_pay/{args.data_name}_{GNN_type}.pt'
run = args.run_name
base = f'./data_folder/{args.data_name}/'


j = 1
for i in range(6,13):
    time_map[pd.Timestamp(2022,i,1)] = 16+j
    j = j+1

fin = pd.read_csv(f'./data_folder/{args.data_name}/' + f'financial_for_profit_{args.data_name}.csv', usecols = ['START_DATE','APPLICATION_USER_ID', 'PROFIT'])

wandb.init(project=f"{wandb_name}", entity="INFLECT", name =f'{run}', dir = f'./wandb_{wandb_name}')

data_dict = torch.load(data_path) 

ts_list = list(data_dict.keys())
ts_list = [int(x) for x in ts_list]
n_nodes =   data_dict[ts_list[0]].x.shape[0]
n_feats =   data_dict[ts_list[0]].x.shape[1]
rnn_input_dim = args.gnn_output_dim if GNN else n_feats

rnn_kw = { 
    'RNN'  : args.RNN,
    'rnn_input_dim' : rnn_input_dim,
    'rnn_hidden_dim' : args.RNN_hidden_dim,
    'rnn_layers' : args.RNN_layers,
    'upsample' : args.upsample_rate,
    'n_nodes' : n_nodes 
}
gnn_kw ={ 
    'GNN' : args.GNN,
    'gnn_input_dim': args.gnn_input_dim,
    'gnn_embedding_dim' : args.embedding_dim,
    'heads' : args.heads,
    'dropout_rate': args.dropout_rate,
    'edge_dim' :data_dict[ts_list[0]].edge_attr.shape[1] if GNN == 'GAT' else None,
    'gnn_output_dim' : args.gnn_output_dim,
    'gnn_layers': args.GNN_layers,
    'eps': args.eps, 
    'train_eps' : args.train_eps,
    'search_depth': args.search_depth_SAGE
}
decoder_kw = {
    "DECODER": args.DECODER
}
epochs = args.epochs
tracker = CarbonTracker(epochs=int(epochs), monitor_epochs = epochs)


if args.full_windows: 
    windows=    ts_list[0:-2]
    val_window = ts_list[-2:-1]
    test_window = ts_list[-1:]
    print(windows,val_window,test_window)
    train_nodes= set(list(data_dict[windows[-1]].edge_index.flatten()))
    val_nodes = set(list(data_dict[val_window[-1]].edge_index.flatten())).difference(train_nodes)
    test_nodes = set(list(data_dict[test_window[-1]].edge_index.flatten())).difference(train_nodes)

else: 
    full_windows = get_window(ts_list)

    windows = full_windows[0:-2]
    val_window = full_windows[-2:-1]
    test_window = full_windows[-1:]


    print(windows,val_window,test_window)
    train_nodes= set(list(data_dict[windows[-1][-1]].edge_index.flatten()))
    val_nodes = set(list(data_dict[val_window[-1][-1]].edge_index.flatten())).difference(train_nodes)
    test_nodes = set(list(data_dict[test_window[-1][-1]].edge_index.flatten())).difference(train_nodes)

model = get_model(gnn_kw=gnn_kw,rnn_kw=rnn_kw,decoder_kw=decoder_kw).to(device)
optimizer  = Adam(model.parameters() , lr = args.lr)


train_losses = []
auc_list = []
auprc_list = []
val_auc_seen = []
val_auc_unseen = []
val_auprc_seen = []
val_auprc_unseen = []
min_val_loss = np.inf
best_e = 0 

for e in tqdm(range(epochs)): 
    tracker.epoch_start() 
    model.train()
    loss = 0 
    running_auprc = []
    running_auc = []
    running_loss = []
    h0 = None
    for  month_window in windows:

        optimizer.zero_grad()
        scores,labels , h0,synth_index = model(month_window, data_dict, device, h0) 

        m = month_window if type(month_window) == int else month_window[-1] 
        nodes_to_backprop = list(set(data_dict[m].edge_index.flatten())) + synth_index

        labels = labels[nodes_to_backprop].flatten()
        scores_for_loss = torch.Tensor(scores[nodes_to_backprop]).float().flatten()

        weight = len(labels[labels == 0])/len(labels[labels == 1])
        prior_class_threshold = 1 - len(labels[labels == 1])/len(labels)

        # NEW LOSS
        if loss_type == 'focal':

            focal_loss = Loss(
                loss_type="focal_loss",
                samples_per_class=[len(labels[labels == 0]), len(labels[labels == 1])],
                class_balanced=True, 
                fl_gamma = fl_gamma,
                beta = fl_beta
            )
            loss_function = focal_loss
            loss = loss_function(scores_for_loss, labels.type(torch.int64))

        elif loss_type == 'bce':
            loss_function = get_loss(args.loss)(pos_weight = torch.FloatTensor ([weight]).to(device))
            loss = loss_function(scores_for_loss, labels)
        
        loss.backward()
        optimizer.step()
        auc,auprc, train_threshold = get_metrics(torch.sigmoid(torch.Tensor(scores_for_loss.detach().cpu().numpy())),labels.cpu())

        
        running_auc.append(auc)
        running_auprc.append(auprc)
        running_loss.append(loss.item())

    model.eval()

    h0_to_save = h0

    print(device)
    val_auc, val_seen_auc, val_unseen_auc, val_auprc,val_seen_auprc,val_unseen_auprc,val_losses,_,h0,\
       scores,\
    labels, nodes_to_eval, train_m =evauluate(model, val_window ,data_dict,loss_function,train_nodes,val_nodes,h0, device, loss_type = loss_type)
    train_auc = mean(running_auc)
    train_auprc = mean(running_auprc)
    train_loss = mean(running_loss)
    auc_list.append(train_auc)
    auprc_list.append(train_auprc)
    train_losses.append(train_loss)

    metrics = [val_auc,val_seen_auc,val_unseen_auc, val_auprc,val_seen_auprc,val_unseen_auprc,train_auc,train_auprc,train_loss,val_losses]
    names = get_var_name(metrics)
    print_log(names,metrics,e)

    logs = {n:m for n,m in zip(names,metrics)}
    wandb.log(logs)


    if min_val_loss >val_losses  :
        min_val_loss = val_losses 
        best_model = deepcopy(model.state_dict())
        best_e = e 
        best_h0 = h0_to_save

    tracker.epoch_end()

tracker.stop()

del train_losses, running_loss, scores_for_loss, labels, nodes_to_backprop, synth_index, tracker, names, weight, scores, optimizer, loss, h0_to_save, \
      val_auc, val_seen_auc, val_unseen_auc, val_auprc, val_seen_auprc, val_unseen_auprc, val_losses, nodes_to_eval

model.load_state_dict(best_model)

model.eval()

with open(base + 'inverse_node_map.pkl', 'rb') as f:
    inv_map = pickle.load(f)

with open(base + 'dict_v.pkl', 'rb') as f:
    dict_v = pickle.load(f)

val_final_auc, val_final_seen_auc, val_final_unseen_auc, val_final_auprc, val_final_seen_auprc,val_final_unseen_auprc,val_final_losses,val_boot,val_h0, val_scores,\
    val_labels, val_nodes_to_eval, val_m = evauluate(model, val_window ,data_dict,loss_function,train_nodes,val_nodes,best_h0, device,boot =True,boot_size = boots,loss_type = loss_type)

profit_calculation(fin, base, val_nodes_to_eval, val_labels, val_scores, val_m, run, inv_map, dict_v,time_map, prior_class_threshold, args.data_name, output_name, type = 'val')

del val_nodes_to_eval
del val_labels
del val_scores
del val_m

test_final_auc, test_final_seen_auc,test_final_unseen_auc, test_final_auprc, test_final_seen_auprc,test_final_unseen_auprc,test_final_losses,test_boot, _, \
    test_scores,\
    test_labels, test_nodes_to_eval, test_m = evauluate(model, test_window ,data_dict,loss_function,train_nodes,test_nodes,val_h0, device,boot= True,boot_size =boots,loss_type = loss_type)

profit_calculation(fin, base, test_nodes_to_eval, test_labels, test_scores, test_m, run, inv_map, dict_v,time_map, prior_class_threshold, args.data_name, output_name, type = 'test')

del test_nodes_to_eval
del test_labels
del test_scores
del test_m


metrics =[val_final_auc,val_final_seen_auc,val_final_unseen_auc,val_final_auprc,val_final_seen_auprc,val_final_unseen_auprc,val_final_losses,test_final_auc,test_final_seen_auc, \
          test_final_unseen_auc,test_final_auprc,test_final_seen_auprc,test_final_unseen_auprc,test_final_losses, train_auc, train_auprc, val_boot['seen_auc_boot_mean'], val_boot['seen_auc_boot_std'], \
            val_boot['seen_auprc_boot_mean'], val_boot['seen_auprc_boot_std'],val_boot['unseen_auc_boot_mean'], val_boot['unseen_auc_boot_std'],val_boot['unseen_auprc_boot_mean'], \
                val_boot['unseen_auprc_boot_std'],test_boot['seen_auc_boot_mean'], test_boot['seen_auc_boot_std'],test_boot['seen_auprc_boot_mean'], test_boot['seen_auprc_boot_std'], \
                    test_boot['unseen_auc_boot_mean'], test_boot['unseen_auc_boot_std'],test_boot['unseen_auprc_boot_mean'], test_boot['unseen_auprc_boot_std']]


names = get_var_name(metrics)

print_log(names,metrics,best_e)

print('val boot')
pprint(val_boot)
print('test boot')
pprint(test_boot)
pprint(logs)

write_log(metrics, args.log_file, args.run_name, model)