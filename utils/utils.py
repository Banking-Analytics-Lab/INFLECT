import pandas as pd 
import numpy as np 
from scipy.spatial.distance import pdist,squareform
import random 
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from torch.nn import BCEWithLogitsLoss
from statistics import mean
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss



def focal_loss(logits, labels, alpha, gamma=2):

    loss = BCEWithLogitsLoss(reduction="none")
    bc_loss = loss(logits, labels)

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class Loss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            Loss instance
        """
        super(Loss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = len(labels.tolist())
        num_classes = len(set(labels.tolist()))
        labels_one_hot = F.one_hot(labels.type(torch.int64), num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels.float(), alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


def upsample_embeddings(embed, labels,edges,sample_rate,end_index = None):

    n_nodes = len(set(edges.flatten()) )
    avg_number = n_nodes * sample_rate
    chosen = np.argwhere(labels.flatten() == 1).flatten()
    original_idx = embed.shape[0]
    #ipdb.set_trace()   

    c_portion = int(avg_number/chosen.shape[0])

    for j in range(c_portion):
        chosen_embed = embed[chosen,:]
        distance = squareform(pdist(chosen_embed.cpu().detach()))
        np.fill_diagonal(distance,distance.max()+100)
        idx_neighbor = distance.argmin(axis=-1)
        interp_place = random.random()

        new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place
        new_labels = torch.zeros((chosen.shape[0],1)).reshape(-1).fill_(1)
        embed = torch.cat((embed,new_embed), 0)
        labels = torch.cat((torch.tensor(labels),new_labels), 0)

    if end_index is not None: 
        embed = embed[:end_index]
    synthetic_index = [x for x in range( original_idx , embed.shape[0]  ,1) ]

    return embed, labels,synthetic_index


def  auprc_wrap(labels,scores):
    pres, recall, _ = precision_recall_curve(labels, scores)
    return recall,pres,_


def get_metrics(scores,labels, ): 

    fpr, tpr, thresholds = roc_curve( labels,scores)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    auprc = average_precision_score(labels,scores)
    return auc(fpr, tpr),auprc, optimal_threshold


def get_loss(loss) : 
    if loss =='bce': 
        return BCEWithLogitsLoss


def get_window(dates):
    return [[int(dates[i-1]),int(dates[i]),int(dates[i+1])] for i in range( 1,len(dates) -1 )]


def evauluate(model,windows,data_dict,loss_function,train_nodes,unseen_nodes_set,h0, device,loss_type,boot =False,boot_size =10000):
    
    losses = []
    auprcs = []
    auprcs_seen= []
    auprcs_unseen=[]
    aucs= []
    aucs_seen= []
    aucs_unseen = []

    for (i,month_window) in enumerate(windows): 
        m = month_window if type(month_window) == int else month_window[-1] 
        nodes_to_eval= list(set(data_dict[m].edge_index.flatten()))
        unseen_nodes = list(set(nodes_to_eval ) & unseen_nodes_set)
        seen_nodes = list(set(nodes_to_eval) & train_nodes)
        scores_for_loss,labels , h0, _ = model(month_window, data_dict,device,h0, train = False) 
        scores_for_loss = torch.Tensor(scores_for_loss.detach().flatten()).float()
        scores_seen = scores_for_loss[seen_nodes]
        labels_seen = labels[seen_nodes]     
        scores_unseen = scores_for_loss[unseen_nodes]
        labels_unseen = labels[unseen_nodes]
        scores_for_loss = scores_for_loss[nodes_to_eval]
        labels = labels[nodes_to_eval]

        if loss_type == 'focal':
            loss = loss_function(scores_for_loss.to(device), labels.type(torch.int64).to(device))
        else:
            loss = loss_function(scores_for_loss.to(device), labels.to(device))

        auc, auprc, threshold = get_metrics(torch.sigmoid(scores_for_loss).cpu(),labels.cpu())
        seen_auc, seen_auprc, seen_threshold = get_metrics(torch.sigmoid(scores_seen).cpu(),labels_seen.cpu())
        unseen_auc, unseen_auprc, unseen_threshold = get_metrics(torch.sigmoid(scores_unseen).cpu(),labels_unseen.cpu())
     
        print(f'SEEN AUPRC FOR WINDOW {month_window} IS {seen_auprc}')
        print(f'UNSEEN AUPRC FOR WINDOW {month_window} IS {unseen_auprc}')

        losses.append(loss.cpu())
        auprcs.append(auprc)
        auprcs_seen.append(seen_auprc)
        auprcs_unseen.append(unseen_auprc)
        aucs.append(auc)
        aucs_seen.append(seen_auc)
        aucs_unseen.append(unseen_auc)

        if boot: 
            seen_boot = bootstrap_preds( scores_seen, labels_seen,num_boot = boot_size)
            unseen_boot = bootstrap_preds( scores_unseen, labels_unseen,num_boot =  boot_size)
            d_seen_auc = get_stats(seen_boot, metric = 'seen_auc')
            d_unseen_auc = get_stats(unseen_boot, metric = 'unseen_auc')
            auprc_seen_boot = bootstrap_preds_auprc(scores_seen, labels_seen,boot_size)
            auprc_unseen_boot = bootstrap_preds_auprc( scores_unseen, labels_unseen,boot_size)
            d_seen_auprc =get_stats(auprc_seen_boot, metric = 'seen_auprc')
            d_unseen_auprc = get_stats(auprc_unseen_boot, metric = 'unseen_auprc')
            boot_dict = dict(d_seen_auc , **d_unseen_auc)
            boot_dict = dict(boot_dict,**d_seen_auprc) 
            boot_dict = dict(boot_dict,**d_unseen_auprc)

            del seen_boot, unseen_boot, d_seen_auc, d_unseen_auc, auprc_seen_boot, auprc_unseen_boot, d_seen_auprc, d_unseen_auprc

        else:
            boot_dict = None

        del unseen_nodes
        del seen_nodes
        del scores_seen
        del scores_unseen
        del labels_seen
        del labels_unseen

    return np.mean(aucs),np.mean(aucs_seen),np.mean(aucs_unseen),np.mean(auprcs),np.mean(auprcs_seen),np.mean(auprcs_unseen),np.mean(losses),boot_dict,h0,\
         torch.sigmoid(scores_for_loss),labels, nodes_to_eval,m


def print_log(names,metrics,e):
    for n,m in zip(names,metrics):
        print(f"{n} = {m} at epoch {e}")


def bootstrap_preds(preds, labs,curve_fun = roc_curve,num_boot = 10000):

    n = len(preds)
    boot_means = np.zeros(num_boot)
    data = pd.DataFrame({'preds':preds.cpu().flatten(),'labs':labs.cpu().flatten()})

    np.random.seed(0)
    for i in range(num_boot):
        label_zero = data[data.labs == 0]
        label_zero_len = len(labs[labs == 0])
        label_one = data[data.labs == 1]
        label_one_len = len(labs[labs == 1])
        d_zero = label_zero.sample(label_zero_len, replace=True)
        d_one = label_one.sample(label_one_len, replace=True)
        d = pd.concat([d_zero, d_one])
        fpr, tpr, thresholds = curve_fun(d.labs,d.preds)
        
        boot_means[i] = auc(fpr,tpr)

    return boot_means


def bootstrap_preds_auprc(preds, labs,curve_fun = average_precision_score,num_boot = 10000):

    n = len(preds)
    boot_means = np.zeros(num_boot)
    data = pd.DataFrame({'preds':preds.cpu().flatten(),'labs':labs.cpu().flatten()})

    np.random.seed(0)
    for i in tqdm(range(num_boot)):
        label_zero = data[data.labs == 0]
        label_zero_len = len(labs[labs == 0])
        label_one = data[data.labs == 1]
        label_one_len = len(labs[labs == 1])
        d_zero = label_zero.sample(label_zero_len, replace=True)
        d_one = label_one.sample(label_one_len, replace=True)
        d = pd.concat([d_zero, d_one])

        boot_means[i] = average_precision_score(d.labs,d.preds)

    return boot_means


def get_stats(arr,metric = 'auc'): 
    boot_dict = {
        f'{metric}_boot_mean' : np.mean(arr),
        f'{metric}_boot_std' : np.std(arr),
        f'{metric}_boot_q0' : np.quantile(arr,0),
        f'{metric}_boot_q1' : np.quantile(arr,.25),
        f'{metric}_boot_q2' : np.quantile(arr,.5),
        f'{metric}_boot_q3' : np.quantile(arr,.75),
        f'{metric}_boot_max' : np.quantile(arr,1),
    }
    return boot_dict


def write_log(metrics, log_file, run_name, model):
    p = f'./INFLECT'
    log_file = p + log_file
    metrics.insert(0,run_name) 
    df = pd.DataFrame({c:[r] for c,r in zip(range(len(metrics)),metrics)})
    print(df)
    with open(log_file, 'a') as f:
        df.to_csv(f, header=False,index = False)


def bootstrap_prob(profit,proba_from = 0.3, proba_to = 0.6):
    profits= []
    for i in range(1000):
        proba = random.uniform(proba_from, proba_to)
        profit_est = profit*(1-proba)
        profits.append(profit_est)
    return mean(profits)


def profit_calculation(fin, base, nodes_for_profit, labels_for_profit, scores_for_profit, m, run,  inv_map, dict_v, time_map, threshold, data_name, output_name, type):

    predictions = make_predictions(nodes_for_profit, labels_for_profit, scores_for_profit, m, run)
    predictions['u'] = predictions.apply(lambda x: inv_map[x['nodes']-1] if x['nodes']-1 in inv_map.keys() else 'None', axis = 1)
    predictions['v'] = predictions.apply(lambda x: dict_v[x['month']][x['u']] if dict_v[x['month']].get(x['u']) else 'None', axis = 1)

    predictions['profit'] = predictions.apply(lambda x: fin[(fin['START_DATE'].isin([x['month'], x['month'] + 1, x['month'] + 2, x['month'] + 3, x['month'] + 4, x['month'] + 5])) 
                                                            & (fin.APPLICATION_USER_ID.isin(list(map(int,x['v']))))]['PROFIT'].sum() 
                    if x['v'] != 'None' else 0, axis = 1)
    
    predictions['pred_label'] = predictions.apply(lambda x: 0 if x['scores'] <= threshold else 1, axis = 1)

    predictions.to_csv(f'./INFLECT/output_{output_name}/{run}_{type}_for_profit.csv')
    #return predictions
    

def make_predictions(nodes_for_profit, labels_for_profit, scores_for_profit, m, run_name):
    return pd.DataFrame({'nodes':nodes_for_profit, 'labels': labels_for_profit.cpu(), 'scores':scores_for_profit.cpu(), 'month': m})