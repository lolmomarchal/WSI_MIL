import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from snorkel.classification import cross_entropy_with_probs
import warnings
warnings.filterwarnings("ignore")

from models.AttentionModel import GatedAttentionModel
# instance_loss=cross_entropy_with_probs,
class MIL_SB(nn.Module):
    def __init__(self, instance_loss,  k=20, classes=2,  input_dim=2048, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.25, k_selection = "shuffle"):
        super().__init__()
        self.instance_loss = instance_loss
        self.k = k
        self.k_selection = k_selection
        self.n_classes = classes
        self.attention_net = GatedAttentionModel(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout_rate=dropout_rate, classes=1)
        self.classifiers = nn.Linear(hidden_dim1, classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(hidden_dim1, 2) for _ in range(classes)])
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.relu = nn.ReLU()
        self.positional_ = nn.Linear(input_dim*2, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.k_middle = self.k

    def instance_evaluation(self, A, h, classifier):
        #print("Eval in")
        # print(f"h shape")
        # print(h.shape)

        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        A = A.view(-1)
        # print(f"A shape reshaping")
        # print(A.shape)

        top_k_indices = torch.topk(A, self.k)[1]
        # print(f"top_k_indices {top_k_indices.shape}")

        top_k = torch.index_select(h, dim=0, index=top_k_indices)
        # print(f"top_k {top_k.shape}")

        bottom_k_indices = torch.topk(-A, self.k)[1]
        bottom_k = torch.index_select(h, dim=0, index=bottom_k_indices)

        # print(f"bottom_k_indices {top_k_indices.shape}")
        # print(f"bottom_k {top_k.shape}")


        top_targets = torch.full((self.k,), 1, device=device).long()
        bottom_targets = torch.full((self.k,), 0, device=device).long()

        # print(f"top_targets {top_targets.shape}")
        # print(f"bottom_targets {bottom_targets.shape}")

        # Only gets the top and bottom targets
        if self.k_selection == "CLAM":
            all_targets = torch.cat([top_targets, bottom_targets], dim=0)
            all_instances = torch.cat([top_k, bottom_k], dim=0)
            # print(f"all_targets {all_targets.shape}")
            # print(f"all_instances {all_instances.shape}")

        # gets top, middle, and bottom targets (adding noise from the "middle portion")
        elif self.k_selection == "shuffle":

            excluded_indices = torch.cat([top_k_indices, bottom_k_indices])
            excluded_indices = excluded_indices.unique()
            # print(f"excluded_indices {excluded_indices.shape}")

            all_indices = torch.arange(A.size(0), device=device)
            mask = ~torch.isin(all_indices, excluded_indices)
            remaining_indices = all_indices[mask]
            # print(f"remaining_indices {remaining_indices.shape}")


            #shuffling the remaining indices and slecting a random sample of k
            if remaining_indices.size(0) > self.k:
                selected_indices = remaining_indices[torch.randperm(remaining_indices.size(0))[:self.k]]
            #if less remaining tiles than k, choose all of the remaining indices
            else:
                selected_indices = remaining_indices
            # print(f"remaining_indices {remaining_indices.shape}")

            # getting the middle targets at the selected indexes
            mid_k = torch.index_select(h, dim=0, index=selected_indices)
            mid_targets = torch.full((mid_k.size(0),), 0.5, device=device).long()
            # print(f"mid_k {mid_k.shape}")
            # print(f"mid_targets {mid_targets.shape}")

            # joining all instances and their targets
            all_instances = torch.cat([top_k, bottom_k, mid_k], dim=0)
            all_targets = torch.cat([top_targets, bottom_targets, mid_targets], dim=0)
            # print(f"all_instances {all_instances.shape}")
            # print(f"all_targets {all_targets.shape}")
        elif self.k_selection == "middle":
            # this one doesn't shuffle, instead it chooses from directly in the middle

           sorted_indices = torch.argsort(A)
    
           remaining_indices = sorted_indices[self.k: -self.k]
           if len(remaining_indices) <= self.k_middle:
                mid_indices = remaining_indices
           else:
                mid_start = (len(remaining_indices) - self.k_middle) // 2
                mid_indices = remaining_indices[mid_start: mid_start + self.k_middle]
        
           mid_k = torch.index_select(h, dim=0, index=mid_indices)
           mid_targets = torch.full((mid_k.size(0),), 0.5, device=device).long()
        
           all_instances = torch.cat([top_k, bottom_k, mid_k], dim=0)
           all_targets = torch.cat([top_targets, bottom_targets, mid_targets], dim=0)


        logits = classifier(all_instances)
        probs = F.softmax(logits, dim=1)
        # print(f"probs {probs.shape}")
        all_targets_one_hot = torch.nn.functional.one_hot(all_targets, num_classes=2).float()
        # print(f"all_targets_one_hot {all_targets_one_hot.shape}")
        instance_loss = self.instance_loss(probs, all_targets_one_hot)
        # print(f"instance_loss {instance_loss}")
        return instance_loss, torch.topk(logits, 1, dim=1)[1].squeeze(1), all_targets

    def instance_evaluation_out(self, A, h,  classifier):
        device = h.device
        # print("----")
        #print("evaluation out")
        if len(A.shape) == 1:
            A = A.view(1, -1)
        A = A.view(-1)
        #print(f"A {A.shape}")
        #print(f"h {h.shape}")
        top_k = torch.index_select(h, dim=0, index=torch.topk(A, self.k)[1][-1])

        top_targets = torch.full((self.k, ), 0, device=device).long().unsqueeze(0)
        # print(f"top_k {top_k.shape}")
        # print(f"top_targets {top_k.shape}")

        logits = classifier(top_k)
        probs = F.softmax(logits, dim=1)
        instance_loss = self.instance_loss(probs, top_targets)
        return instance_loss, torch.topk(logits, 1, dim=1)[1].squeeze(1), top_targets
    def instance_evaluation_middle(self, A, h,  classifier):
        device = h.device

        if len(A.shape) == 1:
            A = A.view(1, -1)
        A = A.view(-1)
        top_k = torch.index_select(h, dim=0, index=torch.topk(A, self.k)[1][-1])
        top_targets = torch.full((self.k, ), 0.5, device=device).long().unsqueeze(0)
        logits = classifier(top_k)
        probs = F.softmax(logits, dim=1)
        instance_loss = self.instance_loss(probs, top_targets)
        return instance_loss, torch.topk(logits, 1, dim=1)[1].squeeze(1), top_targets

    def forward(self, h, label = 1, pos = None, instance_eval=True, return_features=False, attention_only=False):

        # print("-------")
        # print("new batch")
        if pos is not None:
            h = torch.cat([h,pos], dim =-1).float()
            h = self.positional_(h)
            h = self.relu(h)
            h = self.dropout(h)
            
        A, h = self.attention_net(h.float())
        # print(f"h shape {h.shape}")
        A_raw = A

        # print("MIL one branched")
        # print(f"shape of h after attention net: {h.shape}")
        h = h.squeeze(0)
        # print(f"shape of h after squeezing: {h.shape}")
        if attention_only:
            return A
        A = F.softmax(A, dim = 1)
        # print(f"shape of Attention after softmax: {A.shape}")
        if instance_eval:
            # instance evaluation -> calculating the loss
            # here according to CLAM what we are doing is setting high attention as being in class and low attention being out of class
            # should only be doing it to ensure that
            total_instance_loss = 0
            all_predictions = []
            all_targets = []
            # turns the class label (for the sample) into a tensor
            label = torch.tensor(label).long()
            #print(f"sample label {label}")
            # the possible class labels for the instances are then the # of classes hot coded
            instance_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            #print(f"len classifiers: {len(self.instance_classifiers)}")
            #print(f"instance_labels: instance_labels")
            for i in range(len(self.instance_classifiers)):

                instance_label = instance_labels[i].item()
                # print(f"instance label: {instance_label}")

                classifier = self.instance_classifiers[i]

                if instance_label == 1:
                    instance_loss, preds, targets = self.instance_evaluation(A, h, classifier)
                    all_predictions.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    instance_loss, preds, targets = self.instance_evaluation_out(A, h, classifier)
                    all_predictions.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                total_instance_loss += instance_loss
                # print(f"total_instance_loss: {total_instance_loss}")
        A = A.squeeze(0).T

        # print(f"shape of Attention after squeezing 0 dimension and transposing: {A.shape}")
        M = torch.mm(A, h)
        # print(f"final mult shape {M.shape}")
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            all_targets = [item if np.isscalar(item) else item[0] for item in all_targets]
            results_dict = {'instance_loss': total_instance_loss, 'inst_labels': np.array(all_targets), 'inst_preds': np.array(all_predictions)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict, h
