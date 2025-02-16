# python libraries
import pandas as pd
import numpy as np
import os
import h5py
import argparse
from tqdm import tqdm
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from snorkel.classification import cross_entropy_with_probs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# custom
from models.MIL_model import MIL_SB
from AttentionDataset import AttentionDataset, InstanceDataset, instance_dataloader
from models.AttentionModel import GatedAttentionModel
from fold_split import stratified_k_fold_split, stratified_train_test_split

torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class color:
    reset = '\033[0m'
    BOLD = '\033[01m'
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    blue = '\033[34m'


def patient_id(row):
    if "patient" in row:
        return row.split("_node")[0]

    elif "TCGA" in row:
        ar = row.split("-")
        id = ""
        for i in range(4):
            id += ar[i] + "_"
        return id
    else:
        return row


class Trainer:
    def __init__(self, criterion, batch_save, model, train_loader, val_loader, save_path, train_dataset, val_dataset,
                 num_epochs=200, patience=20,
                 positional_embed=False):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device {self.device}")
        self.model = model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.positional_embed = positional_embed
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_save = batch_save
        self.criterion = criterion.to(self.device)
        print(f"Model Device: {next(self.model.parameters()).device}")
        print(f"Criterion Device: {getattr(self.criterion, 'device', 'N/A')}")

        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.best_val_loss = float('inf')
        self.paths = {
            "train": os.path.join(self.save_path, "training"),
            "val": os.path.join(self.save_path, "validation"),
            "weights": os.path.join(self.save_path, "weights"),
        }
        self.training_log_path = os.path.join(self.save_path, 'training_log.csv')
        self.patience_counter = 0
        self._setup_directories()
        self.best_weights_path = ""

    def _setup_directories(self):
        # saving outputs
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        with open(self.training_log_path, 'w', newline='') as f:
            pass

    def _save_patient_data(self, dataset, phase):
        phase_path = self.paths[phase]
        print(f"Initializing {phase} directories")
        for batch in tqdm(dataset, total = len(dataset)):
            bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id = batch
            patient_dir = os.path.join(phase_path, patient_id[0])
            patient_file = os.path.join(patient_dir, f"{patient_id[0]}.csv")
            os.makedirs(patient_dir, exist_ok=True)
    
            if patient_id[0] == "error" or os.path.isfile(patient_file):
                continue
            temp = pd.DataFrame()
            x = np.array(x.squeeze()).flatten()
            y = np.array(y.squeeze()).flatten()
            tile_paths = np.array(tile_paths).flatten()
            scales = np.repeat(scales, len(x))
            original_size = np.repeat(int(original_size), len(x))
    
            temp["x"] = x
            temp["y"] = y
            temp["tile_paths"] = tile_paths
            temp["scale"] = scales
            temp["size"] = original_size
    
            temp.to_csv(patient_file, index=False)
    def _calculate_accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        return correct / len(labels)

    def train(self):
        # train loop
        self._save_patient_data(self.train_loader, "train")
        self._save_patient_data(self.val_loader, "val")
        print("Training")
        with open(self.training_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header only if the file is empty
            if file.tell() == 0:
                writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', "Weights Saved"])

            for epoch in range(self.num_epochs):
                train_loss, train_accuracy = self._train_epoch(epoch)
                val_loss, val_accuracy = self._validate_epoch(epoch)

                print(f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

                saved = "YES" if self._save_best_model(val_loss, epoch) else None
                if saved == "YES":
                    self.best_weights_path = os.path.join(self.paths["weights"], f"weights_epoch_{epoch + 1}.pth")
                    print(f"Model saved with val_loss: {val_loss:.4f}")

                writer.writerow([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, saved])
                file.flush()

                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self, epoch):
        running_loss, running_correct, total, inst_count, train_inst_loss = 0.0, 0, 0, 0, 0
        

        for bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id in tqdm(self.train_loader,
                                                                                                  desc=f"Epoch {epoch + 1} [Train]"):
            if patient_id[0] == "error":
                continue
            try:
                bags, positional, labels = bags.to(self.device), positional.to(self.device), labels.to(self.device)
                self.model.train()
    
                if labels.dtype != torch.long:
                    labels = labels.long()
    
                self.optimizer.zero_grad()
                if positional is not None:
                    logits, Y_prob, _, A_raw, results_dict, h = self.model(bags, pos = positional,label=labels, instance_eval=True)
                else:
                     logits, Y_prob, _, A_raw, results_dict, h = self.model(bags, pos = None,label=labels, instance_eval=True)

                    
                labels_one_hot = nn.functional.one_hot(labels, num_classes=self.model.n_classes).float().to(self.device)
                # now get the loss
                logits = logits.to(self.device)
                loss = self.criterion(logits, labels_one_hot)
                inst_count += 1
                instance_loss = results_dict["instance_loss"].item()
                train_inst_loss += instance_loss
                c1 = 0.7
                total_loss = c1 * loss + (1 - c1) * instance_loss
                total_loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
    
                # if we want to save attention + get instance eval
                if epoch % self.batch_save == 0:
                    self._save_attention(epoch, A_raw, bags, positional, patient_id, h)
            except Exception as e : 
                print(e)
                continue

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = running_correct / total
        return train_loss, train_accuracy

    def _validate_epoch(self, epoch):
        running_loss, running_correct, total = 0.0, 0, 0
        for bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id in tqdm(self.val_loader,
                                                                                                  desc=f"Epoch {epoch + 1} [Val]"):
            if patient_id[0] == "error":
                continue
            try:
                bags, positional, labels = bags.to(self.device), positional.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    self.model.eval()
                    if positional is not None:
                        logits, Y_prob, _, A_raw, results_dict, h = self.model(bags, pos = positional ,label=labels, instance_eval=True)
                    else:
                         logits, Y_prob, _, A_raw, results_dict, h = self.model(bags, pos = None ,label=labels, instance_eval=True)
                   
                    labels_one_hot = nn.functional.one_hot(labels, num_classes=self.model.n_classes).float()
                    logits = torch.sigmoid(logits)
                    loss = self.criterion(logits, labels_one_hot)
    
                    running_loss += loss.item()
                    running_correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
    
                    if epoch % self.batch_save == 0:
                        self._save_attention(epoch, A_raw, bags, positional, patient_id, h, phase="val")
            except:
                    continue 

        val_loss = running_loss / len(self.val_loader)
        val_accuracy = running_correct / total
        return val_loss, val_accuracy

    def _save_best_model(self, val_loss, epoch):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            torch.save(self.model.state_dict(), os.path.join(self.paths["weights"], f"weights_epoch_{epoch + 1}.pth"))
            return True
        self.patience_counter += 1
        return False

    def _instance_eval(self, bags, positional):

        instances = instance_dataloader(bags)
        instance_pos = instance_dataloader(positional)
        instance_labels = []
        instance_probs = []
        self.model.eval()
        with torch.no_grad():
            for instance, position in zip(instances, instance_pos):
                if self.positional_embed:
                    values = instance.unsqueeze(1) + position.unsqueeze(1)
                else:
                    values = instance.unsqueeze(1)
                logits_inst, Y_prob_inst, Y_hat_inst, A_raw_inst, dict, h = self.model(values,
                                                                                       instance_eval=False)
                instance_labels.append(Y_hat_inst.item())
                instance_probs.append(Y_prob_inst[:, 1].item())
        return instance_labels, instance_probs

    def _save_attention(self, epoch, A_raw, bags, positional, patient_id, h, phase="train"):
        if phase == "train":
            patient_path = os.path.join(self.paths["train"], patient_id[0], f"{patient_id[0]}.csv")
            cluster_path = os.path.join(self.paths["train"], patient_id[0], f"{patient_id[0]}_cluster.h5")
        else:
            patient_path = os.path.join(self.paths["val"], patient_id[0], f"{patient_id[0]}.csv")
            cluster_path = os.path.join(self.paths["val"], patient_id[0], f"{patient_id[0]}_cluster.h5")
        patient_csv = pd.read_csv(patient_path)
        patient_csv[f"attention_{epoch + 1}"] = A_raw.squeeze(dim=1).squeeze().detach().cpu().numpy()
        instance_labels, instance_probs = self._instance_eval(bags, positional)

        instance_labels = [label.cpu().item() if isinstance(label, torch.Tensor) else label for label in
                           instance_labels]
        instance_probs = [prob.cpu().item() if isinstance(prob, torch.Tensor) else prob for prob in instance_probs]

        patient_csv[f"instance_label_{epoch + 1}"] = np.array(instance_labels)
        patient_csv[f"instance_prob_{epoch + 1}"] = np.array(instance_probs)
        patient_csv.to_csv(patient_path, index=False)

        #  save cluster info
        with h5py.File(cluster_path, "a") as file:
            if f'cluster_epoch_{epoch}' in file.keys():
                del file[f'cluster_epoch_{epoch}']

            file.create_dataset(f'cluster_epoch_{epoch}', data=h.detach().cpu().numpy())


def evaluate(model, dataloader, instance_eval=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for bags, positional, labels, _, _, _, _, _, _ in dataloader:
            try:
                bags, labels, positional = bags.to(device), labels.to(device), positional.to(device)
                logits, Y_prob, Y_hat, _, _, _ =  model(bags, pos = positional ,label=labels, instance_eval=instance_eval)
                all_probs.extend(Y_prob[:, 1].cpu().numpy())
                all_preds.extend(Y_hat.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
            except:
                continue 

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1_score': f1_score(all_labels, all_preds),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positional_embed", action="store_true")
    parser.add_argument("--testing_csv", default=None)
    parser.add_argument("--training_output", default="/mnt/c/Users/loren/Masters/test")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--k", default=20, type=int)
    parser.add_argument("--tile_selection", default="CLAM")
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--input_dim", default=2048, type=int)
    parser.add_argument("--hidden_dim1", default=512, type=int)
    parser.add_argument("--hidden_dim2", default=256, type=int)
    parser.add_argument("--metadata_path", default="/mnt/c/Users/loren/Masters/colorectal2/patient_files.csv")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--fold_number", default=5, type=int)
    parser.add_argument("--dropout", default=0.35, type=float)
    parser.add_argument("--k_causal", default=20, type=int)
    parser.add_argument("--batch_save", default=5, type=int)
    parser.add_argument("--position_type", default = None)
    return parser.parse_args()


def write_eval_crossval(fold_data, save_path, index):
    evaluation_df = pd.DataFrame(fold_data)
    metrics = {
        'accuracy': np.mean(evaluation_df["accuracy"].to_numpy()),
        'precision': np.mean(evaluation_df["precision"].to_numpy()),
        'recall': np.mean(evaluation_df["recall"].to_numpy()),
        'f1_score': np.mean(evaluation_df["f1_score"].to_numpy()),
        'roc_auc': np.mean(evaluation_df["roc_auc"].to_numpy()),
        'confusion_matrix': None
    }
    evaluation_df.loc[len(evaluation_df)] = metrics
    evaluation_df.index = index
    evaluation_df.to_csv(save_path)
    print(evaluation_df)


def main():
    args = get_args()
    # get the patient labels + ID
    label_data = pd.read_csv(args.metadata_path)
    label_data["sample_id"] = label_data["Patient ID"]
    label_data["patient_id"] = label_data["sample_id"].apply(patient_id)
    # train model with cross val
    # 1. reserve a test set (20%)
    if args.cv:
        train, test = stratified_train_test_split(data=label_data,
                                                  label_column='target',
                                                  patient_id_column='patient_id',
                                                  test_size=0.15,
                                                  random_state=42
                                                  )
        # 2. get folds from train CV set
        folds = stratified_k_fold_split(
            data=train,
            label_column='target',
            patient_id_column='patient_id',
            n_splits=args.fold_number,
            random_state=42
        )
        fold_metrics_val = []
        fold_metrics_testing = []
        index = []
        # 3. For fold, train model
        for i, (train_data, val_data) in enumerate(folds):
            print(f"\nFold {i + 1}")
            index.append(f"Folds {i + 1}")
            # get the loaders
            train_dataset = AttentionDataset(train_data.reset_index())
            val_dataset = AttentionDataset(val_data.reset_index())
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1)
            # initiate model
            instance_loss = cross_entropy_with_probs
            model = MIL_SB(instance_loss, input_dim=args.input_dim, hidden_dim1=args.hidden_dim1,
                           hidden_dim2=args.hidden_dim2, dropout_rate=args.dropout,
                           k=args.k, k_selection=args.tile_selection)
            # general criterion

            print(
                f" number of positive samples  {len(train_data[train_data['target'] == 1])} number of negative samples {len(train_data[train_data['target'] == 0])}")
            pos_num = len(train_data[train_data["target"] == 1]) / len(train_data)
            # weights would be
            weights = torch.tensor([pos_num, (1 - pos_num)])
            criterion = nn.BCEWithLogitsLoss(weight=weights)
            print(f"weights {weights}")

            save_path = os.path.join(args.training_output, f"fold_{i + 1}")
            trainer = Trainer(criterion, args.batch_save, model, train_loader, val_loader, save_path, train_dataset,
                              val_dataset, num_epochs=args.epochs,
                              patience=args.patience, positional_embed=args.positional_embed)

            trainer.train()
            print(f"Evaluating fold {i + 1}")
            best_weights = trainer.best_weights_path
            model.load_state_dict(torch.load(best_weights, weights_only=True))

            # now eval
            test_loader = DataLoader(AttentionDataset(test.reset_index()), batch_size=1)
            results = evaluate(model, test_loader, instance_eval=False)
            fold_metrics_testing.append(results)
            results = evaluate(model, val_loader, instance_eval=False)
            fold_metrics_val.append(results)
            write_eval_crossval([fold_metrics_testing[i], fold_metrics_val[i]],
                                os.path.join(save_path, "test-val_eval.csv"), ["Testing", "Validation", "Average"])

        print("Done with cross-val")

        index.append("Average")
        # get and save average performance for testing
        print("Testing:")
        write_eval_crossval(fold_metrics_testing, os.path.join(args.training_output, "testing_fold_evaluations.csv"),
                            index)
        print("Validation:")
        write_eval_crossval(fold_metrics_val, os.path.join(args.training_output, "validation_fold_evaluations.csv"),
                            index)

    # now do regular train-test-split

    train, test = stratified_train_test_split(data=label_data,
                                              label_column='target',
                                              patient_id_column='patient_id',
                                              test_size=0.4,
                                              random_state=42
                                              )
    val, test = stratified_train_test_split(data=test,
                                            label_column='target',
                                            patient_id_column='patient_id',
                                            test_size=0.5,
                                            random_state=42
                                            )
    train_dataset = AttentionDataset(train.reset_index())
    val_dataset = AttentionDataset(val.reset_index())
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    # initiate model
    instance_loss = cross_entropy_with_probs
    model = MIL_SB(instance_loss, input_dim=args.input_dim, hidden_dim1=args.hidden_dim1,
                   hidden_dim2=args.hidden_dim2, dropout_rate=args.dropout,
                   k=args.k, k_selection=args.tile_selection)
    # general criterion

    pos_num = len(train[train["target"] == 1]) / len(train)
    weights = torch.tensor([pos_num, (1 - pos_num)])
    criterion = nn.BCEWithLogitsLoss(weight=weights)
    print(f"weight {weights}")

    save_path = os.path.join(args.training_output, f"train-test-split")
    trainer = Trainer(criterion, args.batch_save, model, train_loader, val_loader, save_path, train_dataset,
                              val_dataset, num_epochs=args.epochs,
                              patience=args.patience, positional_embed=args.positional_embed)
    trainer.train()
    best_weights = trainer.best_weights_path
    model.load_state_dict(torch.load(best_weights, weights_only=True))
    # now eval
    test_loader = DataLoader(AttentionDataset(test.reset_index()), batch_size=1)
    results_test = evaluate(model, test_loader, instance_eval=False)
    results_val = evaluate(model, val_loader, instance_eval=False)

    write_eval_crossval([results_test, results_val], os.path.join(save_path, "test-val_eval.csv"),
                        ["Testing", "Validation", "Average"])


if __name__ == "__main__":
    main()
