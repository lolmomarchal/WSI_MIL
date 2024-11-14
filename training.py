# python libraries
import pandas as pd
import numpy as np
import torch
import os
import h5py
import argparse
from tqdm import tqdm
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from snorkel.classification import cross_entropy_with_probs
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# custom

from MIL_model import MIL_SB
from AttentionDataset import AttentionDataset, InstanceDataset, instance_dataloader
from AttentionModel import GatedAttentionModel
from fold_split import stratified_k_fold_split, stratified_train_test_split


class Trainer:
    def __init__(self, criterion, batch_save, model, train_loader, val_loader, save_path, num_epochs=200, patience=20,
                 positional_embed=False):
        self.model = model
        self.positional_embed = positional_embed
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_save = batch_save
        self.criterion = criterion
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

    def _save_patient_data(self, loader, phase):
        # initializing patient data dir
        phase_path = self.paths[phase]
        print(f"Initializing {phase} directories")
        for bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id in loader:
            patient_dir = os.path.join(phase_path, patient_id[0])
            os.makedirs(patient_dir, exist_ok=True)
            patient_file = os.path.join(patient_dir, f"{patient_id[0]}.csv")
            temp = pd.DataFrame()

            x = np.array(x.squeeze(dim=0)).flatten()
            y = np.array(y.squeeze(dim=0)).flatten()

            # Check that tile_paths is a 1D array
            tile_paths = np.array(tile_paths).flatten()
            scales = np.repeat(scales, len(x))
            original_size = np.repeat(int(original_size), len(x))

            # Assign to the DataFrame
            temp["x"] = x
            temp["y"] = y
            temp["tile_paths"] = tile_paths
            temp["scale"] = scales
            temp["size"] = original_size

            # Save DataFrame to CSV
            temp.to_csv(patient_file, index = False)

    def _calculate_accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        return correct / len(labels)

    def train(self):
        # train loop
        self._save_patient_data(self.train_loader, "train")
        self._save_patient_data(self.val_loader, "val")
        print("Training")

        with open(self.training_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', "Weights Saved"])

            for epoch in range(self.num_epochs):
                train_loss, train_accuracy = self._train_epoch(epoch)
                val_loss, val_accuracy = self._validate_epoch(epoch)

                print(f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

                saved = "YES" if self._save_best_model(val_loss, epoch) else None
                if saved == "YES":
                    self.best_weights_path = os.path.join(self.paths["weights"], f"weights_epoch_{epoch}.pth")
                writer.writerow([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy, saved])

                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def _train_epoch(self, epoch):
        running_loss, running_correct, total, inst_count, train_inst_loss = 0.0, 0, 0, 0, 0

        for bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id in tqdm(self.train_loader,
                                                                                                  desc=f"Epoch {epoch + 1} [Train]"):
            self.model.train()
            if self.positional_embed:
                values = positional + bags
            else:
                values = bags
            if labels.dtype != torch.long:
                labels = labels.long()

            self.optimizer.zero_grad()
            logits, Y_prob, _, A_raw, results_dict, h = self.model(values, label=labels, instance_eval=True)
            labels_one_hot = nn.functional.one_hot(labels, num_classes=self.model.n_classes).float()
            logits = torch.sigmoid(logits)
            # now get the loss
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

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = running_correct / total
        return train_loss, train_accuracy

    def _validate_epoch(self, epoch):
        running_loss, running_correct, total = 0.0, 0, 0
        for bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id in tqdm(self.val_loader,desc=f"Epoch {epoch + 1} [Val]"):
            with torch.no_grad():
                self.model.eval()
                if self.positional_embed:
                    values = positional + bags
                else:
                    values = bags
                logits, Y_prob, _, A_raw, results_dict, h = self.model(values, label=labels)
                labels_one_hot = nn.functional.one_hot(labels, num_classes=self.model.n_classes).float()
                logits = torch.sigmoid(logits)
                loss = self.criterion(logits, labels_one_hot)

                running_loss += loss.item()
                running_correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                if epoch % self.batch_save == 0:
                    self._save_attention(epoch, A_raw, bags, positional, patient_id, h, phase="val")

        val_loss = running_loss / len(self.val_loader)
        val_accuracy = running_correct / total
        return val_loss, val_accuracy

    def _save_best_model(self, val_loss, epoch):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            torch.save(self.model.state_dict(), os.path.join(self.paths["weights"], f"weights_epoch_{epoch}.pth"))
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
                    values =  instance.unsqueeze(1)
                logits_inst, Y_prob_inst, Y_hat_inst, A_raw_inst, dict, h = self.model(values,
                                                                                  instance_eval=False)
                instance_labels.append(Y_hat_inst.item())
                instance_probs.append(Y_prob_inst[:, 1].item())
        return instance_labels, instance_probs

    def _save_attention(self, epoch, A_raw, bags, positional, patient_id, h, phase="train"):
        if phase == "train":
            patient_path = os.path.join(self.paths["train"],patient_id[0],f"{patient_id[0]}.csv")
            cluster_path = os.path.join(self.paths["train"],patient_id[0],f"{patient_id[0]}_cluster.h5")
        else:
            patient_path = os.path.join(self.paths["val"],patient_id[0],f"{patient_id[0]}.csv")
            cluster_path = os.path.join(self.paths["val"],patient_id[0],f"{patient_id[0]}_cluster.h5")
        patient_csv = pd.read_csv(patient_path)
        patient_csv[f"attention_{epoch + 1}"] = A_raw.squeeze(dim=1).squeeze().detach().numpy()
        instance_labels, instance_probs = self._instance_eval(bags, positional)
        patient_csv[f"instance_label_{epoch + 1}"] = np.array(instance_labels)
        patient_csv[f"instance_prob_{epoch + 1}"] = np.array(instance_probs)
        patient_csv.to_csv(patient_path, index = False)

        #  save cluster info
        with h5py.File(cluster_path, "a") as file:
            if f'cluster_epoch_{epoch}' in file.keys():
                del file[f'cluster_epoch_{epoch}']

            file.create_dataset(f'cluster_epoch_{epoch}', data=h.detach().numpy() )
# ----------------------------------------- EVALUATION ----------------------------------------------------------

def evaluate(model, dataloader, device="cpu", instance_eval=False):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for bags, positional, labels, x, y, tile_paths, scales, original_size, patient_id in dataloader:
            bags, labels = bags.to(device), labels.to(device)

            # get outputs
            outputs = model(bags)
            logits, Y_prob, Y_hat, A_raw, results_dict, h= outputs
            logits_max, indices = torch.max(logits, 1)
            all_probs.extend(Y_prob[:, 1].cpu().numpy())  # Probability of class 1
            all_preds.extend(Y_hat.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    # get metrics
    accuracy = accuracy_score(all_labels, all_preds)
    # TP/(TP+FP)
    precision = precision_score(all_labels, all_preds)
    # TP/(TP+FN)
    recall = recall_score(all_labels, all_preds)
    # TP/(TP+0.5(FP+FN))
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm
    }

    return metrics


# ------------------------------------------------------- MAIN -----------------------------------------------------------


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positional_embed", action="store_true")
    parser.add_argument("--testing_csv", default=None)
    parser.add_argument("--training_output", default = "/mnt/c/Users/loren/Masters/camelyon16_CLAM")
    parser.add_argument("--epochs", default=200,type=int)
    parser.add_argument("--k", default=20,type=int)
    parser.add_argument("--tile_selection", default="CLAM")
    parser.add_argument("--patience", default=20,type=int)
    parser.add_argument("--input_dim", default=2048,type=int)
    parser.add_argument("--hidden_dim1", default=512,type=int)
    parser.add_argument("--hidden_dim2", default=256,type=int)
    parser.add_argument("--metadata_path", default ="/mnt/c/Users/loren/Downloads/Camelyon16/preprocessing_results/patient_files.csv") #
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--fold_number", default=5,type=int)
    parser.add_argument("--dropout", default=0.0,type=float)
    parser.add_argument("--batch_save", default=5,type=int)
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
                                                  test_size=0.2,
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
            index.append(f"Folds {i}")
            # get the loaders
            train_loader = DataLoader(AttentionDataset(train_data.reset_index()), batch_size=1, shuffle=True)
            val_loader = DataLoader(AttentionDataset(val_data.reset_index()), batch_size=1)
            # initiate model
            instance_loss = cross_entropy_with_probs
            model = MIL_SB(instance_loss, input_dim=args.input_dim, hidden_dim1=args.hidden_dim1,
                           hidden_dim2=args.hidden_dim2, dropout_rate=args.dropout,
                           k=args.k, k_selection=args.tile_selection)
            # general criterion
            pos_num = len(train_data[train_data["target"] == 1]) / len(train_data)
            pos_weight = torch.tensor([1 / (1 - pos_num)])
            criterion = nn.BCELoss()
            print(f"percentage of pos {pos_num} weight {pos_weight}")

            save_path = os.path.join(args.training_output, f"fold_{i}")
            trainer = Trainer(criterion, args.batch_save, model, train_loader, val_loader, save_path, args.epochs,
                              args.patience, positional_embed=args.positional_embed)
            trainer.train()
            print(f"Evaluating fold {i}")
            best_weights = trainer.best_weights_path
            model.load_state_dict(torch.load(best_weights, weights_only=True))

            # now eval
            test_loader = DataLoader(AttentionDataset(test.reset_index()), batch_size=1)
            results = evaluate(model, test_loader, device="cpu", instance_eval=False)
            fold_metrics_testing.append(results)
            write_eval_crossval([fold_metrics_testing[i], fold_metrics_val[i]], os.path.join(save_path, "test-val_eval.csv"), ["Testing", "Validation", "Average"])

        print("Done with cross-val")

        index.append("Average")
        # get and save average performance for testing
        print("Testing:")
        write_eval_crossval(fold_metrics_testing, os.path.join(args.training_output, "testing_fold_evaluations.csv"), index)
        print("Validation:")
        write_eval_crossval(fold_metrics_val[i], os.path.join(args.training_output, "validation_fold_evaluations.csv"), index)

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
    train_loader = DataLoader(AttentionDataset(train.reset_index()), batch_size=1, shuffle=True)
    val_loader = DataLoader(AttentionDataset(val.reset_index()), batch_size=1)
    # initiate model
    instance_loss = cross_entropy_with_probs
    model = MIL_SB(instance_loss, input_dim=args.input_dim, hidden_dim1=args.hidden_dim1,
                   hidden_dim2=args.hidden_dim2, dropout_rate=args.dropout,
                   k=args.k, k_selection=args.tile_selection)
    # general criterion

    pos_num = len(train[train["target"] == 1]) / len(train)
    pos_weight = torch.tensor([1/(1 - pos_num)])
    criterion = nn.BCELoss()
    print(f"percentage of pos {pos_num} weight {pos_weight}")

    save_path = os.path.join(args.training_output, f"train-test-split")
    trainer = Trainer(criterion, args.batch_save, model, train_loader, val_loader, save_path, args.epochs,
                      args.patience, positional_embed=args.positional_embed)
    trainer.train()
    best_weights = trainer.best_weights_path
    model.load_state_dict(torch.load(best_weights, weights_only=True))
    # now eval
    test_loader = DataLoader(AttentionDataset(test.reset_index()), batch_size=1)
    results_test= evaluate(model, test_loader, device="cpu", instance_eval=False)
    results_val = evaluate(model, val_loader, device="cpu", instance_eval=False)

    write_eval_crossval([results_test, results_val], os.path.join(save_path, "test-val_eval.csv"), ["Testing", "Validation", "Average"])


if __name__ == "__main__":
    main()
