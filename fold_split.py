import numpy as np
import pandas as pd

def stratified_train_test_split(data, label_column, patient_id_column, cancer_subtype_column=None, test_size=0.2, random_state=None, split=True):

    if random_state is not None:
        np.random.seed(random_state)

    # If using pre-existing split info
    if not split:
        mask = data['split'].astype(str).str.contains("test", case=False, na=False)
        test = data[mask]
        train = data[~mask]
        return train, test

    test_patients = set()
    total_patients = data[patient_id_column].nunique()
    target_test_size = max(1, int(total_patients * test_size))  

    group_columns = [label_column] if cancer_subtype_column is None else [cancer_subtype_column, label_column]

    patient_groups = data.groupby([patient_id_column, label_column]).size().reset_index(name='count')

    grouped_data = patient_groups.groupby(group_columns)

    all_patients = data[patient_id_column].unique()
    np.random.shuffle(all_patients)

    selected_test_patients = set()

    for _, group in grouped_data:
        unique_patients = group[patient_id_column].unique()
        np.random.shuffle(unique_patients)  

        group_test_size = max(1, int(len(unique_patients) * test_size))
        selected_test_patients.update(unique_patients[:group_test_size])

        if len(selected_test_patients) >= target_test_size:
            break

    # Create train and test splits
    test_data = data[data[patient_id_column].isin(selected_test_patients)]
    train_data = data[~data[patient_id_column].isin(selected_test_patients)]

    return train_data, test_data
from sklearn.model_selection import StratifiedKFold

def stratified_k_fold_split(data, label_column, patient_id_column, cancer_subtype_column=None, n_splits=5, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    patient_labels = data.groupby(patient_id_column)[label_column].agg(lambda x: x.value_counts().idxmax()).reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_splits = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_labels[patient_id_column], patient_labels[label_column])):
        train_patients = patient_labels.iloc[train_idx][patient_id_column].values
        val_patients = patient_labels.iloc[val_idx][patient_id_column].values

        assert len(set(train_patients) & set(val_patients)) == 0, f"Patient leakage detected in fold {fold_idx}!"

        train_data = data[data[patient_id_column].isin(train_patients)]
        val_data = data[data[patient_id_column].isin(val_patients)]

        fold_splits.append((train_data, val_data))

    return fold_splits

