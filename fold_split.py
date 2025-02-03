import pandas as pd
import numpy as np

def stratified_train_test_split(data, label_column, patient_id_column, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # group data by patient ID and aggregate labels -> ensures no patient accross samples
    grouped = data.groupby(patient_id_column)[label_column].agg(pd.Series.mode).reset_index()
    label_0 = grouped[grouped[label_column] == 0]
    label_1 = grouped[grouped[label_column] == 1]

    # calculate the number of test samples needed for each label
    num_test_label_0 = int(len(label_0) * test_size)
    num_test_label_1 = int(len(label_1) * test_size)

    # shuffle and select patients for the test set
    test_patients_label_0 = label_0.sample(n=num_test_label_0, random_state=random_state)
    test_patients_label_1 = label_1.sample(n=num_test_label_1, random_state=random_state)

    # combine test patients and determine the train patients
    test_patients = pd.concat([test_patients_label_0, test_patients_label_1])
    train_patients = grouped[~grouped[patient_id_column].isin(test_patients[patient_id_column])]

    # filter the original data to create train and test sets
    train_data = data[data[patient_id_column].isin(train_patients[patient_id_column])]
    test_data = data[data[patient_id_column].isin(test_patients[patient_id_column])]

    return train_data, test_data


# for cross validation
def stratified_k_fold_split(data, label_column, patient_id_column, n_splits=5, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    print (data[patient_id_column].value_counts())

    # Group data by patient ID and aggregate labels
    grouped = data.groupby(patient_id_column)[label_column].agg(pd.Series.mode).reset_index()
    label_0_patients = grouped[grouped[label_column] == 0]
    label_1_patients = grouped[grouped[label_column] == 1]

    label_0_patients = label_0_patients.sample(frac=1, random_state=random_state).reset_index(drop=True)
    label_1_patients = label_1_patients.sample(frac=1, random_state=random_state).reset_index(drop=True)

    label_0_folds = np.array_split(label_0_patients, n_splits)
    label_1_folds = np.array_split(label_1_patients, n_splits)
    folds = []
    for i in range(n_splits):
        # Create validation set for this fold
        val_patients = pd.concat([label_0_folds[i], label_1_folds[i]])
        val_data = data[data[patient_id_column].isin(val_patients[patient_id_column])]

        # Create training set for this fold
        train_patients = pd.concat([label_0_patients, label_1_patients]).drop(index=val_patients.index)
        train_data = data[data[patient_id_column].isin(train_patients[patient_id_column])]

        # for training data if there is an imbalance re sample

        folds.append((train_data, val_data))

    return folds
