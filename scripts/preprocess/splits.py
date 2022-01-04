import numpy as np
import random
import copy


def generate_scaffold(mol, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string.
  Note
  ----
  This function requires `rdkit` to be installed.
  """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)
    return scaffold_smiles


def scaffold_split(dataset,
                   seed: int = None,
                   frac_valid: float = 0.1,
                   frac_test: float = 0.1,
                   **kwargs):
    """Scaffold split for dataset.

    Parameters
    ----------
    dataset:
        Blah blah
    seed: int (default None),
        Seed to ensure reproducibility in splits
    frac_valid: float (default 0.1),
        Valid fraction
    frac_test: float (default 0.1),
        Test fraction
    """
    frac_train = 1 - frac_valid - frac_test
    scaffolds = {}

    for id, mol in dataset.items():
        try:
            scaffold = generate_scaffold(mol)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [id]
            else:
                scaffolds[scaffold].append(id)
        except Exception as e:
            print(e, flush=True)
            continue

    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    train_cutoff = int(frac_train * len(dataset))
    valid_cutoff = int((frac_train + frac_valid) * len(dataset))
    train_ids, valid_ids, test_ids = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_ids) + len(scaffold_set) > train_cutoff:
            if len(train_ids) + len(scaffold_set) + len(
                    valid_ids) > valid_cutoff:
                test_ids += scaffold_set
            else:
                valid_ids += scaffold_set
        else:
            train_ids += scaffold_set
    print(f"Train: {len(train_ids)}, Valid: {len(valid_ids)}, Test: {len(test_ids)}", flush=True)
    return train_ids, valid_ids, test_ids


def random_split(dataset,
                 seed: int = None,
                 frac_valid: float = 0.1,
                 frac_test: float = 0.1,
                 **kwargs):
    """Random split for dataset.

    Parameters
    ----------
    dataset:
        Blah blah
    seed: int (default None),
        Seed to ensure reproducibility in splits
    frac_valid: float (default 0.1),
        Valid fraction
    frac_test: float (default 0.1),
        Test fraction
    """
    ids_copy = list(dataset.keys())
    frac_train = 1 - frac_valid - frac_test
    train_cutoff = int(frac_train * len(dataset))
    valid_cutoff = int((frac_train + frac_valid) * len(dataset))
    random.shuffle(ids_copy)

    train_ids = ids_copy[:train_cutoff]
    valid_ids = ids_copy[train_cutoff:valid_cutoff]
    test_ids = ids_copy[valid_cutoff:]
    print(f"Train: {len(train_ids)}, Valid: {len(valid_ids)}, Test: {len(test_ids)}", flush=True)
    return train_ids, valid_ids, test_ids


def stratified_split(dataset,
                     seed: int = None,
                     frac_valid: float = 0.1,
                     frac_test: float = 0.1):
    """Stratified split for dataset.

    Parameters
    ----------
    dataset:
        Blah blah
    seed: int (default None),
        Seed to ensure reproducibility in splits
    frac_valid: float (default 0.1),
        Valid fraction
    frac_test: float (default 0.1),
        Test fraction
    """
    frac_train = 1 - frac_valid - frac_test
    activities = list(dataset.values())
    sort_idxs = np.argsort(activities)

    train_ids, valid_ids, test_ids = [], [], []
    sorted_ids = np.array(list(dataset.keys()))[sort_idxs]
    sorted_ids = sorted_ids.tolist()

    cutoff = 10
    train_cutoff = int(frac_train * cutoff)
    valid_cutoff = int((frac_train + frac_valid) * cutoff)

    start_idxs = np.arange(0, len(sorted_ids), cutoff)
    for idx in start_idxs:
        ids_batch = sorted_ids[idx:idx + cutoff].copy()
        random.shuffle(ids_batch)

        train_cutoff = int(frac_train * len(ids_batch))
        valid_cutoff = int((frac_train + frac_valid) * len(ids_batch))

        train_ids.extend(ids_batch[:train_cutoff])
        valid_ids.extend(ids_batch[train_cutoff: valid_cutoff])
        test_ids.extend(ids_batch[valid_cutoff:])

    print(f"Train: {len(train_ids)}, Valid: {len(valid_ids)}, Test: {len(test_ids)}", flush=True)
    return train_ids, valid_ids, test_ids


def agglomerative_split(dataset,
                        seed: int = None,
                        frac_valid: float = 0.1,
                        frac_test: float = 0.1):
    pass
