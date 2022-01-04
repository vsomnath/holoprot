import json
import os
import argparse
import torch
import numpy as np
import wandb
import yaml
from scipy import stats

from holoprot.data import DATASETS
from holoprot.models.model_builder import MODEL_CLASSES
from holoprot.utils.metrics import DATASET_METRICS, METRICS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(os.environ['PROT'], "datasets")
EXP_DIR = os.path.join(os.environ['PROT'], "experiments")


def get_model_class(dataset):
    model_class = MODEL_CLASSES.get(dataset)
    return model_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--exp_dir", default=EXP_DIR)
    parser.add_argument("--exp_name", nargs="+")
    args = parser.parse_args()

    metrics_all = {}

    for exp_name in args.exp_name:
        if "run" in exp_name:
            # wandb specific loading
            loaded = torch.load(f"{args.exp_dir}/wandb/{exp_name}/files/best_ckpt.pt", map_location=DEVICE)

            with open(f"{args.exp_dir}/wandb/{exp_name}/files/config.yaml", "r") as f:
                loaded_train_config = yaml.load(f, Loader=yaml.FullLoader)

            train_args = {}
            for key in loaded_train_config:
                if isinstance(loaded_train_config[key], dict):
                    if 'value' in loaded_train_config[key]:
                        train_args[key] = loaded_train_config[key]['value']

        else:
            loaded = torch.load(os.path.join(args.exp_dir, exp_name,
                                "checkpoints", "best_ckpt.pt"), map_location=DEVICE)

            with open(f"{args.exp_dir}/{exp_name}/args.json", "r") as f:
                train_args = json.load(f)

        dataset = train_args['dataset']
        dataset_class = DATASETS.get(dataset)
        prot_mode = train_args['prot_mode']
        split = train_args['split']
        num_workers = train_args['num_workers']

        raw_dir = f"{args.data_dir}/raw/{dataset}"
        processed_dir = f"{args.data_dir}/processed/{dataset}"

        config = loaded['saveables']
        model_class = get_model_class(dataset)
        model = model_class(**config, device=DEVICE)
        model.load_state_dict(loaded['state'])
        model.to(DEVICE)
        model.eval()

        if dataset == 'enzyme':
            metric = DATASET_METRICS.get('enzyme')
            metric_fn, _, _ = METRICS.get(metric)
            kwargs = {'add_target': True}

            test_dataset = dataset_class(mode='test', raw_dir=raw_dir,
                                         processed_dir=processed_dir,
                                         prot_mode=prot_mode, **kwargs)

            test_loader = test_dataset.create_loader(batch_size=1, num_workers=num_workers)

            y_pred = []
            y_true = []

            for idx, inputs in enumerate(test_loader):
                if inputs is None:
                    y_pred.append(np.nan)
                    y_true.append(np.nan)
                else:
                    label_pred = model.predict(inputs).item()
                    label_true = inputs.y.item()
                    y_true.append(label_true)
                    y_pred.append(label_pred)

            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()

            acc_score = metric_fn(y_true, y_pred)

            if metric in metrics_all:
                metrics_all[metric].append(acc_score)
            else:
                metrics_all[metric] = [acc_score]

            msg = f"Test {metric}: {np.round(acc_score, 4)}"
            print(msg, flush=True)

        else:
            base_dataset = dataset_class(mode='test', raw_dir=raw_dir,
                                         processed_dir=processed_dir,
                                         prot_mode=prot_mode, split=split)
            data_loader = base_dataset.create_loader(batch_size=1, num_workers=num_workers)

            activity_true_all = []
            activity_pred_all = []

            for idx, inputs in enumerate(data_loader):
                if inputs is None:
                    continue
                else:
                    activity_pred = model.predict(inputs).item()
                    activity_true = inputs.y.item()
                    activity_true_all.append(activity_true)
                    activity_pred_all.append(activity_pred)

            activity_true_all = np.array(activity_true_all).flatten()
            activity_pred_all = np.array(activity_pred_all).flatten()

            metrics = DATASET_METRICS.get(dataset)
            print_msgs = []

            for metric in metrics:
                metric_fn, _, _ = METRICS.get(metric)
                metric_val = metric_fn(activity_true_all, activity_pred_all)
                if metric not in metrics_all:
                    metrics_all[metric] = [metric_val]
                else:
                    metrics_all[metric].append(metric_val)
                print_msgs.append(f"{metric}: {np.round(metric_val, 4)}")

            print_msg = ", ".join(msg for msg in print_msgs)
            print(print_msg, flush=True)

    final_metrics = {}
    for metric, metric_vals in metrics_all.items():
        mean = np.mean(metric_vals)
        std = np.std(metric_vals)
        final_metrics[metric] = (np.round(mean, 4), np.round(std, 4))

    print(f"Final Metrics: {final_metrics}", flush=True)

if __name__ == "__main__":
    main()
