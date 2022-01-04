import torch
import os
import argparse
from datetime import datetime as dt
from rdkit import RDLogger
import json
import wandb
import yaml

from holoprot.models.model_builder import build_model, MODEL_CLASSES
from holoprot.models import Trainer
from holoprot.utils import str2bool
from holoprot.data import DATASETS

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

try:
    ROOT_DIR = os.environ["PROT"]
    DATA_DIR = os.path.join(ROOT_DIR, "datasets")
    EXP_DIR = os.path.join(ROOT_DIR, "experiments")

except KeyError:
    ROOT_DIR = "./"
    DATA_DIR = os.path.join(ROOT_DIR, "datasets")
    EXP_DIR = os.path.join(ROOT_DIR, "local_experiments")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_exp_name(dataset, prot_mode):
    EXP_NAMES = {'pdbbind': {
        'backbone': 'BaseAffinity',
        'surface': 'SurfaceAffinity',
        'backbone2surface': 'SurfaceHierAffinity',
        'backbone2patch': 'HierAffinity',
        'surface2backbone': 'RevHierAffinity',
    }, 'reacbio': {
        'backbone': 'BaseAffinity',
        'surface': 'SurfaceAffinity',
        'backbone2surface': 'SurfaceHierAffinity',
        'backbone2patch': 'HierAffinity',
        'surface2backbone': 'RevHierAffinity',
    }, 'scope':
        {'backbone': 'ProtClassifier',
         'surface': 'SurfaceProtClassifier',
         'surface2backbone': 'RevHierProtClassifier',
         'backbone2patch': 'HierProtClassifier'
    }, 'enzyme':
        {'backbone': 'ProtClassifier',
         'surface': 'SurfaceProtClassifier',
         'surface2backbone': 'RevHierProtClassifier',
         'backbone2patch': 'HierProtClassifier'
    }}

    exp_dataset = EXP_NAMES.get(dataset)
    exp_name = exp_dataset.get(prot_mode, "")
    return exp_name


def run_model(config):
    # set default type
    if config['use_doubles']:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    # prepare model
    model = build_model(config, device=DEVICE)
    print(f"Converting model to device: {DEVICE}", flush=True)
    model.to(DEVICE)

    print("Param Count: ", sum([x.nelement() for x in model.parameters()]) / 10**6, "M", flush=True)
    print(flush=True)

    print(f"Device used: {DEVICE}", flush=True)

    dataset_class = DATASETS.get(config['dataset'])
    kwargs = {'split': config['split']}
    if config['dataset'] in ['scope', 'enzyme']:
        kwargs['add_target'] = True

    raw_dir = os.path.abspath(f"{config['data_dir']}/raw/{config['dataset']}")
    processed_dir = os.path.abspath(f"{config['data_dir']}/processed/{config['dataset']}")
    train_dataset = dataset_class(mode='train', raw_dir=raw_dir,
                                  processed_dir=processed_dir,
                                  prot_mode=config['prot_mode'], **kwargs)

    train_data = train_dataset.create_loader(batch_size=config['batch_size'], num_workers=config['num_workers'],
                                             shuffle=True)
    eval_data = None

    if config['eval_every'] is not None:
        eval_dataset = dataset_class(mode='valid', raw_dir=raw_dir,
                                     processed_dir=processed_dir,
                                     prot_mode=config['prot_mode'], **kwargs)
        eval_data = eval_dataset.create_loader(batch_size=1,
                                               num_workers=config['num_workers'])

    trainer = Trainer(model=model, dataset=config['dataset'], task_type=config['prot_mode'],
                      print_every=config['print_every'], eval_every=config['eval_every'])
    trainer.build_optimizer(learning_rate=config['lr'])
    trainer.build_scheduler(type=config['scheduler_type'], step_after=config['step_after'],
                            anneal_rate=config['anneal_rate'], patience=config['patience'],
                            thresh=config['metric_thresh'])
    trainer.train_epochs(train_data, eval_data, config['epochs'],
                         **{"accum_every": config['accum_every'],
                            "clip_norm": config['clip_norm']})


def main(args):
    # initialize wandb
    wandb.init(project='exp_holoprot', dir=args.out_dir,
               entity='msprot',
               config=args.config_file)
    config = wandb.config
    tmp_dict = vars(args)
    for key, value in tmp_dict.items():
        config[key] = value

    print(config)
    # start run
    run_model(config)


def sweep(args):
    # load config
    with open(args.config_file) as file:
        default_config = yaml.load(file, Loader=yaml.FullLoader)

    loaded_config = {}
    for key in default_config:
        loaded_config[key] = default_config[key]['value']

    # init wandb
    wandb.init(allow_val_change=True, dir=args.out_dir)

    # update wandb config
    wandb.config.update(loaded_config)
    config = wandb.config
    tmp_dict = vars(args)
    for key, value in tmp_dict.items():
        config[key] = value

    print(config)
    # start run
    run_model(config)


if __name__ == "__main__":

    def get_argparse():
        parser = argparse.ArgumentParser()

        # Logging and setup args
        parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory")
        parser.add_argument("--out_dir", default=EXP_DIR, help="Experiments directory")
        parser.add_argument("--config_file", default=None)
        parser.add_argument("--sweep", action='store_true')

        args = parser.parse_args()
        return args

    # get args
    args = get_argparse()

    if args.sweep:
        # create sweep directory
        sweep(args)
    else:
        main(args)
