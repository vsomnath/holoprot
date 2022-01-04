import os
import torch
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Union
import traceback
import multiprocessing
import sys
from rdkit import Chem
import signal
import pickle
import functools
import traceback

from holoprot.graphs import (Backbone, Surface,
    Surface2Backbone, Patch2Backbone, Complex)


Builder = Union[Backbone, Surface, Surface2Backbone, Patch2Backbone, Complex]


PROT_BUILDERS = {'backbone': Backbone, 
    'surface': Surface, 
    'patch2backbone': Patch2Backbone,
    'surface2backbone': Surface2Backbone,
}


def get_builder_from_config(config: Dict, prot_mode: str) -> Builder:
    builder_class = PROT_BUILDERS.get(prot_mode, None)
    if builder_class is not None:
        builder_fn = builder_class(**config)
        return builder_fn
    else:
        raise ValueError(f"Graph type {prot_mode} not supported.")


def process_id_callback(handler, pdb_id, **kwargs):
    return handler.process_id(pdb_id, **kwargs)


class DataHandler:

    def __init__(self, 
                 dataset: str,
                 prot_mode: str, 
                 data_dir: str = None,
                 use_mp: bool = True,
                 **kwargs):
        self.dataset = dataset
        self.prot_mode = prot_mode
        self.data_dir = data_dir
        self.use_mp = use_mp
        
        self.builder_fn = self._load_builder()
        self._setup_directories(**kwargs)
        self.load_ids()
        self.__dict__.update(**kwargs)

    def _setup_directories(self, **kwargs):
        self.base_dir = f"{self.data_dir}/raw/{self.dataset}"
        self.save_dir = f"{self.data_dir}/processed/{self.dataset}/{self.prot_mode}"
        if 'n_segments' in kwargs:
            n_segments = kwargs['n_segments']
            self.save_dir += f"_n_segments={n_segments}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _load_builder(self):
        with open(f"configs/preprocess/{self.dataset}/{self.prot_mode}.json", "r") as f:
            config = json.load(f)
        builder_fn = get_builder_from_config(config, prot_mode=self.prot_mode)
        if self.dataset in ['pdbbind', 'reacbio']:
            builder_fn = Complex(prot_builder=builder_fn)
        return builder_fn

    def load_ids(self):
        raise NotImplementedError('Subclasses must implement for themselves.')

    @staticmethod
    def run_builder(builder_fn, *args, **kwargs):
        return builder_fn(*args, **kwargs)

    def process_ids(self, pdb_ids: List[str] = None):
        if pdb_ids is None:
            pdb_ids = self.pdb_ids
        if self.use_mp:
            pool = multiprocessing.Pool(multiprocessing.cpu_count() // 5, maxtasksperchild=1)
            results = pool.imap_unordered(functools.partial(process_id_callback, self), pdb_ids, chunksize=500)
            pool.close()

            for ind, target in enumerate(results):
                try:
                    if target is not None:
                        save_file = f"{self.save_dir}/{target['pdb_id'].upper()}.pth"
                        torch.save(target, save_file)
                        print(f"{target['pdb_id']} processed.", flush=True, file=sys.stdout)
                except Exception as e:
                    continue
        else:
            def handler(signum, frame):
                raise Exception("PDB file could not be processed within given time.")

            for pdb_id in pdb_ids:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(300)

                try:
                    target = self.process_id(pdb_id)
                    if target is not None:
                        save_file = f"{self.save_dir}/{target['pdb_id'].upper()}.pth"
                        torch.save(target, save_file)
                        print(f"{target['pdb_id']} processed.", flush=True, file=sys.stdout)
                except Exception as e:
                    print(f"{pdb_id}: {e}")
                    traceback.print_exc()
                    signal.alarm(0)
                    continue

    def process_id(self, pdb_id: str):
        if os.path.exists(f"{self.save_dir}/{pdb_id.upper()}.pth"):
            target = torch.load(f"{self.save_dir}/{pdb_id.upper()}.pth", map_location='cpu')
            return target
        
        orig_file = f"{self.base_dir}/pdb_files/{pdb_id}/{pdb_id}.pdb"
        fixed_file = f"{self.base_dir}/pdb_files/{pdb_id}/{pdb_id}_fixed.pdb"

        def run_builder(pdb_file):
            target = {}
            target['pdb_id'] = pdb_id
            base_inputs = (self.builder_fn, pdb_file)
            kwargs = {}

            if self.prot_mode in ['surface', 'patch2backbone', 'surface2backbone']:
                mesh_file = f"{self.data_dir}/raw/{self.dataset}/pdb_files/{pdb_id}/{pdb_id}.obj"
                kwargs['mesh_file'] = mesh_file

                if self.prot_mode == 'patch2backbone':
                    assign_file = f"{self.data_dir}/assignments/{self.dataset}/{self.exp_name}/{pdb_id.upper()}.pth"
                    kwargs['assignments_file'] = assign_file

            if self.dataset == "pdbbind":
                lig_smi = self.lig_smiles[f"{pdb_id}_ligand"]
                activity = self.affinity_dict[pdb_id]
                base_inputs += (lig_smi, float(activity))
                kwargs['build_prot'] = True

            elif self.dataset == "reacbio":
                kwargs['build_prot'] = True
                for lig_id in self.lig_ids:
                    if lig_id not in self.cas_to_smiles:
                        continue
                    lig_smi = self.cas_to_smiles[lig_id]
                    lig_mol = Chem.MolFromSmiles(lig_smi)
                    if lig_mol is None:
                        continue
                    activity = self.affinity_df.loc[pdb_id, lig_id]
                    if np.isnan(activity):
                        continue

                    inputs = base_inputs + (lig_smi, float(activity))
                    if 'prot' in target:
                        kwargs['build_prot'] = False
                    graph = DataHandler.run_builder(*inputs, **kwargs)

                    if graph is not None:
                        if 'prot' not in target:
                            target['prot'], target[lig_id] = graph
                        else:
                            target[lig_id] = graph
                        print(f"{pdb_id}, {lig_id} processed.", flush=True)
                return target

            graph = DataHandler.run_builder(*base_inputs, **kwargs)
            if graph is not None:
                if isinstance(graph, tuple):
                    protein, ligand = graph
                    target['prot'] = protein
                    target[f"{pdb_id}_ligand"] = ligand
                else:
                    target['prot'] = graph
                return target

        try:
            return run_builder(pdb_file=orig_file)
        except Exception as e:
            msg = f"Failed to generate graph with base file due to {e}. Trying with fixed pdb file."
            print(f"{pdb_id}: {msg}", flush=True)
            traceback.print_exc()
            pass
        
        try:
            return run_builder(pdb_file=fixed_file)
        except Exception as e:
            msg = f"Failed to generate graph with fixed pdb file due to {e}. Returning None."
            print(f"{pdb_id}: {msg}", flush=True)
            traceback.print_exc()
            return None


class PDBBind(DataHandler):

    def load_ids(self):
        with open(os.path.join(self.base_dir, "metadata/affinities.json"), "r") as f:
            self.affinity_dict = json.load(f)
        with open(os.path.join(self.base_dir, "metadata/lig_smiles.json"), "r") as f:
            self.lig_smiles = json.load(f)
        self.pdb_ids = list(self.affinity_dict.keys())


class Enzyme(DataHandler):

    def load_ids(self):
        with open(f"{self.base_dir}/metadata/function_labels.json", "r") as f:
            labels_dict = json.load(f)
        self.pdb_ids = list(labels_dict.keys())