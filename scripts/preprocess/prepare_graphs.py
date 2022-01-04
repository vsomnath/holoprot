import os
import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from holoprot.utils import str2bool
from holoprot.data.handlers import PDBBind, Enzyme

HANDLERS = {'pdbbind': PDBBind, 'enzyme': Enzyme}
DATA_DIR = os.path.join(os.environ['PROT'], "datasets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Data Directory")
    parser.add_argument("--prot_mode", default='surface2backbone')
    parser.add_argument("--dataset", default='pdbbind', help="Dataset for which we prepare graphs.")
    parser.add_argument("--exp_name", default="PatchPointCloud_06-01-2021--20-20-00")
    parser.add_argument("--use_mp", default=True, type=str2bool)
    parser.add_argument("--pdb_ids", nargs="+", default=None)
    parser.add_argument("--n_segments", type=int, default=20, help="Number of patches ")
    args = parser.parse_args()

    handler_cls = HANDLERS.get(args.dataset)

    kwargs = {}
    if args.prot_mode in ['backbone2patch', 'patch2backbone']:
        kwargs['exp_name'] = args.exp_name
        kwargs['n_segments'] = args.n_segments

    handler = handler_cls(dataset=args.dataset, 
                          data_dir=args.data_dir, 
                          prot_mode=args.prot_mode,
                          use_mp=args.use_mp, **kwargs)
    handler.process_ids(pdb_ids=args.pdb_ids)

if __name__ == "__main__":
    main()
