import argparse
import os
import numpy as np
import networkx as nx
import torch

DATA_DIR = os.path.join(os.environ["PROT"], "datasets")


def ers_segment(args):
    from superpixel.ers import ERS
    exp_name = "ERS"
    exp_name += f"_balance={args.balance_fac}_n_segments={args.n_segments}"

    assignments_dir = f"{args.data_dir}/assignments/{args.dataset}/{exp_name}"
    os.makedirs(assignments_dir, exist_ok=True)
    processed_dir = f"{args.data_dir}/processed/{args.dataset}"
    pdb_files = os.listdir(f"{processed_dir}/surface")

    for pdb_file in pdb_files:
        pdb_id = pdb_file.split(".")[0]
        fullpath = f"{processed_dir}/surface/{pdb_file}"
        target_dict = torch.load(fullpath)

        if 'prot' not in target_dict:
            print(f"prot not found for {pdb_id}")
            continue
        data = target_dict['prot']

        model = ERS(args.balance_fac)
        G = nx.Graph()
        n = len(data.pos)
        G.add_nodes_from(np.arange(n))

        # Extract edges from triangular faces of mesh
        f = np.array(data.face.numpy(), dtype = int)
        rowi = np.concatenate([f[:,0], f[:,0], f[:,1], f[:,1], f[:,2], f[:,2]], axis = 0)
        rowj = np.concatenate([f[:,1], f[:,2], f[:,0], f[:,2], f[:,0], f[:,1]], axis = 0)
        edges = np.stack([rowi, rowj]).T

        G.add_edges_from(edges)
        edge_list = np.array(list(G.edges()), dtype=np.int)

        feats = data.x.numpy()
        edge_sim = np.sum(feats[edge_list[:, 0]] * feats[edge_list[:, 1]], axis=1)
        edge_sim = edge_sim.astype(np.double)
        
        patch_labels = model.computeSegmentation(edge_list, edge_sim, len(G.nodes),
                                                args.n_segments)
        if patch_labels is None:
            print(f"Assignments for {pdb_id} could not be generated. Edges: {len(G.edges())}, Nodes: {len(G.nodes())}", flush=True)

        assert len(patch_labels) == len(data.pos)
        if len(np.unique(patch_labels)) > 1:
            torch.save(patch_labels, f"{assignments_dir}/{pdb_id}.pth")
            #print(f"Saved assignments for {pdb_id}, Edges: {len(G.edges())}, Nodes: {len(G.nodes())}", flush=True)
        else:
            print(f"No unique labels for {pdb_id}. Edges: {len(G.edges())}, Nodes: {len(G.nodes())}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=DATA_DIR, type=str)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--ckpt_id", type=str, default=None)
    parser.add_argument("--seg_mode", default='ers', help="Which function to use for segmentation")
    parser.add_argument("--dataset", default="pdbbind")
    parser.add_argument("--balance_fac", type=float, default=0.5)
    parser.add_argument("--n_segments", type=int, default=20)
    args = parser.parse_args()

    SEG_FNS = {'ers': ers_segment}
    segmentation_fn = SEG_FNS.get(args.seg_mode)
    segmentation_fn(args)

if __name__ == "__main__":
    main()
