import gemmi
import json
import numpy as np
import subprocess

xyzin = 'data/Q5VSL9_AF_model.pdb'
pae_json = 'data/Q5VSL9_AF_model.json'
struct = gemmi.read_structure(xyzin)
clustering_method = 'pae'

# Take data from models and write to input.json
molecule_type = "protein"
input_data = {}
array = []
struct.setup_entities()
struct.remove_ligands_and_waters()
count = 0
for model in struct:
    for chain in model:
        for residue in chain:
            if molecule_type in ["protein", "mixed"]:
                atom = residue.get_ca()
            elif molecule_type in ["na", "mixed"]:
                atom = residue.get_p()
            if atom:
                input_data[count] = {"x": atom.pos.x, "y": atom.pos.y, "z": atom.pos.z}
                array.append([atom.pos.x, atom.pos.y, atom.pos.z])
                count += 1

with open('data/input.json', 'w') as f:
    json.dump(input_data, f)


# Run slice
exe_path = '../bin/slice'
subprocess.call([exe_path, '--input_json', 'data/input.json', '--clustering_method', clustering_method, '--nclusters', '4', '--input_pae', pae_json, '--output_json', 'data/output.json'])

# Read output.json
with open('data/output.json', 'r') as f:
    output_data = json.load(f)

# Process the output data
labels = np.array([])
for dictionary in output_data:
    for key, value in dictionary.items():
        labels = np.append(labels, value)

def output_split_model(xyzin, xyzout, seqid_to_keep):
    """Function to take part of xyzin by index and output it as xyzout

    Args:
        xyzin (str): Path to input XYZ file
        xyzout (str): Path to output XYZ file
        seqid_to_keep (list): List of seqids in the form ['model:chain:seqid']

    Returns:
        file: Output XYZ file
    """

    struct = gemmi.read_structure(str(xyzin))
    struct.setup_entities()
    struct.remove_ligands_and_waters()
    count = 0
    for model in struct:
        for chain in model:
            to_remove = []
            for idx, residue in enumerate(chain):
                if (idx not in seqid_to_keep):
                    to_remove.append(idx)
                count += 1

            for i in to_remove[::-1]:
                del chain[i]
    
    if all([model.count_atom_sites() > 0 for model in struct]):
        with open(xyzout, "w") as f_out:
            f_out.write(struct.make_minimal_pdb())

for x in set(labels):
    if x == -1:
        output_pdb = "outliers.pdb"
    else:
        output_pdb = f"cluster_{x}.pdb"

    res_idx_to_keep = np.where(labels == x)[0].tolist()
    seqid_to_keep = [list(input_data.keys())[x] for x in res_idx_to_keep] # should be the same in this example but different if multiple chains/models

    if len(seqid_to_keep) > 0:
        output_split_model(xyzin, output_pdb, seqid_to_keep)
