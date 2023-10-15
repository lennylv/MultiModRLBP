from RGCN.utils import available_pdbids
from RGCN.utils import graph_from_pdbid

pdbids = available_pdbids()
graphs = [graph_from_pdbid(p) for p in pdbids]
for i,G in enumerate(graphs):
    try:
        binding_nodes = [(n, d['binding_ion']) for n, d in G.nodes(data=True) if d['binding_ion'] is not None]
    except KeyError:
        print(pdbids[i])
# 2z75
# 7o80
# 7o7y
# 6ole
# 7o7z
# 2dlc
# 3k1v
# 6t4q
# 7jnh
# 4z31
# 3ovb
# 7o81
# 6r47
# 7qgg
# 1qbp
# buggy_pdb = '3k1v' # '2z75.B.49'

from RGCN.prepare_data.main import cif_to_graph

# from Bio.PDB.PDBList import PDBList
#
# pl = PDBList()
# res = pl.download_pdb_files([buggy_pdb], pdir='.')
# print(res)
# cif_to_graph(f"{buggy_pdb}.cif", output_dir='.')
# from utils.graph_io import load_json
# graph2 = load_json(f'./graphs/{buggy_pdb}.json')

# graph = graph_from_pdbid(buggy_pdb)
# pdbid, error_type, new_graph = cif_to_graph(f"{buggy_pdb}.cif", return_graph=True)
# # binding_nodes = [(n, d['binding_small-molecule']) for n, d in graph2.nodes(data=True) if
# #                  d['binding_small-molecule'] is not None]
# missing_new = [0 if 'binding_small-molecule' in d else 1 for n, d in new_graph.nodes(data=True)]
# print('missing nodes in new version : ', sum(missing_new))
# missing_old = [0 if 'binding_small-molecule' in d else 1 for n, d in graph.nodes(data=True)]
# print('missing nodes in new version : ', sum(missing_old))
# print(len(new_graph.nodes()))
# print(len(graph.nodes()))



