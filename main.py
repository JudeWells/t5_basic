import sys
import pickle
import os
import numpy as np
# from bio_embeddings.embed import ProtTransT5BFDEmbedder
import HuggingT5
import pandas as pd
from parse_chain_gcf import get_gcf_sequences_and_mapping

def pickler(embedding_dict, output_dir, chain_id):
	filename = output_dir + chain_id + '.pickle'
	with open(filename, 'wb') as filehandle:
		pickle.dump(embedding_dict, filehandle)


def unpickler(dir, chain_id):
	if '.pickle' not in chain_id:
		chain_id += '.pickle'
	with open(dir + chain_id, 'rb') as filehandle:
		return pickle.load(filehandle)

if __name__=="__main__":
	embedder = HuggingT5.HuggingT5()
	df = pd.read_csv('example_df.csv')
	output_dir = 't5_sequence_embeddings/'
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	try:
		# if running as an array job process one row based on integer passed as argument then quit
		args = sys.argv
		row_index = int(args[1]) - 1
	except:
		# if not in array-job mode iterate through whole dataset
		for i, row in df.iterrows():
			chain_id = row.PDBID + row.CHAIN
			if 'sequence' in df.columns:
				seq = row.sequence
			else:
				seq, pdb, mapping = get_gcf_sequences_and_mapping(chain_id)
			embed_chain = np.asarray(embedder(seq))
			embed_dict = {
				'sequence': seq,
				'sequence_embedding': embed_chain
			}
			pickler(embed_dict, output_dir, chain_id)