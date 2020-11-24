
# Example Data and Code in Matlab
# https://github.com/brain-networks


from pathlib import Path
import scipy.io as sio
import numpy as np

import eFC



# ==================================
# main
# ==================================
def load_data(fn_str):

	# load data
	mat_fn = Path(fn_str)
	mat_contents = sio.loadmat(mat_fn)

	return mat_contents['lab'], mat_contents['net'], mat_contents['ts']
	
	



# ==================================
# main
# ==================================
def main():

	# load the example data
	fn_str = './data/example_data.mat'
	cat, cat_label, ts = load_data(fn_str)
	
	
	# make data in the proper format
	bold_sig = ts                  # -> shape = (time, node ~ 1400x200)
	node_region = cat.flatten()    # -> shepe = (nodes ~ 200)
	region_name = np.array(cat_label.T[0].tolist()).flatten()  # -> shepe = (regions ~ 10)
	
	
	
	# declare eFC instance
	eFC_rest1 = eFC.eFC(fn_str, bold_sig, node_region, region_name)
	
	
	# generate eFC matrix and save (or load if exists)
	eFC_rest1.gen_eFC()
	
	
	# calculating Eigenvector
	eFC_rest1.gen_Eigen()
	
	
	# Plot t-SNE
	# eFC_rest1.plot_tsne()


	# calculate g_{ij}
	eFC_rest1.gen_gij()
	
	
	# calculate p_{ic}
	eFC_rest1.gen_pic()
	
	
	# calculate normalized entropy from p_{ic}
	df_ent = eFC_rest1.normalized_entropy()
	# print(df_ent)
	
	
	# calculate s_{ij}
	eFC_rest1.gen_sij()
	
	
	
	
	
	

# ==================================
# run
# ==================================	
if __name__ == '__main__':
	main()
	

