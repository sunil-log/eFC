
from pathlib import Path

from scipy import stats
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw 		


# ===========================
# Class
# ===========================
class eFC:

	# =====================================
	# Constructor (parameters)
	# =====================================
	def __init__(self, fn_str, bold_sig, node_region, region_name):	
		
		# -------------------------
		# bold_sig.shape = (time, node ~ 1400x200)
		# node_region.shape = (nodes ~ 200)
		# region_name.shape = (regions ~ 10)
		# -------------------------
	
		self.bold_sig = bold_sig
		self.node_region = node_region
		self.region_name = region_name
		
		
		# where intermediate files will be stored
		self.fn_str = fn_str
		self.fn_path = Path(self.fn_str)
		self.fn_pure = self.fn_path.parent.absolute() / self.fn_path.stem
		
		self.fn_eFC = Path(  str(self.fn_pure ) + "_eFC.npy"  )
		self.fn_eFC_fig = Path(  str(self.fn_pure ) + "_eFC_fig.png"  )
		
		self.fn_eigen_vec = Path(  str(self.fn_pure ) + "_eigh_vec.npy"  ) 
		self.fn_eigen_val = Path(  str(self.fn_pure ) + "_eigh_val.npy"  )
		
		self.fn_tsne = Path(  str(self.fn_pure ) + "_tsne.png"  )
		self.fn_gij = Path(  str(self.fn_pure ) + "_gij.cvs"  )
		
		self.fn_pic_fig = Path(  str(self.fn_pure ) + "_pic.png"  )
		self.fn_pic_table = Path(  str(self.fn_pure ) + "_pic.csv"  )
		
		
		
		
	# =====================================
	# Constructor (parameters)
	# =====================================		
	def gen_eFC(self):
	
		# --------
		# if eFC matrix exists
		# --------
		if self.fn_eFC.exists():
			print('> Previously calculated eFC matrix exists')
			# self.eFC = np.load(self.fn_eFC)



		# --------
		# if not
		# --------
		else:
			print('> Calculating eFC matrix from BOLD signal')

			# length of each dimensions
			n_t, n_ch = self.bold_sig.shape
			
			# making ts mean=0, sd=1
			z = stats.zscore(self.bold_sig)	
			
			# calculating c_ij = [ (z_i(1) z_j(1)), (z_i(2) z_j(2)), ...  ]
			c = np.einsum('ti,tj->tij',z,z)

			
			# take only upper triangle and make it linear
			tri_idx = np.triu_indices(n_ch, k=1)   # k: offset, k=1 -> remove diagonal
			c = np.transpose(c, (1, 2, 0))     # -> (200, 200, 1200)
			c = c[tri_idx]                     # -> (20000, 1200)
			c = c.T                            # -> (1200, 20000)
			
			
			# sum_t c_{i*}(t) c_{u*}(t)   -> numer.shape=(40000, 40000);    (numerator)
			numer = np.einsum('ti,tu->iu',c,c)	
			
		
			# g_{i*} = sqrt[ sum_t  c_{i*}^2  ] -> g.shape = (40000) ;      (part of denominator)
			g = np.sqrt(  np.einsum('ti,ti->i',c,c)  )
			
			
			# g_{i*} x g_{u*}  ->  denom.shape=(40000, 40000);              (denominator)
			denom = np.einsum('i,u->iu',g,g) 
			
			
			# eFC = numer/denom  -> eFC.shape=(40000, 40000)
			eFC = numer/denom
			
			# save file
			# now_int = int(datetime.now().timestamp())
			print("Saving eFC file")
			np.save(self.fn_eFC, eFC)

			
			# plot figure
			fig, ax = plt.subplots(figsize=(12, 12))
			ax.imshow(eFC, interpolation='nearest', aspect='auto', cmap=plt.cm.get_cmap('seismic'))
			plt.savefig(self.fn_eFC_fig)
			plt.close()

				
	
			
	# =====================================
	# Constructor (parameters)
	# =====================================		
	def gen_Eigen(self):
		
		
		# ERROR, if there is no eFC matrix
		if not( self.fn_eFC.exists() ):	
			print('> ERROR: No eFC matrix found')
			exit()		
	
		
		# --------
		# if Eigenvector matrix exists
		# --------
		if self.fn_eigen_vec.exists():
			print('> Previously calculated Eigenvector exists')
					

		# --------
		# if not -> calculate Eigenvector
		# --------
		else:
		
			# load matrix
			print('> Load previously calculated eFC matrix')
			A = np.load(self.fn_eFC)
			
			# calculate eigenvalues and eigenvectors 
			print('> Calculating Eigenvector and Eigenvalues')	
			vals, vecs = np.linalg.eigh(A)
			
			# sort these based on the eigenvalues
			vecs = vecs[:,np.argsort(vals)]
			vals = vals[np.argsort(vals)]
			
			# save file
			print("Saving Eigenvector files")
			np.save(self.fn_eigen_vec, vecs)
			np.save(self.fn_eigen_val, vals)

		
	# =====================================
	# Constructor (parameters)
	# =====================================		
	def plot_tsne(self, n_eigen=50, n_clusters=10):
	
		# ERROR, if there is no eFC matrix
		if not( self.fn_eigen_vec.exists() ):	
			print('> ERROR: No Eigenvector matrix found')
			exit()			



		# load Eigenvector matrix
		print('> Load previously calculated Eigenvector matrix')
		Q = np.load(self.fn_eigen_vec)		
		
		# normalize column 
		col_max = np.max(np.abs(Q), axis=0)    # col_max.shape = (20000)
		Q = Q/col_max                          # div each columns by max 		
				
		# choose high contributing columns
		Qx = Q.T[-n_eigen:]  
		Qx = Qx.T          # Qx.shape = (20000, n_eigen)		
	
	
	
		# ---------------------------
		# CLUSTERING
		# ---------------------------
			
		
		# TSNE for display
		from sklearn.manifold import TSNE
		TSNE_model = TSNE(n_components=2)
		X_tsne = TSNE_model.fit_transform(Qx)
		df = pd.DataFrame(X_tsne, columns=['x', 'y'])
		
		
		# k-means for color
		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=n_clusters)
		kmeans.fit(Qx)
		labels = kmeans.predict(Qx)
		
		
		# plot (could make ERROR; if n_clusters \neq tab10)
		plt.close()
		fig, ax = plt.subplots(1, figsize=(10,10))
		ax.scatter(df['x'], df['y'], s=0.1, c=labels, cmap=plt.get_cmap('tab10'))
		plt.savefig(self.fn_tsne)
		



	# =====================================
	# Constructor (parameters)
	# =====================================		
	def gen_gij(self, n_eigen=50, n_clusters=10):
	
		# ERROR, if there is no eFC matrix
		if not( self.fn_eigen_vec.exists() ):	
			print('> ERROR: No Eigenvector matrix found')
			exit()	
			
			
		# --------
		# if gij matrix exists
		# --------
		if self.fn_gij.exists():
			print('> Previously calculated g_ij exists')
					

		# --------
		# if not -> calculate gij
		# --------
		else:
			

			# load Eigenvector matrix
			print('> Load previously calculated Eigenvector matrix')
			Q = np.load(self.fn_eigen_vec)		
			
			# normalize column 
			col_max = np.max(np.abs(Q), axis=0)    # col_max.shape = (20000)
			Q = Q/col_max                          # div each columns by max 		
					
			# choose high contributing columns
			Qx = Q.T[-n_eigen:]  
			Qx = Qx.T          # Qx.shape = (20000, n_eigen)		
		
		
		
			# ---------------------------
			# CLUSTERING
			# ---------------------------
			
			
			# k-means cluster
			from sklearn.cluster import KMeans
			kmeans = KMeans(n_clusters=n_clusters)
			kmeans.fit(Qx)
			labels = kmeans.predict(Qx)		
			
			
			# tri_idx -> node index
			n_ch = int(self.bold_sig.shape[1])
			tri_idx = np.triu_indices(n_ch, k=1)   # k: offset, k=1 -> remove diagonal
			df1 = pd.DataFrame({"i": tri_idx[0], "j": tri_idx[1], "g": labels})
			df2 = pd.DataFrame({"i": tri_idx[1], "j": tri_idx[0], "g": labels})
			dfx = df1.append(df2).reset_index(drop=True)				
			
			# store gij table
			print("> Saving gij table")
			dfx.to_csv(self.fn_gij, index=False)		
		
	
	
	# =====================================
	# Constructor (parameters)
	# =====================================		
	def gen_pic(self):
	
		# ERROR, if there is no g_ij matrix
		if not( self.fn_gij.exists() ):	
			print('> ERROR: No Eigenvector matrix found')
			exit()			
		
				
		# load g_ij matrix
		df = pd.read_csv(self.fn_gij)
		

		# group by i
		df = df[['i', 'g']]
		dfg_i = df.groupby("i")


		# loop over set of pairs including "i"
		g_joint = []
		g_node = []
		for name, group in dfg_i:

			# ---------
			# name: i
			# group: (199 x [i, g]);    [i 가 고정되고 j\neq i 인 j 의 개수이므로 199 가 맞다.
			# ---------	
			
			# group [dfg_i] by c and sum()
			grc = group.groupby("g")
			ccc = grc.count()
			ccc = ccc / ccc.sum()

			# append res	
			g_joint.append(ccc)
			g_node.append(name)
			
				
		# cancat -> (200 x 10);   [200개의 서로 다른 i]
		dfx = pd.concat(g_joint, axis=1, keys=g_node, sort=False)
		dfx.columns = g_node
		dfx = dfx.fillna(0)
		dfx = dfx.T		
		
		
		
		# Euclidean similarlity between clusters
		#     sort colume by the Euclidean distance 
		#     col (cluster) 이름을 0 으로부터 col 까지 Euclidean distance 로 한 뒤, 
		#     col 이름으로 sort 한다
		empty_vec = np.zeros(len(dfx))
		f = lambda x: int(np.linalg.norm(empty_vec - x)*1000)
		eu_dist = dfx.apply(f, axis=0).values
		dfx.columns = eu_dist
		dfx = dfx.reindex(sorted(dfx.columns), axis=1)	
	
	
		# sort row by node categoroes (self.node_region)	
		cat = pd.Series(self.node_region, index=dfx.index, name='nr')		
		dfx2 = pd.concat([dfx, cat], axis=1, sort=False)
		dfx2.to_csv(self.fn_pic_table, index=False)
		
		
		
		# --------------------------
		# PLOT
		# --------------------------
		
		plt.close()
		fig, ax = plt.subplots(figsize=(8, 12))

		
		# over all cats (nr)
		dfg_nr = dfx2.groupby("nr")
		height_cumul = 0
		for name, group in dfg_nr:
		
			# ----------------
			# name: index for a region 
			#    (in the format of self.node_region; e.g. 1,..., 15)
			# group: nodes in the region
			# ---------------- 
		
		
			# make raster -> im
			raw = group.values[:, :-1]  # exclude 'nr'
			im = self.gen_raster(name, raw)
			width, height = im.size
			height_cumul += height
			
			# append raster to fig
			fig.figimage(im, xo=100, yo=1100-height_cumul, cmap=plt.cm.gray, origin='upper')
			
			
			# append region label
			reg_name = (self.region_name[name-1])
			reg_name = str(reg_name)
			text_im = Image.new("RGB", (50, 20), color=(255, 255, 255)) 
			draw = ImageDraw.Draw(text_im)
			draw.text((0, 0), reg_name, fill=(0,0,0,255))
			fig.figimage(text_im, xo=50, yo=1100-height_cumul, cmap=plt.cm.gray, origin='upper')		
		
		
		# save figure
		print("> Saving p_ic figure")
		plt.savefig(self.fn_pic_fig)
		


	# =====================================
	# Constructor (parameters)
	# =====================================		
	def normalized_entropy(self):
	
		# ERROR, if there is no p_ic matrix
		if not( self.fn_pic_table.exists() ):	
			print('> ERROR: No Eigenvector matrix found')
			exit()		
		
		# load p_ic matrix
		#     +1e-10 to prevent log(0) error
		df = pd.read_csv(self.fn_pic_table) + 1e-10  
		

		# remove nr (node region code)
		new_col = df.columns.to_list()
		new_col.remove('nr')
		df_new = df[new_col]
		
		
		# log pic
		df_log = np.log2(df_new)
		
		
		# entropy	
		n_k = df_new.shape[1]
		df['entropy'] = -(df * df_log).sum(axis=1)
		df['entropy'] = df['entropy'] / np.log2(n_k)  # normalize
		df_res = df[['nr', 'entropy']]
		# df_res['nr'] = df_res['nr'].astype('int')
		
		
		print(df_res)	
		
		
	# =====================================
	# Constructor (parameters)
	# =====================================		
	def normalized_entropy(self):
	
		
	
		
		
	# ==================================
	# main
	# ==================================
	def gen_raster(self, gi, raw):
		 
		# information
		len_raw = raw.shape[0]
		
		# normalize row 
		row_max = np.max(raw, axis=1)
		row_max = row_max.reshape(-1, 1)  # make column vector
		raw = raw/(row_max)	
		
		# apply threshold
		raw[raw < 0.5] = 0
		

		# choose color code
		vals = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
		        'YlOrBr', 'PuRd', 'RdPu', 'BuPu', 'GnBu']
		myMap = dict( zip(range(len(vals)), vals) ) 
		cc = myMap[gi%len(vals)]
		
		# Generate Raster	 
		cm = plt.get_cmap(cc)
		color_raw = cm(raw)
		im = Image.fromarray((color_raw[:, :, :3] * 255).astype(np.uint8))
		im = im.resize((600, len_raw*5), Image.NEAREST)
		
		
		# apply borders
		szx, szy = im.size
		new_im = Image.new("RGB", (szx+4, szy+4)) 
		new_im.paste(im, (2, 2))	

		return new_im			

	
	# =====================================
	# stop code
	# =====================================		
	def stop_code(self):
		plt.close()
		fig, ax = plt.subplots(figsize=(12, 12))
		ax.plot([0, 1], [0, 1])
		plt.show()
		
	
