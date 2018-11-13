import numpy as np
import numpy.linalg as la
from sklearn.decomposition import PCA

data=np.asarray(
	[[79.9,13.9,6.2,3.3],
	[78.5,16.3,7.2,2.5],
	[68.9,22.6,8.5,3.6],
	[62.2,20.2,17.6,2.8],
	[69.2,23.7,7.1,0.9],
	[67.8,19.8,12.4,3.8],
	[61.3,24.9,13.8,2.2],
	[71.6,19.2,9.2,3.6],
	[83.7,10.5,5.8,4.4],
	[67.1,26.5,6.4,1.4],
	[59.8,27.9,12.3,3.5],
	[66.7,23.2,10.1,2.9],
	[72.8,14.5,12.7,1.9],
	[60.9,28.9,10.2,1.5],
	[61.4,29.2,9.4,2.5],
	[75,16.8,8.2,3.1],
	[80.5,11.9,7.6,3.8],
	[71.3,18.5,10.2,2.6],
	[56.6,28.9,14.5,2.8],
	[55.9,32.8,11.3,3.1],
	[61.5,28.1,10.4,2.7],
	[59.2,28.4,12.4,2.8],
	[76.9,16.3,6.8,2.9],
	[58,27.6,14.4,3.4]])


cov_data=np.cov(np.transpose(data)) # Obtém matriz de covariância dos dados
w,v=la.eig(cov_data)                # Obtém autovalores e autovetores

ind=np.argsort(w)[::-1]  # Obtém índices para ordenação decrescente dos autovalores
w_dec=w[ind]
v_dec=v[ind]

EVR=w/np.sum(w)

datac=data-np.mean(data,axis=0)

U,S,V=la.svd(datac)

pca=PCA()                                    # inicializa classe
pca.fit(datac)                                # submete os dados 
pca.components_                              # componentes principais
pca.explained_variance_ratio_                # EVR
pca.explained_variance_

print(pca.components_)
print(EVR)
print(v_dec)

print('Y = ' + str(pca.components_[:,0].T) + ' X1' + ' + ' + str(pca.components_[:,1].T) + ' X2 ' + str(pca.components_[:,2].T) + ' X3 ' + ' + ' + str(pca.components_[:,3].T) + ' X4')