from sklearn import cluster
import torch

class KMeansPruner():

    
    def get_masks_and_centroids(self, t, n_clusters, device='cpu'):

        out_sh, in_sh, h, w = t.shape
        mtrx = t.reshape(out_sh, -1)

        kmeans = cluster.KMeans(n_clusters, 
                               random_state=4)
        kmeans.fit(mtrx.tolist())
        
        #Leave only one first element for each cluster, mask others 
        indices = [1 if (not kmeans.labels_[i] in kmeans.labels_[:i]
                        ) else 0 for i in range(len(mtrx))] 
        
        mask = torch.zeros_like(mtrx)#.int()
        mask[torch.nonzero(torch.Tensor(indices))] = 1
        mask = mask.reshape([out_sh, in_sh, h, w])
        
        #Replace cluster elements by cluster center
        return mask, torch.tensor(
            [kmeans.cluster_centers_[kmeans.labels_[i]] for i in range(len(t))]
        ).reshape([out_sh, in_sh, h, w]).float().to(device)
