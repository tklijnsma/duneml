'''Python imports'''
import numba as nb
import numpy as np
import awkward as ak
from math import sqrt
import networkx as nx
import tqdm
import os, os.path as osp
from time import strftime

'''Torch imports'''
import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_mean, scatter_add

import torch_geometric
from torch_geometric.nn import max_pool_x
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import EdgeConv
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.data import DataLoader


#https://github.com/eldridgejm/unionfind
from unionfind import UnionFind
import pdb 



def center_embedding_truth(coords, truth_label_by_hits, device='cpu'):
    _debug = False
    def debug(*args, **kwargs):
        if _debug: print('debug center_embedding_truth: ', *args, **kwargs)

    debug('coords: {}; truth_label_by_hits: {}'.format(coords.shape, truth_label_by_hits.shape))

    truth_ordering = torch.argsort(truth_label_by_hits)
    debug('truth_ordering: {}'.format(truth_ordering.shape))
    uniques, counts = torch.unique(truth_label_by_hits, return_counts=True)
    debug('uniques: {}; counts: {}'.format(uniques.shape, counts.shape))


    n_hits = truth_label_by_hits.size()[0]
    n_clusters = uniques.size()[0]
    debug(f'n_hits: {n_hits}; n_clusters: {n_clusters}')
    
    centers = scatter_mean(coords.T, truth_label_by_hits, dim=1, dim_size=n_clusters).T
    debug(f'centers: {centers.shape}')

    all_dists = torch.cdist(coords.expand(n_clusters,-1,-1).contiguous(), centers[:,None])
    copied_truth = truth_label_by_hits.expand(n_clusters,-1)
    copied_uniques = uniques[:,None].expand(-1, n_hits)
    truth = torch.where(copied_truth == copied_uniques,
                        torch.ones((n_clusters, n_hits), dtype=torch.int64, device=device), 
                        torch.full((n_clusters, n_hits), -1, dtype=torch.int64, device=device))
    
    dists_out = all_dists.reshape(n_clusters*n_hits)
    truth_out = truth.reshape(n_clusters*n_hits)

    debug('dists_out: {}; truth_out: {}'.format(dists_out.shape, truth_out.shape))    
    # debug('dists_out:\n{}\n; truth_out: \n{}'.format(dists_out[:6], truth_out[:6]))
    return (dists_out, truth_out)

    return out_truths

@nb.njit()
def build_cluster_list(indices, clusters, labels):
    pred_clusters = []
    for label in labels:
        pred_clusters.append(indices[clusters == label])
    return pred_clusters

@nb.njit()
def do_matching(indices, pred_labels, true_labels, data_y, data_y_particle_barcodes, data_truth_barcodes, 
               data_truth_pt, data_truth_eta, data_truth_phi):
    matched_pred_clusters = []
    true_cluster_properties = np.zeros((len(true_labels), 3), dtype=np.float32)
    for i, label in enumerate(true_labels):
        true_indices = set(indices[data_y == label])
        best_pred_cluster = -1
        best_iou = 0
        for j, pc in enumerate(pred_labels):
            #print('i-pred',i)
            pc = set(pc)
            isec = true_indices & pc
            iun = true_indices | pc
            iou = len(isec)/len(iun)
            if best_pred_cluster == -1 or iou > best_iou:
                best_pred_cluster = j
                best_iou = iou
        matched_pred_clusters.append(best_pred_cluster)
        # now make the properties vector
        thebc = np.unique(data_y_particle_barcodes[data_y == label])[0]
        select_truth = (data_truth_barcodes == thebc)
        pt_inv = np.reciprocal(data_truth_pt[select_truth])
        eta = data_truth_eta[select_truth]
        phi = data_truth_phi[select_truth]
        true_cluster_properties[i][0] = pt_inv[0]
        true_cluster_properties[i][1] = eta[0]
        true_cluster_properties[i][2] = phi[0]

    return matched_pred_clusters, true_cluster_properties

def match_cluster_targets(clusters, truth_clusters, data):
    np_truth_clusters = truth_clusters.cpu().numpy()
    true_cluster_labels = np.unique(np_truth_clusters)   
    np_clusters = clusters.cpu().numpy()
    pred_cluster_labels = np.unique(np_clusters)
    pred_cluster_mask = np.ones_like(np_truth_clusters, dtype=np.bool)
        
    indices = np.arange(np_clusters.size, dtype=np.int64)
    pred_clusters = build_cluster_list(indices, np_clusters, pred_cluster_labels)
    
    matched_pred_clusters, true_cluster_properties = \
        do_matching(indices, pred_clusters, true_cluster_labels, 
                    data.y.cpu().numpy(),
                    data.y_particle_barcodes.cpu().numpy(),
                    data.truth_barcodes.cpu().numpy(),
                    data.truth_pt.cpu().numpy(), 
                    data.truth_eta.cpu().numpy(), 
                    data.truth_phi.cpu().numpy())

    matched_pred_clusters = np.array(matched_pred_clusters, dtype=np.int64)
    pred_indices = torch.from_numpy(matched_pred_clusters).to(clusters.device)
    
    true_cluster_properties = np.array(true_cluster_properties, dtype=np.float)
    y_properties = torch.from_numpy(true_cluster_properties).to(clusters.device).float()
    
    return pred_indices, y_properties



class SimpleEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, ncats_out=2, nprops_out=1, output_dim=8,
                 conv_depth=3, edgecat_depth=6, property_depth=3, k=8, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.]), interm_out=6):
        super(SimpleEmbeddingNetwork, self).__init__()
        
        # self.datanorm = nn.Parameter(norm, requires_grad=False)
        self.k = k
        self.nprops_out = nprops_out
        
        start_width = 2 * (hidden_dim )
        middle_width = (3 * hidden_dim ) // 2
        
        '''Main Input Net'''
        # embedding loss
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        )        
        
        '''Main Edge Convolution'''
        self.edgeconvs = nn.ModuleList()
        for i in range(conv_depth):
            convnn = nn.Sequential(
                nn.Linear(start_width, middle_width),
                nn.ELU(),
                nn.Linear(middle_width, middle_width),                                             
                nn.ELU(),
                nn.Linear(middle_width, hidden_dim),                                             
                nn.ELU(),
                #nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
            )
            self.edgeconvs.append(EdgeConv(nn=convnn, aggr=aggr))
        
        
        '''Embedding Output Net'''
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
            # nn.Linear(hidden_dim, interm_out)
        )
        # self.plotlayer = nn.Sequential(
        #     nn.Linear(interm_out, interm_out),
        #     nn.ELU(),
        #     nn.Linear(interm_out, output_dim))

        
        # edge categorization
        '''InputNetCat'''
        self.inputnet_cat =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),            
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Tanh(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Tanh()            
        )
        
        '''EdgeConcat Convolution'''
        self.edgecatconvs = nn.ModuleList()
        for i in range(edgecat_depth):
            convnn = nn.Sequential(
                nn.Linear(start_width + 2*hidden_dim + 2*input_dim, middle_width),
                nn.ELU(),
                nn.Linear(middle_width, hidden_dim),                                             
                nn.ELU(),
                # nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
            )
            self.edgecatconvs.append(EdgeConv(nn=convnn, aggr=aggr))
        
        '''Edge Classifier'''
        self.edge_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.BatchNorm1d(num_features=hidden_dim//2, track_running_stats=False),
            nn.Linear(hidden_dim//2, ncats_out)
        )
        
        # '''InputNet for Cluster Properties'''
        # self.inputnet_prop =  nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),            
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
        #     nn.Tanh(),            
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
        #     nn.Tanh()            
        # )
        
        # '''Convolution for Cluster Properties'''
        # self.propertyconvs = nn.ModuleList()
        # for i in range(property_depth):
        #     convnn = nn.Sequential(
        #         nn.Linear(start_width + 2*hidden_dim + 2*input_dim, middle_width),
        #         nn.ELU(),
        #         nn.Linear(middle_width, hidden_dim),                                             
        #         nn.ELU(),
        #         # nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        #     )
        #     self.propertyconvs.append(EdgeConv(nn=convnn, aggr='max'))

        # '''Classifier for Cluster Properties'''
        # self.property_predictor = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim//2, nprops_out)
        # )       

        self._debug = False


    def forward(self, x, batch: OptTensor=None):
        def debug(*args, **kwargs):
            if self._debug: print('debug forward: ', *args, **kwargs)

        if batch is None:
            batch = torch.zeros(x.size()[0], dtype=torch.int64, device=x.device)
        
        '''Embedding1: Intermediate Latent space features (hiddenDim)'''
        x_emb = self.inputnet(x)
        debug('Embedding: {} -> {}'.format(x.shape, x_emb.shape))

        '''KNN(k neighbors) over intermediate Latent space features'''     
        debug('Doing edge_convs...')
        for ec in self.edgeconvs:
            edge_index = knn_graph(x_emb, self.k, batch, loop=False, flow=ec.flow)
            x_emb = x_emb + ec(x_emb, edge_index)
        debug('edgeconvs done: x_emd: {}, edge_index: {}'.format(x_emb.shape, edge_index.shape))
    
        '''
        [1]
        Embedding2: Final Latent Space embedding coords from x,y,z to ncats_out
        '''

        out = self.output(x_emb)
        debug('To n cats out: {} -> {}'.format(x_emb.shape, out.shape))
        #plot = self.plotlayer(out)


        '''KNN(k neighbors) over Embedding2 features''' 
        debug('Doing radius_graph...')
        edge_index = radius_graph(out, r=0.1, batch=batch, max_num_neighbors=self.k//2, loop=False)
        debug('radius_graph done: edge_index: {}'.format(edge_index.shape))
        

        ''' 
        use Embedding1 to build an edge classifier
        inputnet_cat is residual to inputnet
        '''
        x_cat = self.inputnet_cat(x) #+ x_emb
        debug('x_cat: {}'.format(x_cat.shape))

        '''
        [2]
        Compute Edge Categories Convolution over Embedding1
        '''
        for ec in self.edgecatconvs:            
            x_cat = x_cat + ec(torch.cat([x_cat, x_emb.detach(), x], dim=1), edge_index)
        
        edge_scores = self.edge_classifier(torch.cat([x_cat[edge_index[0]], 
                                                      x_cat[edge_index[1]]], 
                                                      dim=1))#.squeeze()
        debug('Created edge_scores: {}'.format(edge_scores.shape))

        '''
        use the predicted graph to generate disjoint subgraphs
        these are our physics objects
        '''
        objects = UnionFind(x.size()[0])
        good_edges = edge_index[:,torch.argmax(edge_scores, dim=1) > 0]
        good_edges_cpu = good_edges.cpu().numpy() 

        for edge in good_edges_cpu.T:
            objects.union(edge[0],edge[1])
        cluster_map = torch.from_numpy(np.array([objects.find(i) for i in range(x.shape[0])], 
                                                dtype=np.int64)).to(x.device)
        cluster_roots, inverse = torch.unique(cluster_map, return_inverse=True)
        # remap roots to [0, ..., nclusters-1]
        cluster_map = torch.arange(cluster_roots.size()[0], 
                                   dtype=torch.int64, 
                                   device=x.device)[inverse]
        

        # ''' 
        # [3]
        # use Embedding1 to learn segmented cluster properties 
        # inputnet_cat is residual to inputnet
        # '''
        # x_prop = self.inputnet_prop(x) #+ x_emb
        # # now we accumulate over all selected disjoint subgraphs
        # # to define per-object properties
        # for ec in self.propertyconvs:
        #     x_prop = x_prop + ec(torch.cat([x_prop, x_emb.detach(), x], dim=1), good_edges)        
        # props_pooled, cluster_batch = max_pool_x(cluster_map, x_prop, batch)
        # cluster_props = self.property_predictor(props_pooled)    

        return out, edge_scores, edge_index, cluster_map #, cluster_batch





def main():
    from dataset import DuneDataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--limited', action='store_true',
        help='Use the dataset with limited number of tracks instead'
        )
    args = parser.parse_args()

    ckpt_dir = strftime('ckpts_%b%d_%H%M%S')
    n_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)
    norm = torch.tensor([1., 1., 1., 1.])

    if args.limited:
        batch_size = 16
        train_dataset = DuneDataset('data_lim/train')
        test_dataset = DuneDataset('data_lim/test')
        model = SimpleEmbeddingNetwork(
            input_dim=4,
            hidden_dim=32,
            output_dim=2,
            ncats_out=2,
            conv_depth=3,
            edgecat_depth=6,
            k=16,
            aggr='add',
            norm=norm,
            ).to(device)
    else:
        batch_size = 6
        train_dataset = DuneDataset('data/train')
        test_dataset = DuneDataset('data/test')
        model = SimpleEmbeddingNetwork(
            input_dim=4,
            hidden_dim=16,
            output_dim=2,
            ncats_out=2,
            conv_depth=2,
            edgecat_depth=2,
            k=8,
            aggr='add',
            norm=norm,
            ).to(device)

    print('Model:\n',model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    optimizer = torch.optim.AdamW([
        {'params': list(model.inputnet.parameters()) + list(model.edgeconvs.parameters()) + list(model.output.parameters())},
        {'params': list(model.inputnet_cat.parameters()) + list(model.edgecatconvs.parameters()) + list(model.edge_classifier.parameters()), 'lr': .001},
        ],
        lr=0.001, weight_decay=1e-3
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.7, patience=10
        )


    def train(epoch):
        _debug = False
        def debug(*args, **kwargs):
            if _debug: print('debug train: ', *args, **kwargs)


        print('Training epoch', epoch)
        model.train()
        # scheduler.step()

        avg_loss = 0.

        for data in tqdm.tqdm(train_loader, total=len(train_loader), position=0, leave=True):
            data = data.to(device)
            data.y -= 1
            optimizer.zero_grad()
            x = data.x.type(torch.FloatTensor).to(device)
            # batch = x.unsqueeze(dim=0)
            # print('data.x: {} -> x: {}'.format(data.x.shape, x.shape))
            coords, edge_scores, edges, cluster_map = model(x, data.batch)

            '''Compute latent space distances'''
            d_hinge, y_hinge = center_embedding_truth(coords, data.y, device=device)

            '''Compute centers in latent space '''
            centers = scatter_mean(coords, data.y, dim=0, dim_size=(torch.max(data.y).item()+1))
            
            '''Compute Losses'''
            # Hinge: embedding distance based loss
            loss_hinge = F.hinge_embedding_loss(torch.where(y_hinge == 1, 
                                                            d_hinge**2, 
                                                            d_hinge), 
                                                y_hinge, 
                                                margin=2.0, reduction='mean')
            
            #Cross Entropy: Edge categories loss
            y_edgecat = (data.y[edges[0]] == data.y[edges[1]]).long()

            debug('edge_scores: {}; y_edgecat: {}'.format(edge_scores.shape, y_edgecat.shape))
            # debug('edge_scores: {}; y_edgecat: {}'.format(edge_scores[:6], y_edgecat[:6]))

            # edge_scores = edge_scores[:,0]
            # debug('edge_scores: {}'.format(edge_scores.shape))

            # y_edgecat = y_edgecat.unsqueeze(1)
            # debug('y_edgecat: {}'.format(y_edgecat.shape))

            # edge_predictions = torch.argmax(
            #     torch.exp(torch.nn.functional.log_softmax(edge_scores, 1)), 1
            #     )
            # debug('edge_predictions:')
            # debug(edge_predictions.shape)
            # debug(edge_predictions)

            loss_ce = F.cross_entropy(edge_scores, y_edgecat, reduction='mean')

            
            # #MSE: Cluster loss
            # pred_cluster_match, y_properties = match_cluster_targets(cluster_map, data.y, data)
            # mapped_props = cluster_props[pred_cluster_match].squeeze()
            # props_pt = F.softplus(mapped_props[:,0])
            # props_eta = 5.0*(2*torch.sigmoid(mapped_props[:,1]) - 1)
            # props_phi = math.pi*(2*torch.sigmoid(mapped_props[:,2]) - 1)
    
            # loss_mse = ( F.mse_loss(props_pt, y_properties[:,0], reduction='mean') +
            #              F.mse_loss(props_eta, y_properties[:,1], reduction='mean') +
            #              F.mse_loss(props_phi, y_properties[:,2], reduction='mean') ) / model.nprops_out
            
            #Combined loss
            # loss = (loss_hinge + loss_ce + loss_mse) / config.batch_size
            loss = (loss_hinge + loss_ce) / batch_size
            # avg_loss += loss.item()

            # print('loss_hinge:', loss_hinge.item())
            # print('loss_ce:', loss_ce.item())
            # print('loss:', loss.item())
            # print('avg_loss:', avg_loss)

            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())


    def test():
        with torch.no_grad():
            model.eval()
            n_correct = 0
            n_total = 0
            for data in tqdm.tqdm(test_loader, total=len(test_loader), position=0, leave=True):
                data = data.to(device)
                data.y -= 1
                x = data.x.type(torch.FloatTensor).to(device)
                coords, edge_scores, edges, cluster_map = model(x, data.batch)
                edge_predictions = torch.argmax(
                    torch.exp(torch.nn.functional.log_softmax(edge_scores, 1)), 1
                    )

                y_edgecat = (data.y[edges[0]] == data.y[edges[1]]).long()

                n_correct += torch.sum(edge_predictions == y_edgecat)
                n_total += edge_predictions.shape[0]
        acc = n_correct / n_total
        print(f'Test acc: {acc:.3f} ({n_correct}/{n_total})')
        return acc


    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        os.makedirs(ckpt_dir, exist_ok=True)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        torch.save(dict(model=model.state_dict()), ckpt)


    best_test_acc = 0.0
    for epoch in range(1, 1+n_epochs):
        train(epoch)
        test_acc = test()
        write_checkpoint(epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            write_checkpoint(epoch, best=True)



if __name__ == "__main__":
    main()
