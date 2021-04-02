'''Python imports'''
import numba as nb
import numpy as np
import awkward as ak
from math import sqrt
import networkx as nx
import tqdm

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
    truth_ordering = torch.argsort(truth_label_by_hits)    
    uniques, counts = torch.unique(truth_label_by_hits, return_counts=True)

    n_hits = truth_label_by_hits.size()[0]
    n_clusters = uniques.size()[0]
    
    centers = scatter_mean(coords, truth_label_by_hits, dim=0, dim_size=n_clusters)

    all_dists = torch.cdist(coords.expand(n_clusters,-1,-1).contiguous(), centers[:,None])
    copied_truth = truth_label_by_hits.expand(n_clusters,-1)
    copied_uniques = uniques[:,None].expand(-1, n_hits)
    truth = torch.where(copied_truth == copied_uniques,
                        torch.ones((n_clusters, n_hits), dtype=torch.int64, device=device), 
                        torch.full((n_clusters, n_hits), -1, dtype=torch.int64, device=device))
    
    dists_out = all_dists.reshape(n_clusters*n_hits)
    truth_out = truth.reshape(n_clusters*n_hits)
    
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

        self._debug = True


    def forward(self, x, batch: OptTensor=None):
        def debug(*args, **kwargs):
            if self._debug: print('debug: ', *args, **kwargs)

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
        debug('edgeconvs done: {}, edge_index: {}'.format(x_emb.shape, edge_index.shape))
    
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

        '''
        [2]
        Compute Edge Categories Convolution over Embedding1
        '''
        for ec in self.edgecatconvs:            
            x_cat = x_cat + ec(torch.cat([x_cat, x_emb.detach(), x], dim=1), edge_index)
        
        edge_scores = self.edge_classifier(torch.cat([x_cat[edge_index[0]], 
                                                      x_cat[edge_index[1]]], 
                                                      dim=1)).squeeze()
        

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

        return out, edge_scores, edge_index, cluster_map, cluster_batch





def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm = torch.tensor([1., 1., 1.])
    model = SimpleEmbeddingNetwork(
        input_dim=3, 
        hidden_dim=16, 
        output_dim=2,
        ncats_out=1,
        conv_depth=2, 
        edgecat_depth=2, 
        k=4,
        aggr='add',
        norm=norm,
        ).to(device)
    print('Model:\n',model)

    from dataset import DuneDataset
    dataset = DuneDataset('data')

    batch_size = 16
    train_dataset = DuneDataset('data/train')
    test_dataset = DuneDataset('data/test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW([
        {'params': list(model.inputnet.parameters()) + list(model.edgeconvs.parameters()) + list(model.output.parameters())},
        {'params': list(model.inputnet_cat.parameters()) + list(model.edgecatconvs.parameters()) + list(model.edge_classifier.parameters()), 'lr': .001},
        ],
        lr=0.001, weight_decay=1e-3
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.7, patience=30
        )


    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        # scheduler.step()

        for data in tqdm.tqdm(train_loader, total=len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            result = model(data.x.type(torch.FloatTensor))
            print(result)
            break




    train(0)

    return











    model.train()

    combo_loss_avg = []
    sep_loss_avg = []
    pred_cluster_properties = []
    edge_acc_track = np.zeros(config.train_samples, dtype=np.float)
    
    color_cycle = plt.cm.coolwarm(np.linspace(0.1,0.9,(config.input_classes+config.input_class_delta)*config.k))
    marker_hits =    ['^','v','s','h','<','>']
    marker_centers = ['+','1','x','3','2','4']

    if(input_classes_rand==None):
        '''Input Class +- Range'''
        input_classes_rand = torch.randint(low = config.input_classes - config.input_class_delta, 
                                            high= config.input_classes + config.input_class_delta+1,
                                            size= (config.train_samples,), device=torch.device('cuda'))

    print('\n[TRAIN]:')

    t1 = timer()
    
    for epoch in range(start_epoch, start_epoch+config.total_epochs):
        
        '''book-keeping'''
        sep_loss_track = np.zeros((config.train_samples,3), dtype=np.float)
        avg_loss_track = np.zeros(config.train_samples, dtype=np.float)
        edge_acc_track = np.zeros(config.train_samples, dtype=np.float)
        edge_acc_conf  = np.zeros((config.train_samples,config.ncats_out,config.ncats_out), dtype=np.int)
        pred_cluster_properties = []
        avg_loss = 0

        opt.zero_grad()
        # if opt.param_groups[0]['lr'] < lr_threshold_1 and not converged_embedding:
        #     converged_embedding = True
        #     opt.param_groups[1]['lr'] = lr_threshold_1
        #     opt.param_groups[2]['lr'] = lr_threshold_2
            
        # if opt.param_groups[1]['lr'] < lr_threshold_1 and not converged_categorizer and converged_embedding:
        #     converged_categorizer = True
        #     opt.param_groups[2]['lr'] = lr_threshold_2
        

        for idata, d in enumerate(data[0:config.train_samples]):            
            
            d_gpu = d.to('cuda')
            y_orig = d_gpu.y

            d_gpu.x = d_gpu.x[d_gpu.y < input_classes_rand[idata]] 
            d_gpu.x = (d_gpu.x - torch.min(d_gpu.x, axis=0).values)/(torch.max(d_gpu.x, axis=0).values - torch.min(d_gpu.x, axis=0).values) # Normalise
            d_gpu.y_particle_barcodes = d_gpu.y_particle_barcodes[d_gpu.y < input_classes_rand[idata]]
            d_gpu.y = d_gpu.y[d_gpu.y < input_classes_rand[idata]]
            # plot_event(d_gpu.x.detach().cpu().numpy(), d_gpu.y.detach().cpu().numpy())
            
            '''
            project embedding to some nd latent space where it is seperable using the deep model
            compute edge net scores and seperated cluster properties with the ebedding
            '''
            coords, edge_scores, edges, cluster_map, cluster_props, cluster_batch = model(d_gpu.x)

            #-------------- LINDSEY TRAINING VERSION ------------------         
            '''Compute latent space distances'''
            d_hinge, y_hinge = center_embedding_truth(coords, d_gpu.y, device='cuda')
            
            '''Compute centers in latent space '''
            centers = scatter_mean(coords, d_gpu.y, dim=0, dim_size=(torch.max(d_gpu.y).item()+1))
            
            '''Compute Losses'''
            # Hinge: embedding distance based loss
            loss_hinge = F.hinge_embedding_loss(torch.where(y_hinge == 1, 
                                                            d_hinge**2, 
                                                            d_hinge), 
                                                y_hinge, 
                                                margin=2.0, reduction='mean')
            
            #Cross Entropy: Edge categories loss
            y_edgecat = (d_gpu.y[edges[0]] == d_gpu.y[edges[1]]).long()
            loss_ce = F.cross_entropy(edge_scores, y_edgecat, reduction='mean')
            
            #MSE: Cluster loss
            pred_cluster_match, y_properties = match_cluster_targets(cluster_map, d_gpu.y, d_gpu)
            mapped_props = cluster_props[pred_cluster_match].squeeze()
            props_pt = F.softplus(mapped_props[:,0])
            props_eta = 5.0*(2*torch.sigmoid(mapped_props[:,1]) - 1)
            props_phi = math.pi*(2*torch.sigmoid(mapped_props[:,2]) - 1)
    
            loss_mse = ( F.mse_loss(props_pt, y_properties[:,0], reduction='mean') +
                         F.mse_loss(props_eta, y_properties[:,1], reduction='mean') +
                         F.mse_loss(props_phi, y_properties[:,2], reduction='mean') ) / model.nprops_out
            
            #Combined loss
            loss = (loss_hinge + loss_ce + loss_mse) / config.batch_size
            avg_loss_track[idata] = loss.item()
            
            avg_loss += loss.item()

            '''Track Losses, Acuracies and Properties'''   
            sep_loss_track[idata,0] = loss_hinge.detach().cpu().numpy() / config.batch_size
            sep_loss_track[idata,1] = loss_ce.detach().cpu().numpy() / config.batch_size
            sep_loss_track[idata,2] = loss_mse.detach().cpu().numpy() / config.batch_size

            true_edges = y_edgecat.sum().item()
            edge_accuracy = (torch.argmax(edge_scores, dim=1) == y_edgecat).sum().item() / (y_edgecat.size()[0])
            edge_acc_track[idata] = edge_accuracy

            edge_acc_conf[idata,:,:] = confusion_matrix(y_edgecat.detach().cpu().numpy(), torch.argmax(edge_scores, dim=1).detach().cpu().numpy())

            true_prop = y_properties.detach().cpu().numpy()
            pred_prop = cluster_props[pred_cluster_match].squeeze().detach().cpu().numpy()
            pred_cluster_properties.append([(1./y_properties[:,0], 1./y_properties[:,1], 1./y_properties[:,2]),
                                            (1./props_pt), (1./props_eta), (1./props_phi)])

            '''Plot Training Clusters'''
            # if (config.make_train_plots==True):
            if (config.make_train_plots==True and (epoch==0 or epoch==start_epoch+config.total_epochs-1) and idata%(config.train_samples/10)==0):     
                fig = plt.figure(figsize=(8,8))
                if config.output_dim==3:
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(centers.size()[0]):  
                        ax.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                            coords[d_gpu.y == i,1].detach().cpu().numpy(),
                            coords[d_gpu.y == i,2].detach().cpu().numpy(),
                            color=color_cycle[(i*config.k)%((config.input_classes+config.input_class_delta)*config.k - 1)], marker = marker_hits[i%6], s=100)

                        ax.scatter(centers[i,0].detach().cpu().numpy(), 
                            centers[i,1].detach().cpu().numpy(), 
                            centers[i,2].detach().cpu().numpy(), 
                            marker=marker_centers[i%6], color=color_cycle[(i*config.k)%((config.input_classes+config.input_class_delta)*config.k- 1)], s=100); 
                elif config.output_dim==2:
                    for i in range(int(centers.size()[0])):
                            plt.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                                        coords[d_gpu.y == i,1].detach().cpu().numpy(),
                                        color=color_cycle[(i*config.k)%((config.input_classes+config.input_class_delta)*config.k - 1)], 
                                        marker = marker_hits[i%6] )

                            plt.scatter(centers[i,0].detach().cpu().numpy(), 
                                        centers[i,1].detach().cpu().numpy(), 
                                        color=color_cycle[(i*config.k)%((config.input_classes+config.input_class_delta)*config.k - 1)],  
                                        edgecolors='b',
                                        marker=marker_centers[i%6]) 
        
                plt.title('train_plot_epoch_'+str(epoch)+'_ex_'+str(idata)+'_EdgeAcc_'+str('{:.5e}'.format(edge_accuracy)))
                plt.savefig(config.plot_path+'train_plot_epoch_'+str(epoch)+'_ex_'+str(idata)+'.pdf')   
                plt.close(fig)

            '''Loss Backward''' 
            loss.backward()
            
            '''Update Weights'''
            if ( ((idata + 1) % config.batch_size == 0) or ((idata + 1) == config.train_samples) ):
                opt.step()
                if(config.schedLR):
                    sched.step(avg_loss)
        

        '''track Epoch Updates'''
        combo_loss_avg.append(avg_loss_track.mean())
        sep_loss_avg.append([sep_loss_track[:,0].mean(), sep_loss_track[:,1].mean(), sep_loss_track[:,2].mean()])
        
        true_0_1 = edge_acc_conf.sum(axis=2)
        pred_0_1 = edge_acc_conf.sum(axis=1) 
        total_true_0_1 =   true_0_1.sum(axis=0)
        total_pred_0_1 =   pred_0_1.sum(axis=0)
        
        # print('true_0_1:',true_0_1)
        # pdb.set_trace()

        if(epoch%10==0 or epoch==start_epoch or epoch==start_epoch+config.total_epochs-1):
            '''Per Epoch Stats'''
            print('--------------------')
            print("Epoch: {}\nLosses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                    epoch,combo_loss_avg[epoch-start_epoch],sep_loss_avg[epoch-start_epoch][0],sep_loss_avg[epoch-start_epoch][1],sep_loss_avg[epoch-start_epoch][2]))
            print("LR: opt.param_groups \n[0]: {:.9e}  \n[1]: {:.9e}  \n[2]: {:.9e}".format(opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr']))
            print("Track/Class Count Variation per event: {} +/- {} ".format(config.input_classes , config.input_class_delta))
            print("[TRAIN] Average Edge Accuracies over {} events: {:.5e}".format(config.train_samples,edge_acc_track.mean()) )
            print("Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
            print("Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))
        
            if(epoch==start_epoch+config.total_epochs-1 or epoch==start_epoch):
                logtofile(config.plot_path, config.logfile_name, "Epoch: {}\nLosses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                    epoch,combo_loss_avg[epoch-start_epoch],sep_loss_avg[epoch-start_epoch][0],sep_loss_avg[epoch-start_epoch][1],sep_loss_avg[epoch-start_epoch][2]))
                logtofile(config.plot_path, config.logfile_name,"LR: opt.param_groups \n[0]: {:.9e}  \n[1]: {:.9e}  \n[2]: {:.9e}".format(opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr']))
                logtofile(config.plot_path, config.logfile_name,"Track/Class Count Variation per event: {} +/- {} ".format(config.input_classes , config.input_class_delta))
                logtofile(config.plot_path, config.logfile_name,"Average Edge Accuracies over {} events, {} Tracks: {:.5e}".format(config.train_samples, config.input_classes,edge_acc_track.mean()) )                    
                logtofile(config.plot_path, config.logfile_name,"Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
                logtofile(config.plot_path, config.logfile_name,"Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))                
                logtofile(config.plot_path, config.logfile_name,'--------------------------')

            if(combo_loss_avg[epoch-start_epoch] < best_loss):
                best_loss = combo_loss_avg[epoch-start_epoch]
                is_best = True
                checkpoint = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(),
                    'converged_embedding':False,
                    'converged_categorizer':False,
                    'best_loss':best_loss,
                    'input_classes_rand':input_classes_rand
                }
                checkpoint_name = 'event'+str(config.train_samples)+'_classes' + str(config.input_classes) + '_epoch'+str(epoch) + '_loss' + '{:.5e}'.format(combo_loss_avg[epoch-start_epoch]) + '_edgeAcc' + '{:.5e}'.format(edge_acc_track.mean())
                save_checkpoint(checkpoint, is_best, config.checkpoint_path, checkpoint_name)

    t2 = timer()
    print('--------------------')
    print("Training Complted in {:.5f}mins.".format((t2-t1)/60.0))

    # print('1/properties: ',1/y_properties)
    # print('pred cluster matches: ',pred_cluster_match)
    # print('1/cluster_prop[cluster_match]: ',1/cluster_props[pred_cluster_match].squeeze())    

    return combo_loss_avg, sep_loss_avg, edge_acc_track, pred_cluster_properties, edge_acc_conf




if __name__ == "__main__":
    main()
