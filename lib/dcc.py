import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import os
import math
import collections
import pickle
import json

from lib.utils import acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import pandas as pd

class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return torch.mean((input-target)**2)


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class IDEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0, alpha=1., gamma=0.1):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(z)
        xrecon = self._dec(h)
        # compute q -> NxK
        q = self.soft_assign(z)
        return z, q, xrecon

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z,_, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        kldloss = kld(p, q)
        return self.gamma*kldloss

    def recon_loss(self, x, xrecon):
        recon_loss = torch.mean((xrecon-x)**2)
        return recon_loss

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return cl_loss

    def global_size_loss(self, p, cons_detail):
        m_p = torch.mean(p, dim=0)
        m_p = m_p / torch.sum(m_p)
        return torch.sum((m_p-cons_detail)*(m_p-cons_detail))

    def difficulty_loss(self, q, mask):
        mask = mask.unsqueeze_(-1)
        mask = mask.expand(q.shape[0], q.shape[1])
        mask_q = q * mask
        diff_loss = -torch.norm(mask_q, 2)
        penalty_degree = 0.1
        return penalty_degree * diff_loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def triplet_loss(self, anchor, positive, negative, margin_constant):
        # loss = max(d(anchor, negative) - d(anchor, positve) + margin, 0), margin > 0
        # d(x, y) = q(x) * q(y)
        negative_dis = torch.sum(anchor * negative, dim=1)
        positive_dis = torch.sum(anchor * positive, dim=1)
        margin = margin_constant * torch.ones(negative_dis.shape).cuda()
        diff_dis = negative_dis - positive_dis
        penalty = diff_dis + margin
        triplet_loss = 1*torch.max(penalty, torch.zeros(negative_dis.shape).cuda())

        return torch.mean(triplet_loss)

    def satisfied_constraints(self,ml_ind1,ml_ind2,cl_ind1, cl_ind2,y_pred):
        
        if ml_ind1.size == 0 or ml_ind2.size == 0 or cl_ind1.size == 0 or cl_ind2.size == 0:
            return 1.1

        count = 0
        satisfied = 0
        for (i, j) in zip(ml_ind1, ml_ind2):
            count += 1
            if y_pred[i] == y_pred[j]:
                satisfied += 1
        for (i, j) in zip(cl_ind1, cl_ind2):
            count += 1
            if y_pred[i] != y_pred[j]:
                satisfied += 1

        return float(satisfied)/count


    def predict(self, X, y):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        latent = self.encodeBatch(X)
        q = self.soft_assign(latent)

        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        y = y.data.cpu().numpy()
        if y is not None:
            print("acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
        return final_acc, final_nmi

    def fit(self,anchor, positive, negative, ml_ind1,ml_ind2,cl_ind1, cl_ind2, mask, use_global, ml_p, cl_p, X,y=None, lr=0.001, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, use_kmeans=True, plotting="",clustering_loss_weight=1):    
        
        # save intermediate results for plotting
        intermediate_results = collections.defaultdict(lambda:{})
        
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Training IDEC=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        if use_kmeans:
            print("Initializing cluster centers with kmeans.")
            kmeans = KMeans(self.n_clusters, n_init=20)
            data = self.encodeBatch(X)
            y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            y_pred_last = y_pred
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        else:
            # use kmeans to randomly initialize cluster ceters
            print("Randomly initializing cluster centers.")
            kmeans = KMeans(self.n_clusters, n_init=1, max_iter=1)
            data = self.encodeBatch(X)
            y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            y_pred_last = y_pred
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        if y is not None:
            y = y.cpu().numpy()
            # print("Kmeans acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/batch_size))
        cl_num_batch = int(math.ceil(1.0*cl_ind1.shape[0]/batch_size))
        tri_num_batch = int(math.ceil(1.0*anchor.shape[0]/batch_size))
        cl_num = cl_ind1.shape[0]
        ml_num = ml_ind1.shape[0]
        tri_num = anchor.shape[0]

        final_acc, final_nmi, final_epoch = 0, 0, 0
        update_ml = 1
        update_cl = 1
        update_triplet = 1
        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                if use_global:
                    y_dict = collections.defaultdict(list)
                    ind1, ind2 = [], []
                    for i in range(y_pred.shape[0]):
                        y_dict[y_pred[i]].append(i)
                    for key in y_dict.keys():
                        if y is not None:
                            print("predicted class: ", key, " total: ", len(y_dict[key]))
                            #, " mapped index(ground truth): ", np.bincount(y[y_dict[key]]).argmax())

                if y is not None:
                    print("acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
                    print("satisfied constraints: %.5f"%self.satisfied_constraints(ml_ind1,ml_ind2,cl_ind1, cl_ind2,y_pred))
                    final_acc = acc(y, y_pred)
                    final_nmi = normalized_mutual_info_score(y, y_pred)
                    final_epoch = epoch

                # save model for plotting
                if plotting and (epoch in [10,20,30,40] or epoch%50 == 0 or epoch == num_epochs-1):
                    
                    df = pd.DataFrame(latent.cpu().numpy())
                    df["y"] = y
                    df.to_pickle(os.path.join(plotting,"save_model_%d.pkl"%(epoch)))
                    
                    intermediate_results["acc"][str(epoch)] = acc(y, y_pred)
                    intermediate_results["nmi"][str(epoch)] = normalized_mutual_info_score(y, y_pred)
                    with open(os.path.join(plotting,"intermediate_results.json"), "w") as fp:
                        json.dump(intermediate_results, fp)

                # check stop criterion
                try:
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                    y_pred_last = y_pred
                    if epoch>0 and delta_label < tol:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")

                        # save model for plotting
                        if plotting:
                            
                            df = pd.DataFrame(latent.cpu().numpy())
                            df["y"] = y
                            df.to_pickle(os.path.join(plotting,"save_model_%d.pkl"%epoch))
                            
                            intermediate_results["acc"][str(epoch)] = acc(y, y_pred)
                            intermediate_results["nmi"][str(epoch)] = normalized_mutual_info_score(y, y_pred)
                            with open(os.path.join(plotting,"intermediate_results.json"), "w") as fp:
                                json.dump(intermediate_results, fp)
                        break
                except:
                    pass

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            instance_constraints_loss_val = 0.0
            global_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                mask_batch = mask[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)
                cons_detail = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                global_cons = torch.from_numpy(cons_detail).float().to("cuda")

                z, qbatch, xrecon = self.forward(inputs)
                if use_global == False:
                    cluster_loss = self.cluster_loss(target, qbatch)
                    recon_loss = self.recon_loss(inputs, xrecon)
                    instance_constraints_loss = self.difficulty_loss(qbatch, mask_batch)
                    loss = cluster_loss + recon_loss + instance_constraints_loss
                    loss.backward()
                    optimizer.step()
                    cluster_loss_val += cluster_loss.data * len(inputs)
                    recon_loss_val += recon_loss.data * len(inputs)
                    instance_constraints_loss_val += instance_constraints_loss.data * len(inputs)
                    train_loss = clustering_loss_weight*cluster_loss_val + recon_loss_val + instance_constraints_loss_val
                else:
                    cluster_loss = self.cluster_loss(target, qbatch)
                    recon_loss = self.recon_loss(inputs, xrecon)
                    global_loss = self.global_size_loss(qbatch, global_cons)
                    loss = cluster_loss + recon_loss + global_loss
                    loss.backward()
                    optimizer.step()
                    cluster_loss_val += cluster_loss.data * len(inputs)
                    recon_loss_val += recon_loss.data * len(inputs)
                    train_loss = clustering_loss_weight*cluster_loss_val + recon_loss_val


            if instance_constraints_loss_val != 0.0:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f Instance Difficulty Loss: %.4f"% (
                    epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, instance_constraints_loss_val / num))
            elif global_loss_val != 0.0 and use_global:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f Global Loss: %.4f"% (
                    epoch + 1, train_loss / num + global_loss_val/num_batch, cluster_loss_val / num, recon_loss_val / num, global_loss_val / num_batch))
            else:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f" % (
                    epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))
            ml_loss = 0.0
            if epoch % update_ml == 0:
                for ml_batch_idx in range(ml_num_batch):
                    px1 = X[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    px2 = X[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pbatch1 = p[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    loss = (ml_p*self.pairwise_loss(q1, q2, "ML")+self.recon_loss(inputs1, xr1) + self.recon_loss(inputs2, xr2))
                    # 0.1 for mnist/reuters, 1 for fashion, the parameters are tuned via grid search on validation set
                    ml_loss += loss.data
                    loss.backward()
                    optimizer.step()

            cl_loss = 0.0
            if epoch % update_cl == 0:
                for cl_batch_idx in range(cl_num_batch):
                    px1 = X[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    px2 = X[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    pbatch1 = p[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    loss = cl_p*self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if ml_num_batch >0 and cl_num_batch > 0:
                print("Pairwise Total:", round(float(ml_loss.cpu()), 2) + float(cl_loss.cpu()), "ML loss", float(ml_loss.cpu()), "CL loss:", float(cl_loss.cpu()))
            triplet_loss = 0.0
            if epoch % update_triplet == 0:
                for tri_batch_idx in range(tri_num_batch):
                    px1 = X[anchor[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    px2 = X[positive[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    px3 = X[negative[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    pbatch1 = p[anchor[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[positive[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    pbatch3 = p[negative[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    inputs3 = Variable(px3)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    target3 = Variable(pbatch3)
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    z3, q3, xr3 = self.forward(inputs3)
                    loss = self.triplet_loss(q1, q2, q3, 0.1)
                    triplet_loss += loss.data
                    loss.backward()
                    optimizer.step()
            if tri_num_batch > 0:
                print("Triplet Loss:", triplet_loss)
        return final_acc, final_nmi, final_epoch
