import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import matplotlib.pyplot as plt
import random
import time
import faiss
import torch.nn.functional as F

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)
# from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
#                             index_init_gpu, index_init_cpu
def swig_ptr_from_FloatTensor(x):
    x=x.to(torch.float64)
    assert x.is_contiguous()
    assert x.dtype == torch.float64
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I

def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_FloatTensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    faiss.bruteForceKnn(res, metric,
                xb_ptr, xb_row_major, nb,
                xq_ptr, xq_row_major, nq,
                d, k, D_ptr, I_ptr)

    return D, I

def index_init_gpu(ngpus, feat_dim):
    flat_config = []
    for i in range(ngpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    indexes = [faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i]) for i in range(ngpus)]
    index = faiss.IndexShards(feat_dim)
    for sub_index in indexes:
        index.add_shard(sub_index)
    index.reset()
    return index

def index_init_cpu(feat_dim):
    return faiss.IndexFlatL2(feat_dim)
def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]
def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1,N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print ("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                while len(color_pos[batch_idx[i]]) < 4 or len(thermal_pos[batch_idx[i]]) < 4:
                    batch_idx[i] = np.random.choice(uni_label, 1, replace=False)
                    # print("re-sampling")
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad


class AllSampler(Sampler):
    def __init__(self, dataset, train_color_label, train_thermal_label, shuffle=True):
        N1 = len(train_color_label)
        N2 = len(train_thermal_label)
        # N = np.maximum(len(train_color_label), len(train_thermal_label))
        if dataset == 'regdb':
            index1 = np.concatenate((np.arange(N1), np.arange(20)))
            index2 = np.concatenate((np.arange(N2), np.arange(20)))
        else:
            index1 = np.concatenate((np.arange(N1), np.arange(14)))
            index2 = np.concatenate((np.arange(N2), np.arange(N1 - N2 + 14)))

        if shuffle:
            np.random.shuffle(index1)
            np.random.shuffle(index2)

        self.index1 = index1
        self.index2 = index2
        self.N = len(index1)

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
        uni_label_rgb = np.unique(train_color_label)
        uni_label_ir = np.unique(train_thermal_label)
        self.n_classes = max(len(uni_label_rgb), len(uni_label_ir))
        # print(uni_label_rgb)
        # print(uni_label_ir)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        label_rgb_index = np.arange(len(uni_label_rgb))
        #label_ir_index = np.arange(len(uni_label_ir))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx_rgb = np.random.choice(label_rgb_index, batchSize, replace=False)
            #batch_idx_ir = np.random.choice(label_ir_index, batchSize, replace=False)
            for i in range(batchSize):
                while len(color_pos[batch_idx_rgb[i]]) < num_pos or len(thermal_pos[uni_label_rgb[batch_idx_rgb[i]]]) < num_pos:
                #while len(color_pos[uni_label_ir[batch_idx_ir[i]]]) < num_pos or len(thermal_pos[batch_idx_ir[i]]) < num_pos:
                    batch_idx_rgb[i] = np.random.choice(label_rgb_index, 1, replace=False)
                    #batch_idx_ir[i] = np.random.choice(label_ir_index, 1, replace=False)
                    print("re-sampling")
                #sample_thermal = np.random.choice(thermal_pos[batch_idx_ir[i]], num_pos)
                #sample_color = np.random.choice(color_pos[uni_label_ir[batch_idx_ir[i]]], num_pos)
                sample_color = np.random.choice(color_pos[batch_idx_rgb[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[uni_label_rgb[batch_idx_rgb[i]]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
    

# class IdentitySampler(Sampler):
#     """Sample person identities evenly in each batch.
#         Args:
#             train_color_label, train_thermal_label: labels of two modalities
#             color_pos, thermal_pos: positions of each identity
#             batchSize: batch size
#     """
#
#     def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
#         uni_label = np.unique(train_color_label)
#         self.n_classes = len(uni_label)
#
#         N = np.maximum(len(train_color_label), len(train_thermal_label))
#         for j in range(int(N / (batchSize * num_pos)) + 1):
#             batch_idx = np.random.choice(uni_label, batchSize, replace=False)
#             for i in range(batchSize):
#                 sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
#                 sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
#
#                 if j == 0 and i == 0:
#                     index1 = sample_color
#                     index2 = sample_thermal
#                 else:
#                     index1 = np.hstack((index1, sample_color))
#                     index2 = np.hstack((index2, sample_thermal))
#
#         self.index1 = index1
#         self.index2 = index2
#         self.N = N
#
#     def __iter__(self):
#         return iter(np.arange(len(self.index1)))
#
#     def __len__(self):
#         return self.N