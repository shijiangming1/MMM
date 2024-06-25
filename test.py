from __future__ import print_function
import argparse
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from datasets.data_loader import SYSUData, RegDBData, TestData
from datasets.data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils.utils import *

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--resume-net1', default='/', type=str,
                    help='resume net1 from checkpoint')

parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model-path', default='./save_model/', type=str,
                    help='model save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='dart', type=str,
                    metavar='m', help='method type: base or dart')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='3', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='indoor', type=str, help='all or indoor for sysu')
parser.add_argument('--shot', default=1, type=int, help='single or multi shot for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
parser.add_argument('--data-dir', default='/', type=str, help='path to dataset')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
data_path = args.data_dir
if dataset == 'sysu':
    pseudo_label=np.load("./labelfile/SYSU_MMM_pseudo_labels_ir.npy")
    n_class = len(set(pseudo_label))-1
    # n_class = 395
    test_mode = [1, 2]
elif dataset == 'regdb':
    pseudo_label_ir = np.load("./labelfile/RegDB_MMM_pseudo_labels_ir.npy")
    pseudo_label_rgb = np.load("./labelfile/RegDB_MMM_pseudo_labels_rgb.npy")
    n_class = max(len(set(pseudo_label_ir)) ,len(set(pseudo_label_rgb)))
    # n_class = 206
    test_mode = [2, 1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
pool_dim = 2048
print('==> Building model..')
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
else:
    net1 = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
net1.to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def extract_gall_feat(gall_loader, net):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_fc = net(input, input, test_mode[0])
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_fc


def extract_query_feat(query_loader, net):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_fc = net(input, input, test_mode[1])
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_fc


if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume_net1) > 0:
        model_path1 = checkpoint_path + args.resume_net1
        if os.path.isfile(model_path1):
            print('==> loading checkpoint {}'.format(args.resume_net1))
            checkpoint1 = torch.load(model_path1)
            net1.load_state_dict(checkpoint1['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume_net1, checkpoint1['epoch']))
        else:
            print('==> no checkpoint found at {} '.format(args.resume_net1))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    # gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, shot=args.shot)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_fc1 = extract_query_feat(query_loader, net1)
    query_feat_fc = query_feat_fc1
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial, shot=args.shot)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc1 = extract_gall_feat(trial_gall_loader, net1)
        gall_feat_fc = gall_feat_fc1

        if args.tvsearch:
            print('thermal to visible')
            distmat = np.matmul(gall_feat_fc, np.transpose(query_feat_fc))
            cmc, mAP, mINP = eval_sysu(-distmat, gall_label, query_label, gall_cam, query_cam)

        else:
            print('visible to thermal')
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP

        print('Test Trial: {}'.format(trial))
        print(
            'Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))


elif dataset == 'regdb':
    test_trial = args.trial
    # model_path1 = checkpoint_path + (args.resume_net1).format(test_trial)
    # if os.path.isfile(model_path1):
    #     print('==> loading checkpoint {} '.format((args.resume_net1).format(test_trial)))
    #     checkpoint1 = torch.load(model_path1)
    #     net1.load_state_dict(checkpoint1['net'])
    #     print('==> loaded checkpoint {} (epoch {}), {} (epoch {})'
    #           .format((args.resume_net1).format(test_trial), checkpoint1['epoch']))
    # else:
    #     print('==> no checkpoint found at {} '.format(args.resume_net1))

    print('==> Resuming from checkpoint..')
    if len(args.resume_net1) > 0:
        model_path1 = checkpoint_path + args.resume_net1
        if os.path.isfile(model_path1):
            print('==> loading checkpoint {}'.format(args.resume_net1))
            checkpoint1 = torch.load(model_path1)
            net1.load_state_dict(checkpoint1['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume_net1, checkpoint1['epoch']))
        else:
            print('==> no checkpoint found at {} '.format(args.resume_net1))

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    nquery = len(query_label)
    ngall = len(gall_label)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_fc1 = extract_query_feat(query_loader, net1)
    query_feat_fc = query_feat_fc1

    gall_feat_fc1 = extract_gall_feat(gall_loader, net1)
    gall_feat_fc = gall_feat_fc1

    if args.tvsearch:
        print('thermal to visible')
        distmat = np.matmul(gall_feat_fc, np.transpose(query_feat_fc))
        cmc, mAP, mINP = eval_regdb(-distmat, gall_label, query_label)

    else:
        print('visible to thermal')
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
    print('Test Trial: {}'.format(test_trial))
    print(
        'Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

