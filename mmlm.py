# multi-memory matching
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import numpy as np


def multi_memory_learning_matching(datapath,datasetname):
    if datasetname=='sysu':
        print("Loading SYSU Pseudo-labels")
        GT_rgb = np.load(datapath+ '/train_rgb_resized_label.npy')
        GT_ir = np.load(datapath + '/train_ir_resized_label.npy')
        pseudo_labels_rgb = np.load('./labelfile/SYSU_Baseline_pseudo_labels_rgb.npy')
        pseudo_labels_ir = np.load('./labelfile/SYSU_Baseline_pseudo_labels_ir.npy')
        GT_all_label = np.concatenate((GT_rgb, GT_ir), axis=0)
        print("Loading Baseline Features")
        features_rgb= np.load('./labelfile/SYSU_Baseline_features_rgb.npy')
        features_ir= np.load('./labelfile/SYSU_Baseline_features_ir.npy')


        rgb_indexs = []
        ir_indexs = []
        rgb_centers = []
        ir_centers = []
        # rgb_label_set = set(pseudo_labels_rgb)
        # ir_label_set = set(pseudo_labels_ir)
        rgb_label_set = {label for label in set(pseudo_labels_rgb) if label != -1}
        ir_label_set = {label for label in set(pseudo_labels_ir) if label != -1}
        for i in range(len(rgb_label_set) - 1):
            indices = np.where(pseudo_labels_rgb == i)
            rgb_indexs.append(indices)

        for i in range(len(ir_label_set) - 1):
            indices = np.where(pseudo_labels_ir == i)
            ir_indexs.append(indices)
        print("Multi Memory Lerning")

        for i,rgb_index in enumerate(rgb_indexs):
            if i%50==0:
                print("Sub_cluster rgb {}/{}".format(i,len(rgb_indexs)))

            rgb_id_feature = features_rgb[rgb_index]
            try:
                kmeans = KMeans(n_clusters=4, random_state=0)
                # 进行聚类
                clusters = kmeans.fit_predict(rgb_id_feature)
                rgb_center = kmeans.cluster_centers_
            except:
                rgb_center=rgb_id_feature.mean(axis=0)
            rgb_centers.append(rgb_center)

        for j,ir_index in enumerate(ir_indexs):
            if j%50==0:
                print("Sub_cluster ir {}/{}".format(j,len(ir_indexs)))

            ir_id_feature = features_ir[ir_index]
            try:
                ir_kmeans = KMeans(n_clusters=4, random_state=0)
                # 进行聚类
                ir_clusters = ir_kmeans.fit_predict(ir_id_feature)
                ir_center = ir_kmeans.cluster_centers_
            except:
                ir_center = ir_id_feature.mean(axis=0)
            ir_centers.append(ir_center)

        print("Multi Memory Matching")

        for rgb_index in range(len(rgb_centers)):
            rgb_center = rgb_centers[rgb_index]
            dis_max = 20
            k = 0
            for center in ir_centers:
                distances = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        distances[i, j] = np.linalg.norm(center[i] - rgb_center[j])
                min_values = np.min(distances, axis=1)
                dis = np.sum(min_values)
                if dis < dis_max:
                    dis_max = dis
                    aligned_index = k
                k = k + 1
            fin_ir = ir_indexs[aligned_index]
            fin_rgb = rgb_indexs[rgb_index]

            rgb_pl = pseudo_labels_rgb[fin_rgb]
            ir_pl = pseudo_labels_ir[fin_ir]
            pseudo_labels_rgb[fin_rgb] = ir_pl[0]

        ari_rgb = adjusted_rand_score(pseudo_labels_rgb, GT_rgb)
        ari_ir = adjusted_rand_score(pseudo_labels_ir, GT_ir)
        PL_all_label = np.concatenate((pseudo_labels_rgb, pseudo_labels_ir), axis=0)
        ari_all = adjusted_rand_score(PL_all_label, GT_all_label)
        np.save('./labelfile/SYSU_MMM_pseudo_labels_rgb.npy', pseudo_labels_rgb)
        np.save('./labelfile/SYSU_MMM_pseudo_labels_ir.npy', pseudo_labels_ir)
        print('ari_rgb', ari_rgb)
        print('ari_ir', ari_ir)
        print('ari_all', ari_all)
        return pseudo_labels_rgb, pseudo_labels_ir
    else:
        print("Loading RedDB Pseudo-labels")
        pseudo_labels_rgb = np.load('./labelfile/RegDB_Baseline_pseudo_labels_rgb.npy')
        pseudo_labels_ir = np.load('./labelfile/RegDB_Baseline_pseudo_labels_ir.npy')
        print("Loading Baseline Features")
        features_rgb = np.load('./labelfile/RegDB_Baseline_features_rgb.npy')
        features_ir = np.load('./labelfile/RegDB_Baseline_features_ir.npy')

        rgb_indexs = []
        ir_indexs = []
        rgb_centers = []
        ir_centers = []

        rgb_label_set = {label for label in set(pseudo_labels_rgb) if label != -1}
        ir_label_set = {label for label in set(pseudo_labels_ir) if label != -1}
        for i in range(len(rgb_label_set) - 1):
            indices = np.where(pseudo_labels_rgb == i)
            rgb_indexs.append(indices)

        for i in range(len(ir_label_set) - 1):
            indices = np.where(pseudo_labels_ir == i)
            ir_indexs.append(indices)
        print("Multi Memory Lerning")

        for i, rgb_index in enumerate(rgb_indexs):
            if i % 50 == 0:
                print("Sub_cluster rgb {}/{}".format(i, len(rgb_indexs)))

            rgb_id_feature = features_rgb[rgb_index]
            try:
                kmeans = KMeans(n_clusters=4, random_state=0)
                # 进行聚类
                clusters = kmeans.fit_predict(rgb_id_feature)
                rgb_center = kmeans.cluster_centers_
            except:
                rgb_center = rgb_id_feature.mean(axis=0)
            rgb_centers.append(rgb_center)

        for j, ir_index in enumerate(ir_indexs):
            if j % 50 == 0:
                print("Sub_cluster ir {}/{}".format(j, len(ir_indexs)))

            ir_id_feature = features_ir[ir_index]
            try:
                ir_kmeans = KMeans(n_clusters=4, random_state=0)
                # 进行聚类
                ir_clusters = ir_kmeans.fit_predict(ir_id_feature)
                ir_center = ir_kmeans.cluster_centers_
            except:
                ir_center = ir_id_feature.mean(axis=0)
            ir_centers.append(ir_center)

        print("Multi Memory Matching")

        for rgb_index in range(len(rgb_centers)):
            rgb_center = rgb_centers[rgb_index]
            dis_max = 20
            k = 0
            for center in ir_centers:
                distances = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        distances[i, j] = np.linalg.norm(center[i] - rgb_center[j])
                min_values = np.min(distances, axis=1)
                dis = np.sum(min_values)
                if dis < dis_max:
                    dis_max = dis
                    aligned_index = k
                k = k + 1
            fin_ir = ir_indexs[aligned_index]
            fin_rgb = rgb_indexs[rgb_index]

            rgb_pl = pseudo_labels_rgb[fin_rgb]
            ir_pl = pseudo_labels_ir[fin_ir]
            pseudo_labels_rgb[fin_rgb] = ir_pl[0]

        np.save('./labelfile/RegDB_MMM_pseudo_labels_rgb.npy', pseudo_labels_rgb)
        np.save('./labelfile/RegDB_MMM_pseudo_labels_ir.npy', pseudo_labels_ir)
        return pseudo_labels_rgb, pseudo_labels_ir

if __name__ == '__main__':
    multi_memory_learning_matching('/data/yxb/datasets/ReIDData/SYSU-MM01/')
