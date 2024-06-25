import numpy as np
def resort_list(label_list_pred):
    label_list_pred=list(label_list_pred)
    alone_list = list(set(label_list_pred))
    alone_list.sort()
    update_list = np.arange(len(alone_list))
    print('alone_list',alone_list)
    print('update_list',update_list)
    ii = 0
    for idx1, jj in enumerate(alone_list):
        for idx2, i in enumerate(label_list_pred):
            if i == jj:
                label_list_pred[idx2] = update_list[idx1]

    return label_list_pred
