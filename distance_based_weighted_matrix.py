import numpy as np
from copy import deepcopy


def aspect_oriented_tree(opt, token, head, as_start, as_end):

    stoi = {}
    for i, t in enumerate(token):
        stoi[i] = t
    children = []
    for _ in range(len(token)):
        children += [{}]
    

    for i in range(len(token)):
        for j in range(len(head)):
            if head[j] - 1 == i and j not in children[i].keys() and head[j] != 0:
                children[i][j] = 1
                children[j][i] = 1
        if head[i] - 1 not in children[i].keys() and head[i] != 0:
            children[i][head[i] - 1] = 1
            children[head[i] - 1][i] = 1

    
    children_asp_all = []
    for asp_idx in range(as_start, as_end):
        children_asp = deepcopy(children)
        head_idx = list(children_asp[asp_idx].keys())
        head_stack = deepcopy(head_idx)
        while (len(head_idx) < len(token)) and (len(head_stack) > 0):
            idx_in_sent = head_stack.pop(0)
            ids = list(children_asp[idx_in_sent].keys())
            for idx in ids:
                if idx not in head_idx and idx != asp_idx:
                    children_asp[asp_idx][idx] = children_asp[idx_in_sent][idx] + children_asp[asp_idx][idx_in_sent]
                    head_stack = [idx] + head_stack
                    head_idx += [idx]
        children_asp_all.append(children_asp)

    # distance based weighted matrix
    if 'bert' in opt.model_name:
        dm = np.ones((len(token), len(token))) * (np.inf)
    else:
        dm = np.ones((opt.max_length, opt.max_length)) * (np.inf)
    
    aspect_indices = list(range(as_start, as_end))
    for word_id in range(len(token)):
        distances = [np.inf]
        for child_id, asp_id in enumerate(aspect_indices):
            asp_child = children_asp_all[child_id][asp_id]
            try:
                distances.append(asp_child[word_id])
            except:
                distances.append(np.inf)
        real_distance = min(distances)
        for asp_id in aspect_indices:
            dm[asp_id][word_id] = real_distance
            dm[word_id][asp_id] = real_distance
    for asp_id in aspect_indices:
        for asp_mutual in aspect_indices:
            dm[asp_id][asp_mutual] = 1

    # self-loop
    for i in range(len(dm)):
        dm[i][i] = 1

    return dm
