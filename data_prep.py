import pandas as pd
import numpy as np

def k_mer(seq):
    def get_1mer(seq):
        A_count = seq.count("A")
        T_count = seq.count("T")
        C_count = seq.count("C")
        G_count = seq.count("G")
        return [A_count / len(seq), T_count / len(seq), C_count / len(seq), G_count / len(seq)]

    def get_2mer(seq):
        res_dict = {x + y: 0 for x in "ATCG" for y in "ATCG"}
        for i in range(len(seq) - 1):
            k = seq[i:i + 2]
            if k in res_dict:
                res_dict[k] += 1
        return [x / len(seq) for x in res_dict.values()]

    def get_3mer(seq):
        res_dict = {x + y + z: 0 for x in "ATCG" for y in "ATCG" for z in "ATCG"}
        for i in range(len(seq) - 2):
            k = seq[i:i + 3]
            if k in res_dict:
                res_dict[k] += 1
        return [x / len(seq) for x in res_dict.values()]

    return get_1mer(seq) + get_2mer(seq) + get_3mer(seq)

def lncRNA_mer():
    df = pd.read_excel('../data/dataset2/lncRNA_sequences.xlsx', usecols=['lncRNA_name', 'sequence'])
    df['sequence'] = df['sequence'].str.replace('U', 'T')
    lncRNA_dict = dict(zip(df['lncRNA_name'], df['sequence']))

    result = [k_mer(seq) for seq in lncRNA_dict.values()]

    print("lncRNA features:", len(result))
    np.savetxt("../data/dataset2/lncRNA_mer_feature.txt", result)

def miRNA_mer():
    df = pd.read_excel('../data/dataset2/miRNA_sequences.xlsx', usecols=['miRNA_name', 'Sequence'])
    df['Sequence'] = df['Sequence'].str.replace('U', 'T')
    miRNA_dict = dict(zip(df['miRNA_name'], df['Sequence']))

    result = [k_mer(seq) for seq in miRNA_dict.values()]

    print("miRNA features:", len(result))
    np.savetxt("../data/dataset2/miRNA_mer_feature.txt", result)

def cosine_similarity(features):
    sim_matrix = np.zeros((len(features), len(features)))
    for i in range(len(features)):
        for j in range(i, len(features)):
            v1, v2 = features[i], features[j]
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix

def construct_knn_graph(sim_matrix, top_k=22):
    num_nodes = sim_matrix.shape[0]
    knn_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        sorted_indices = np.argsort(-sim_matrix[i, :])[:top_k]
        for j in sorted_indices:
            knn_matrix[i, j] = 1
            knn_matrix[j, i] = 1

    knn_matrix -= np.diag(np.diag(knn_matrix))
    return knn_matrix

if __name__ == '__main__':
    lncRNA_mer()
    miRNA_mer()

    # lncRNA attribute graph
    lnc_features = np.loadtxt('../data/dataset2/lncRNA_mer_feature.txt')
    lnc_seq_sim = cosine_similarity(lnc_features)
    lnc_att_graph = construct_knn_graph(lnc_seq_sim)
    np.savetxt("../data/dataset1/lnc_att_graph.txt", lnc_att_graph, fmt="%d")

    # miRNA attribute graph
    mi_features = np.loadtxt('../data/dataset2/miRNA_mer_feature.txt')
    mi_seq_sim = cosine_similarity(mi_features)
    mi_att_graph = construct_knn_graph(mi_seq_sim)
    np.savetxt("../data/dataset1/mi_att_graph.txt", mi_att_graph, fmt="%d")

    # disease attribute graph
    dis_sim_matrix = np.loadtxt("../data/dataset1/dis_sem_sim.txt")
    dis_att_graph = construct_knn_graph(dis_sim_matrix)
    np.savetxt("../data/dataset1/dis_att_graph.txt", dis_att_graph, fmt="%d")
