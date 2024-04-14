import os

import matplotlib.pylab as plt
import networkx as nx
import pandas as pd


def get_basic_features(dataset_name, data_dict, setting_dict):
    edge_index_model_to_dataset = data_dict['edge_index_model_to_dataset'].numpy()
    # print(edge_index_model_to_dataset[:4])
    dataset_id = ['dataset_' + str(node) for node in edge_index_model_to_dataset[1, :]]
    model_id = ['model_' + str(node) for node in edge_index_model_to_dataset[0, :]]
    md_elist = list(zip(dataset_id, model_id))
    print(f'len(md_elist): {len(md_elist)}, md_elist: {md_elist[:4]}')

    edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset'].numpy()
    data_node1 = ['dataset_' + str(node) for node in edge_index_dataset_to_dataset[0, :]]
    data_node2 = ['dataset_' + str(node) for node in edge_index_dataset_to_dataset[1, :]]
    ds_elist = list(zip(data_node1, data_node2))
    # print(f'ds_elist:{ds_elist[:4]}')

    unique_model_id = list(set(model_id))
    unique_dataset_id = list(set(dataset_id + data_node1 + data_node2))
    print(f'len_unique_dataset_id:{len(unique_dataset_id)}')
    print(f'len_unique_model_id:{len(unique_model_id)}')

    G = nx.Graph()
    G.add_nodes_from(unique_dataset_id, color="red")
    G.add_nodes_from(unique_model_id, color="blue")
    G.add_edges_from(md_elist)
    G.add_edges_from(ds_elist)

    # plot Graph
    plot(G)

    mapped_dataset_id = data_dict['unique_dataset_id']
    dataset_id = 'dataset_' + str(mapped_dataset_id[mapped_dataset_id['dataset'] == dataset_name]['mappedID'].values[0])
    testing_edges = list(zip(len(unique_model_id) * [dataset_id], unique_model_id))
    print(f'len(testing_edges): {len(testing_edges)}, testing_edges: {list(testing_edges)[:4]}')

    mapped_model_id = data_dict['unique_model_id']
    # testing_edges = [(dataset_id,'model_66')]
    # Feature: Jaccard coefficient
    jaccard_coefficient = nx.jaccard_coefficient(G, testing_edges)
    print()
    print_features('jaccard_coefficient', jaccard_coefficient, dataset_name, mapped_model_id, setting_dict, save=True)

    # Feature: resource_allocation_index
    preds = nx.resource_allocation_index(G, testing_edges)
    print()
    print_features('resource_allocation_index', preds, dataset_name, mapped_model_id, setting_dict, save=True)

    # Feature: preferential_attachment - with the largest number of edges with value
    preds = nx.preferential_attachment(G, testing_edges)
    print()
    print_features('preferential_attachment', preds, dataset_name, mapped_model_id, setting_dict, save=True)

    # Feature: adamic_adar_index
    preds = nx.adamic_adar_index(G, testing_edges)
    print()
    print_features('adamic_adar_index', preds, dataset_name, mapped_model_id, setting_dict, save=True)

    # Feature: Common Neighbor
    print()
    preds = []
    for edge in testing_edges:
        pred = list(nx.common_neighbors(G, edge[0], edge[1]))
        preds.append([edge[0], edge[1], len(pred)])
    print_features('common_neighbor', preds, dataset_name, mapped_model_id, setting_dict, save=True)


def print_features(name, preds, dataset_name, mapped_model_id, setting_dict, save=False):
    print(f'==== {name}')
    print(type(preds))
    results = []
    for u, v, p in preds:
        if p > 0:
            print(f"({u}, {v}) -> {p:.8f}")
        model_name = mapped_model_id[mapped_model_id['mappedID'] == int(v.split('_')[1])]['model'].values[0]
        results.append((model_name, p))
    # print(results[:4])
    # Save graph embedding distance results
    dir_path = os.path.join('./rank', f'{dataset_name}', 'basic')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    config_list = ['accuracy_thres', 'finetune_ratio']
    config_name = '_'.join([('{0}=={1}'.format(k, setting_dict[k])) for k in config_list])
    # np.save(os.path.join(dir_path,config_name+'.npy'),pred)
    pd.DataFrame.from_records(results, columns=['model', 'score']).to_csv(os.path.join(dir_path, f'{name}_{config_name}.csv'))


def plot(G):
    print(f'len(G.nodes):{len(G.nodes)}')
    # print(nx.dijkstra_path(G, unique_dataset_id[0], unique_dataset_id[1]))
    colors = nx.get_node_attributes(G, "color")
    a = [c for c in list(colors.values()) if c == 'red']
    print(f'len(node red): {len(a)}')
    b = [c for c in list(colors.values()) if c == 'blue']
    print(f'len(node blue): {len(b)}')
    options = {"node_size": 50, "alpha": 0.3, "node_color": colors.values()}
    # pos = nx.kamada_kawai_layout(G)
    # pos=nx.fruchterman_reingold_layout(G)
    pos = nx.fruchterman_reingold_layout(G, k=0.5, iterations=100)
    nx.draw(G, pos, **options)
    plt.savefig("simple_graph.png")  # save as png
