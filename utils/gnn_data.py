import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.GNN_TARnet_hyper import *
# from models.GAT_TARnet_hyper import *
from models.TARnet_hyper import *
from models.SLearner_hyper import *
from hyperparameters import *
import scipy.stats
import shutil
from models.CausalModel import *
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import json
import lingam
from lingam.utils import make_prior_knowledge, make_dot
from causalnex.structure.notears import from_numpy


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def find_edges(i, datset_name):
    params = {'dataset_name': datset_name, 'num': 100, 'binary': False, 'num_train': None}
    model = CausalModel(params)
    kwargs = {'count': i}
    data_train, data_test = model.load_data(**kwargs)
    # data_train = pd.read_csv("JOBS" + '/jobs_train_' + str(i) + '.csv', delimiter=',')

    y = data_train['y']
    x = data_train['x']

    data = np.concatenate([x, y], axis=1)

    # https://causalnex.readthedocs.io/en/latest/03_tutorial/01_first_tutorial.html

    num_y = data.shape[1] - 1
    sm = from_numpy(data, tabu_parent_nodes=[num_y])
    if datset_name == "jobs":
        sm.remove_edges_below_threshold(0.05)
    elif datset_name == "ihdp_b":
        sm.remove_edges_below_threshold(0.99)
    else:
        sm.remove_edges_below_threshold(0.5)
    influence_y = np.asarray(list(sm.in_edges(num_y)))[:, 0]
    # remove edges from nodes to y
    sm.remove_edges_from(list(sm.in_edges(num_y)))
    # get the final edges
    edges = np.asarray(list(sm.edges))

    return edges, influence_y

import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_prior_knowledge, make_dot

def find_edges_lingam(i, datset_name):
    params = {'dataset_name': datset_name, 'num': 100, 'binary': False, 'num_train': None}
    model = CausalModel(params)
    kwargs = {'count': i}
    data_train, data_test = model.load_data(**kwargs)
    # data_train = pd.read_csv("JOBS" + '/jobs_train_' + str(i) + '.csv', delimiter=',')

    y = data_train['y']
    x = data_train['x']

    data = np.concatenate([x, y], axis=1)

    # https://causalnex.readthedocs.io/en/latest/03_tutorial/01_first_tutorial.html
    n_variables = data.shape[1]
    prior_knowledge = make_prior_knowledge(
        n_variables=n_variables,
        sink_variables=[n_variables - 1],
    )
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(data)
    influence_y = np.asarray(np.nonzero(np.transpose(model.adjacency_matrix_[n_variables-1, :])))
    # remove edges from nodes to y
    # edges = np.transpose(np.asarray(np.nonzero(np.transpose(model.adjacency_matrix_[:n_variables-1, :n_variables-1]))))
    edges = np.transpose(np.asarray(np.nonzero(np.transpose(model.adjacency_matrix_))))

    edge_weights = model.adjacency_matrix_[edges[:, 1], edges[:, 0]]
    if np.squeeze(influence_y).size == 0:
        acyclic_W = np.eye(x.shape[1])
        graph = np.asarray(np.nonzero(acyclic_W))
        edges = np.transpose(graph)
        influence_y = np.arange(x.shape[1])
        edge_weights = np.ones(shape=(edges.shape[0]))
        edge_weights = np.expand_dims(edge_weights / np.sum(edge_weights, axis=0), axis=-1)

    return edges, influence_y, edge_weights



def generate_graphs(dataset_name):
    print('generating graphs for ' + dataset_name)
    for i in range(100):
        folder_path = "graphs/" + dataset_name
        graph_path = "/graph_" + str(i) + '.json'
        graph_exist = exists(folder_path + graph_path)
        if graph_exist:
            continue
        print('creating graph for ' + str(i) + 'th dataset')
        graph, influence_y, edge_weights = find_edges_lingam(i, dataset_name)

        graph_struct = {}
        graph_struct['from'] = graph[:, 0].tolist()
        graph_struct['to'] = graph[:, 1].tolist()
        graph_struct['influence_y'] = influence_y.tolist()
        graph_struct['weights'] = edge_weights.tolist()

        folder_exists = exists(folder_path)
        # check if folder exists
        if not folder_exists:
            os.makedirs(folder_path)

        # data = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in data.items()}

        with open(folder_path + graph_path, "w") as file:
            json.dump(graph_struct, file)
    print('graphs for', dataset_name, 'are created')


def run_gnn_on_sum(eye, num, num_layers):
    model_name_s = "GNNTARnet"
    dataset_name = "sum"
    params = find_params(model_name_s, dataset_name)
    model_name = GNNTARnetHyper
    params['model_name'] = model_name_s
    params['dataset_name'] = dataset_name
    params['ipm_type'] = "None"
    params['defaults'] = "True"
    params['tuner'] = kt.RandomSearch
    params['tuner_name'] = 'random'
    params['num'] = num
    params['eye'] = eval(eye)
    params['num_layers'] = num_layers

    # create new sum dataset if it doesn't exist
    file_exists_gnn = exists('SUM_' + str(num_layers))
    if not file_exists_gnn:
        # create the sum dataset
        generate_sum_dataset(num_layers=num_layers)

    # sum_sizes = {500}
    data = pd.DataFrame()

    model = model_name(params)
    if params['eye']:
        file_name = 'results/result_eye_' + str(num_layers) + '.csv'
    else:
        file_name = 'results/result_' + str(num_layers) + '.csv'

    train_sizes = {16, 32, 64, 128}
    for size in train_sizes:
        file_exists = os.path.isfile(file_name)
        if file_exists:
            data = pd.read_csv(file_name)
        if str(size) in data.columns:
            continue
        else:
            print('Chosen model is', model_name_s, dataset_name, "size:", size, 'default:', params['defaults'],
                  'eye:', params['eye'], 'num_layers:', str(num_layers))
            model.num_train = size
            metric_list_train, metric_list_test, average_train, average_test = model.evaluate_performance()
            m_test, h_test = mean_confidence_interval(metric_list_test, confidence=0.95)
            # find t value
            print('pehe test:', m_test, '+-', h_test)
            data[str(size)] = metric_list_test
            data.to_csv(file_name, index=False)


    print('You already finished the computing! Check the results.')
    return

def run_model(results_path, args):
    model_names = {"TARnet": TARnetHyper,
                   "GNNTARnet": GNNTARnetHyper,
                   "TLearner": TLearner, "CFRNet": CFRNet,
                   "GANITE": GANITE, "SLearner": SLearner, "TEDVAE": TEDVAE
                   }
    tuners = {'random': kt.RandomSearch, 'bayesian': kt.BayesianOptimization, 'hyperband': kt.Hyperband}
    params = find_params(args.model_name, args.dataset_name)
    model_name = model_names[args.model_name]
    params['model_name'] = args.model_name
    params['dataset_name'] = args.dataset_name
    params['ipm_type'] = args.ipm_type
    params['defaults'] = eval(args.defaults)
    params['num'] = args.num
    params['drop'] = args.drop
    params['tuner'] = tuners[args.tuner_name]
    params['tuner_name'] = args.tuner_name
    params['num_layers'] = args.num_layers
    params['num_train'] = args.num_train
    params['eye'] = eval(args.eye)

    # sum_sizes = {500}
    data = pd.DataFrame()

    model = model_name(params)
    if args.model_name == "GNNTARnet" and params['eye']:
        file_name = results_path + '/result_eye_GNNTARnet_' + str(args.dataset_name) + '.csv'
    else:
        file_name = results_path + '/result_' + str(args.model_name) + '_' + args.dataset_name + '.csv'

    train_sizes = {16, 32, 64, 128, 256, 512}
    train_sizes = {None}
    for size in train_sizes:
        file_exists = os.path.isfile(file_name)
        if file_exists:
            data = pd.read_csv(file_name)
        if str(size) in data.columns:
            continue
        else:
            print('Chosen model is', args.model_name, args.dataset_name, "size:", size, 'default:', params['defaults'],
                  'eye:', params['eye'])
            model.num_train = size
            metric_list_train, metric_list_test, average_train, average_test = model.evaluate_performance()
            m_test, h_test = mean_confidence_interval(metric_list_test, confidence=0.95)
            # find t value
            print('pehe test:', m_test, '+-', h_test)
            data[str(size)] = metric_list_test
            data.to_csv(file_name, index=False)

    print('You already finished the computing! Check the results.')
    return

def run_tarnet_on_sum(num, num_layers):
    model_name_s = "TARnet"
    dataset_name = 'sum'
    params = find_params(model_name_s, dataset_name)
    model_name = TARnetHyper
    params['model_name'] = model_name_s
    params['dataset_name'] = dataset_name
    params['ipm_type'] = "None"
    params['defaults'] = True
    params['tuner'] = kt.RandomSearch
    params['tuner_name'] = 'random'
    params['num'] = num
    params['eye'] = False
    params['batch_size'] = 32
    params['num_layers'] = num_layers

    # create new sum dataset if it doesn't exist
    file_exists_gnn = exists('SUM_' + str(num_layers))
    if not file_exists_gnn:
        # create the sum dataset
        generate_sum_dataset(num_layers=num_layers)

    # sum_sizes = {500}
    data = pd.DataFrame()

    model = model_name(params)
    file_name = 'results/result_tarnet_' + str(num_layers) + '.csv'

    sum_sizes = {16, 32, 64, 128}
    for size in sum_sizes:
        file_exists = os.path.isfile(file_name)
        if file_exists:
            data = pd.read_csv(file_name)
        if str(size) in data.columns:
            continue
        else:
            print('Chosen model is', model_name_s, dataset_name, "size:", size, 'default:', params['defaults'],
                  'eye:', params['eye'], 'num_layers:', str(num_layers))
            model.num_train = size
            metric_list_train, metric_list_test, average_train, average_test = model.evaluate_performance()
            m_test, h_test = mean_confidence_interval(metric_list_test, confidence=0.95)
            # find t value
            print('pehe test:', m_test, '+-', h_test)
            data[str(size)] = metric_list_test
            data.to_csv(file_name, index=False)

    print('You already finished the computing! Check the results.')
    return


from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
from scipy.stats import ttest_ind


def process_data(dataset_name, level, tarnet=False):
    layers = [1, 2, 3, 4]
    # read gnn4ite results eye
    file_name_eye = 'results/result_eye_' + str(level) + '.csv'
    results_eye = pd.read_csv(file_name_eye)
    # read gnn4ite results
    file_name = 'results/result_' + str(level) + '.csv'
    results = pd.read_csv(file_name)
    # read for tarnet
    file_name = 'results/result_tarnet_' + str(level) + '.csv'
    results_tarnet = pd.read_csv(file_name)

    pehe_struct = {"1": [], "2": [], "3": [], "4": []}
    pehe_eye_struct = {"1": [], "2": [], "3": [], "4": []}
    if not tarnet:
        sum_sizes = {16, 32, 64, 128}
        # sum_sizes = {256}
        m_tests = np.zeros([len(sum_sizes)])
        m_tests_eye = np.zeros([len(sum_sizes)])

        h_tests = np.zeros([len(sum_sizes)])
        h_tests_eye = np.zeros([len(sum_sizes)])
        for i, j in enumerate(sum_sizes):
            data_eye = results_eye[str(j)]
            data = results[str(j)]
            m_test_eye, h_test_eye = mean_confidence_interval(data_eye, confidence=0.95)
            m_test, h_test = mean_confidence_interval(data, confidence=0.95)

            m_tests[i] = m_test
            m_tests_eye[i] = m_test_eye
            h_tests[i] = h_test
            h_tests_eye[i] = h_test_eye

            # find t value
            print('Data size:', j)
            print('pehe eye:', m_test_eye, '+-', h_test_eye)
            print('pehe:', m_test, '+-', h_test)
            # Assuming you have two arrays of data: data1 and data2
            stat, p_value = ks_2samp(data, data_eye)

            print("Kolmogorov-Smirnov test statistic:", stat)
            print("p-value:", p_value)
            stat, p_value = ttest_ind(data_eye, data)
            print("t-test statistic:", stat)
            print("p-value:", p_value)
            result = anderson_ksamp([data_eye, data])
            print("Anderson-Darling test statistic:", result.statistic)
            print("p-values:", result.significance_level, '\n')

        return m_tests, m_tests_eye, h_tests, h_tests_eye
    else:
        sum_sizes = {16, 32, 64, 128}
        # sum_sizes = {256}
        m_tests = np.zeros([len(sum_sizes)])
        h_tests = np.zeros([len(sum_sizes)])
        for i, j in enumerate(sum_sizes):
            data = results_tarnet[str(j)]
            m_test, h_test = mean_confidence_interval(data, confidence=0.95)
            # find t value
            print('Data size:', j, 'level:', level)
            print('pehe:', m_test, '+-', h_test)
            m_tests[i] = m_test
            h_tests[i] = h_test
        return m_tests, h_tests



# Define a function with a delta distribution (deterministic)
def deterministic_function(parent_value):
    return parent_value


# Define a function with a wider delta distribution (less deterministic)
def probabilistic_function(parent_value, delta):
    return random.randint(int(parent_value - delta), int(parent_value + delta))


# Define interactive behavior for nodes
def interactive_behavior(sel):
    node = sel.target.get_text()
    print(f"Clicked Node {node}")


def find_parents(graph, node):
    parents = []
    for parent in graph.keys():
        if node in graph[parent]:
            parents.append(parent)
    return parents


def generate_layers(num_layers):
    num_parent_nodes = random.randint(10, 20)
    parent_nodes = np.arange(0, num_parent_nodes)
    layer_nodes = {}
    for n in range(num_layers):
        if n == num_layers - 1:
            num_out = parent_nodes[-1] + 1
            num_child_nodes = int(0.3 * num_out)
        else:
            num_child_nodes = random.randint(3, 8)

        child_nodes = np.arange(parent_nodes[-1] + 1, parent_nodes[-1] + 1 + num_child_nodes)
        layer_nodes[str(n)] = parent_nodes.tolist()
        parent_nodes = child_nodes
    return layer_nodes


def create_layered_graph(num, num_layers=3):
    # layer 1 has a random number of root nodes
    # layer 2...num_layers-1 has a random number of nodes
    # layer num_layers has a random number of leaf nodes
    # each node in layer i has a random number of parents in layer i-1
    num_parent_nodes = random.randint(10, 20)
    parent_nodes = np.arange(0, num_parent_nodes)
    # child_nodes = np.arange(num_parent_nodes, num_parent_nodes + num_child_nodes)
    # connect parent nodes to child nodes
    graph = {node: [] for node in parent_nodes}
    # store nodes in each layer
    layer_nodes = {}

    for n in range(num_layers):
        graph_layer = {node: [] for node in parent_nodes}
        # we don't want to have too many nodes in the last layer
        if n == num_layers - 1:
            num_out = parent_nodes[-1] + 1
            num_child_nodes = int(0.3 * num_out)
        else:
            num_child_nodes = random.randint(3, 8)

        child_nodes = np.arange(parent_nodes[-1] + 1, parent_nodes[-1] + 1 + num_child_nodes)

        for parent in parent_nodes:
            num_children = random.randint(1, num_child_nodes)
            children = random.sample(list(child_nodes), num_children)
            for child in children:
                graph_layer[parent].append(int(child))
        layer_nodes[str(n)] = parent_nodes.tolist()

        parent_nodes = child_nodes
        graph.update(graph_layer)

    layer_nodes[str(num_layers)] = parent_nodes.tolist()

    plot_graph(graph, num)
    return graph, layer_nodes


def plot_graph(graph, num):
    # Create a directed graph using NetworkX
    dag = nx.DiGraph(graph)
    # Obtain topological ordering
    node_order = list(nx.topological_sort(dag))
    # Draw the DAG using NetworkX and Matplotlib with sorted nodes
    pos = nx.spring_layout(dag, scale=5, k=2)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(dag, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, arrows=True,
                     nodelist=node_order)
    plt.title("Random DAG Visualization (Sorted)" + str(num))
    plt.show()


def save_graph(graph, path, layer_nodes=None):
    new_graph = {str(key): value for key, value in graph.items()}
    graph_dict = {'graph': new_graph}
    graph_dict['layer_nodes'] = layer_nodes
    with open(path, 'w') as f:
        json.dump(graph_dict, f)


def load_graph(path):
    with open(path, 'r') as f:
        graph_dict = json.load(f)
    return graph_dict


def generate_deterministic_data(num, num_layers=3):
    # if graph exists, use load it, else create it
    graph_path = 'graphs/sum_graph_raw_' + str(num_layers)
    file_path = "/graph_" + str(num) + '.json'
    file_exists = exists(graph_path + file_path)
    if file_exists:
        print("Loading graph ", num)
        # load graph
        laded_graph = load_graph(graph_path + file_path)
        graph = laded_graph['graph']
        layer_nodes = laded_graph['layer_nodes']
        num_loaded_layers = len(layer_nodes.keys()) - 1
        plot_graph(graph, num)
        if num_loaded_layers != num_layers:
            print("Need to create a new graph with ", num_layers, " layers ", num)
            # Generate multiple datasets with random graphs
            graph, layer_nodes = create_layered_graph(num_layers=num_layers, num=num)
            # save graph
            save_graph(graph, graph_path + file_path, layer_nodes=layer_nodes)
            plot_graph(graph, num)
    else:
        print("Creating graph ", num)
        if not exists(graph_path):
            os.makedirs(graph_path)
        # Generate multiple datasets with random graphs
        graph, layer_nodes = create_layered_graph(num_layers=num_layers, num=num)
        # save graph
        save_graph(graph, graph_path + file_path, layer_nodes=layer_nodes)

    dataset = get_data_from_graph(graph, layer_nodes, index=0)
    for i in range(600):
        dataset = dataset.append(get_data_from_graph(graph, layer_nodes, index=i + 1))

    return dataset, graph, layer_nodes


def generate_sum_dataset(num_layers=3):
    for num in range(100):
        # Generate deterministic data
        data, graph, layer_nodes = generate_deterministic_data(num, num_layers)
        graph = {int(k): [int(i) for i in v] for k, v in graph.items()}
        data = data.values
        num_nodes = data.shape[1]
        # Generate outcome data
        # Find nodes without children
        nodes_without_children = np.asarray(list(layer_nodes[str(int(len(layer_nodes) - 1))]))
        # Generate outcome data
        # flatten the graph
        flat_graph = [[i, j] for i in graph for j in graph[i]]
        # select weights for outcome

        mean_0 = np.sum(data[:, nodes_without_children], axis=1)
        mean_1 = np.mean(data[:, nodes_without_children], axis=1)

        mu_0 = np.expand_dims(mean_0, axis=-1)
        mu_1 = np.expand_dims(mean_1, axis=-1)

        # select nodes defining treatment
        influence_t = nodes_without_children
        data_influence_t = data[:, influence_t]
        mean_t = np.mean(data_influence_t, axis=1)

        mu_t = np.expand_dims(mean_t, axis=-1)
        mean_t = np.mean(mu_t)
        # create treatment
        t = np.zeros((data.shape[0], 1))
        t_1 = mu_t > mean_t
        t[t_1] = 1

        y = np.zeros((data.shape[0], 1))
        t_0 = t == 0
        t_1 = t == 1
        y[t_0] = mu_0[t_0]
        y[t_1] = mu_1[t_1]

        # the idea is to mask certain layers in the graph
        # and then to train the model on the masked graph
        # and then to test the model on the masked graph
        # but first we need to select the nodes that we want to mask at random
        # for that use np.arange from 1 to num_layers - 1
        # then select at random number of nodes to mask from the nodes in the layer
        # then mask the nodes in the layer
        # we also need to update the graph
        # if nodes are not masked, then we need to remove the edges that are connected to the not masked nodes
        # if nodes are masked, then we need to keep the edges that are connected to the masked nodes

        # select the last layer
        selected_layers = [num_layers]

        # select nodes to mask
        nodes_to_mask = []
        for layer in selected_layers:
            nodes_in_layer = layer_nodes[str(layer)]
            nodes_to_mask.append(nodes_in_layer)
        nodes_to_mask = np.concatenate(nodes_to_mask)
        # # add some random nodes to mask from the previous layers

        # make graph to follow children: parents structure
        # chilren: parents graph
        children_parents_graph = {}
        for i in range(num_nodes):
            parents = find_parents(graph, i)
            children_parents_graph[i] = parents

        # # this makes nodes in the not last layer to be parents of themselves
        # # only the nodes in the last layer are not parents of themselves
        # this is an alternative of using the whole graph
        # Flatten the graph
        flat_graph = np.asarray(flat_graph)

        graph = flat_graph.astype(int)

        data = np.concatenate([t, y, mu_0, mu_1, data], axis=1)
        data_train, data_test = train_test_split(data, test_size=0.2)

        t_train = pd.DataFrame(data_train[:, 0], columns=["t"])
        y_train = pd.DataFrame(data_train[:, 1], columns=["y"])
        mu_0_train = pd.DataFrame(data_train[:, 2], columns=["mu_0"])
        mu_1_train = pd.DataFrame(data_train[:, 3], columns=["mu_1"])
        x_train = pd.DataFrame(data_train[:, 4:])

        t_test = pd.DataFrame(data_test[:, 0], columns=["t"])
        y_test = pd.DataFrame(data_test[:, 1], columns=["y"])
        mu_0_test = pd.DataFrame(data_test[:, 2], columns=["mu_0"])
        mu_1_test = pd.DataFrame(data_test[:, 3], columns=["mu_1"])
        x_test = pd.DataFrame(data_test[:, 4:])

        # do some modifications to the data
        # make some nodes missing

        x_train.iloc[:, nodes_to_mask] = 0
        x_test.iloc[:, nodes_to_mask] = 0

        data_test = pd.DataFrame(pd.concat([t_test, y_test, mu_0_test, mu_1_test, x_test], axis=1))
        data_train = pd.DataFrame(pd.concat([t_train, y_train, mu_0_train, mu_1_train, x_train], axis=1))

        new_path = "SUM_" + str(num_layers)
        file_exists_gnn = exists(new_path)

        # check if folder exists
        if not file_exists_gnn:
            os.makedirs(new_path)

        path_train = new_path + "/sum_train_" + str(num) + '.csv'
        path_test = new_path + "/sum_test_" + str(num) + '.csv'

        data_train.to_csv(path_train, index=False)
        data_test.to_csv(path_test, index=False)

        graph_path = "/graph_" + str(num) + '.json'

        graph_struct = {}
        graph_struct['from'] = graph[:, 0].tolist()
        graph_struct['to'] = graph[:, 1].tolist()
        graph_struct['influence_y'] = np.asarray(list(layer_nodes[str(int(len(layer_nodes) - 1))])).tolist()
        graph_struct['nodes_to_mask'] = nodes_to_mask.tolist()
        folder_path = "graphs/sum_graph_" + str(num_layers)
        folder_exists = exists(folder_path)
        # check if folder exists
        if not folder_exists:
            os.makedirs(folder_path)

        with open(folder_path + graph_path, "w") as file:
            json.dump(graph_struct, file)


def get_data_from_graph(graph, layer_nodes, index):
    dataset = {}
    # find total number of nodes
    num_nodes = 0
    for layer in layer_nodes:
        num_nodes = num_nodes + len(layer_nodes[layer])
    for node in range(num_nodes):
        # find children
        parents = find_parents(graph, node)
        if len(parents) == 0:
            dataset[str(node)] = random.random()
        else:
            parent_values = [dataset.get(str(parent), 0) for parent in parents]
            dataset[str(node)] = sum(parent_values)
    dataset = pd.DataFrame(dataset, index=[index])
    return dataset

