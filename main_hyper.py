from models.CFRNet_hyper import *
from models.TARnet_hyper import *
from models.GNN_TARnet_hyper import *
from models.SLearner_hyper import *
from models.TLearner_hyper import *
from models.TEDVAE_hyper import *
from models.GANITE_hyper import *
import argparse
from utils.gnn_data import *
from codecarbon import EmissionsTracker
from hyperparameters import *

import tensorflow as tf
from datetime import datetime
import keras_tuner as kt
# sudo chmod -R a+r /sys/class/powercap/intel-rapl

import warnings

warnings.filterwarnings('ignore')

from utils.gnn_data import *
from utils.defenitions import ROOT_DIR
import os

os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpu, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def main(args):
    model_names = {"TARnet": TARnetHyper,
                   "GNNTARnet": GNNTARnetHyper,
                   "TLearner": TLearner, "CFRNet": CFRNet,
                   "GANITE": GANITE, "SLearner": SLearner, "TEDVAE": TEDVAE
                   }
    tuners = {'random': kt.RandomSearch}

    datasets = {'ihdp_a', 'ihdp_b', 'jobs', 'sum'}
    ipm_list = {'wasserstein', 'None'}

    if args.model_name in model_names and args.dataset_name in datasets and args.ipm_type in ipm_list \
            and args.tuner_name in tuners:
        print('\n-------------------------------------------------\n'
              '\n Chosen model is', args.model_name, args.dataset_name, '  |  ', 'ipm type: ', args.ipm_type, '  |  ',
              'defaults: ', args.defaults, '  |  ', 'drop: ', args.drop, '  |  ',
              'tuner: ', args.tuner_name, '  |  ', 'eye:', args.eye,
              '\n \n-------------------------------------------------\n')
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
        params['eye'] = eval(args.eye)
        model = model_name(params)

        metric_list_train, metric_list_test, average_train, average_test = model.evaluate_performance()
        m_test, h_test = mean_confidence_interval(metric_list_test, confidence=0.95)
        m_train, h_train = mean_confidence_interval(metric_list_train, confidence=0.95)
        am_test, ah_test = mean_confidence_interval(average_test, confidence=0.95)
        am_train, ah_train = mean_confidence_interval(average_train, confidence=0.95)

        em_am_test, em_ah_test = mean_confidence_interval(model.emission_test, confidence=0.95)
        em_am_train, em_ah_train = mean_confidence_interval(model.emission_train, confidence=0.95)

        total_em_test = np.sum(model.emission_test)
        total_em_train = np.sum(model.emission_train)

        print(f'EMISSIONS:' f'total test {total_em_test} | total train: {total_em_train} ||'
              f' mean test {em_am_test} | std test: {em_ah_test} || mean train: {em_am_train} | std train: {em_ah_train} ')

        em_test = str(em_am_test) + ' +- ' + str(em_ah_test)
        em_train = str(em_am_train) + ' +- ' + str(em_ah_train)

        print(
            f'mean test {m_test} | std test: {h_test} || mean train: {m_train} | std train: {h_train} '
            f'|| ate mean test: {am_test} | std mean test {ah_test} || ate mean train: {am_train} | std mean train {ah_train}')
        # save results
        folder_path = "results"
        folder_exists = exists(folder_path)
        # check if folder exists
        if not folder_exists:
            os.makedirs(folder_path)
        now = datetime.now()
        date_day = now.strftime("%m_%d_%Y")
        date_time = now.strftime("%H:%M:%S")
        file_name = folder_path + "/results_" + date_day + '.csv'
        file_exists = exists(file_name)

        columns = ["time", "model_name", "dataset", "defaults", "mean_train", "std_train", "mean_test", "std_test",
                   "ate_mean_train", "std_mean_train", "ate_mean_test",
                   "std_mean_test", "tuner", "comments", "emissions_train", "emissions_test", "total_emissions_test",
                   "total_emission_train", "params"]
        if file_exists:
            results = pd.read_csv(file_name)
        else:
            results = pd.DataFrame(data=None,
                                   columns=columns)

        if args.model_name == 'CFRNet':
            model_name = args.model_name + '_' + args.ipm_type
        else:
            model_name = args.model_name

        # save results
        result = pd.DataFrame([[date_time, model_name, args.dataset_name, str(args.defaults), m_train, h_train, m_test,
                                h_test, am_train, ah_train, am_test,
                                ah_test, args.tuner_name, args.comments, em_test, em_train, total_em_test,
                                total_em_train,
                                model.sparams]],
                              columns=columns)
        if results.empty:
            results = result
        else:
            results = results.append(result, ignore_index=True)
        # save the results
        results.to_csv(file_name, index=False)
        return 0
    else:
        raise ValueError(f'{args.model_name} has not been implemented yet!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Causal Model')
    parser.add_argument("--model-name", default='GNNTARnet', type=str)
    parser.add_argument("--ipm-type", default='None', type=str)
    parser.add_argument("--defaults", default="True", type=str)
    parser.add_argument("--dataset-name", default='ihdp_a', type=str)
    parser.add_argument("--tuner-name", default='random', type=str)
    parser.add_argument("--drop", default=None, type=int)
    #change num to 1
    parser.add_argument("--num", default=1, type=int)
    parser.add_argument("--eye", default="False", type=str)
    parser.add_argument("--num_layers", default=5, type=int)
    parser.add_argument("--comments", default='graph infly true', type=str)
    args = parser.parse_args()

    folder_path = "results"
    folder_exists = exists(folder_path)

    if args.dataset_name != "sum":
        generate_graphs(dataset_name=args.dataset_name)

    # check if folder exists
    if not folder_exists:
        os.makedirs(folder_path)

    if args.model_name == "GNNTARnet" and args.dataset_name == "sum":
        run_gnn_on_sum(eye=args.eye, num=args.num,
                   num_layers=args.num_layers)

    if args.model_name == "TARnet" and args.dataset_name == "sum":
        run_tarnet_on_sum(num=args.num, num_layers=args.num_layers)

    main(args)