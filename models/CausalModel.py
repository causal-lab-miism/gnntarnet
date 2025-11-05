from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from scipy.special import expit
from utils.set_seed import *
import json
from os.path import exists
import shutil
import warnings, logging
import scipy.stats
from datetime import datetime
import keras_tuner as kt
from codecarbon import EmissionsTracker
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').disabled = True

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h

class CausalModel:
    def __init__(self, params):
        self.dataset_name = params['dataset_name']
        self.num = params['num']
        self.params = params
        self.binary = params['binary']
        self.sparams = ""
        self.emission_test = list()
        self.emission_train = list()
        self.folder_ind = None
        self.sum_size = None

    @staticmethod
    def setSeed(seed):
        os.environ['PYTHONHASHSEED'] = '0'

        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    def train_and_evaluate(self, pehe_list_train, pehe_list_test, ate_list_train, ate_list_test, **kwargs):
        pass

    def evaluate_performance(self):
        if self.dataset_name in ['ihdp_a', 'ihdp_b', 'ihdp_g', 'sum']:
            return self.evaluate_performance_ihdp()
        if self.dataset_name == 'twins':
            return self.evaluate_performance_twins()
        if self.dataset_name == 'jobs':
            return self.evaluate_performance_jobs()
        if self.dataset_name == 'acic':
            return self.evaluate_performance_acic()
        if self.dataset_name == 'gnn':
            return self.evaluate_performance_gnn()

    def evaluate_performance_gnn(self):
        pehe_list_train = list()
        pehe_list_test = list()
        ate_list_train = list()
        ate_list_test = list()
        # number of gnn folders
        len_gnn_folder = len(os.listdir('./GNN/'))
        now = datetime.now()
        date_day = now.strftime("%m_%d_%Y")
        # check if the file exists and load it. If the file doesn't exist, create it
        file_name = 'gnn_' + self.params['model_name'] + '_edges_drop_' + str(self.params['drop']) + '.csv'
        file_exists = exists(file_name)
        if file_exists:
            results = pd.read_csv(file_name)
            start = len(results)
        else:
            results = pd.DataFrame(data=None,
                                   columns=["model_name", "num_edges", "mean_pehe", "std_gnn_pehe"])
            start = 0

        if start == len_gnn_folder-1:
            print('All the results are saved. No need to run the code again.')
            return pehe_list_train, pehe_list_test, ate_list_train, ate_list_test

        for folder in range(start, len_gnn_folder):
            len_folder_files = 100
            pehe_list_train = list()
            pehe_list_test = list()
            ate_list_train = list()
            ate_list_test = list()
            for file in range(len_folder_files):
                kwargs = {'folder_ind': folder, 'count': file}
                self.train_and_evaluate(pehe_list_train, pehe_list_test, ate_list_train, ate_list_test, **kwargs)

            mean_pehe, std_pehe = mean_confidence_interval(pehe_list_test, confidence=0.95)
            num_edges = folder * 10 + 10

            # save results
            result = pd.DataFrame([[self.params['model_name'], num_edges, mean_pehe, std_pehe]],
                                  columns=["model_name", "num_edges", "mean_pehe", "std_gnn_pehe"])
            if results.empty:
                results = result
            else:
                results = results.append(result, ignore_index=True)

            # save the results
            results.to_csv(file_name, index=False)
        return pehe_list_train, pehe_list_test, ate_list_train, ate_list_test

    # evaluate the model performance
    def evaluate_performance_ihdp(self):
        num = self.num
        pehe_list_train = list()
        pehe_list_test = list()
        ate_list_train = list()
        ate_list_test = list()
        folder = 1
        for file in range(num):
            kwargs = {'folder_ind': folder, 'count': file}
            self.train_and_evaluate(pehe_list_train, pehe_list_test, ate_list_train, ate_list_test, **kwargs)
        return pehe_list_train, pehe_list_test, ate_list_train, ate_list_test

    def evaluate_performance_twins(self):
        num = self.num
        pehe_list = list()
        folder = 1
        for file in range(num):
            kwargs = {'folder_ind': folder, 'count': file}
            self.train_and_evaluate(pehe_list, None, None, **kwargs)
        return pehe_list

    def evaluate_performance_jobs(self):
        num = self.num
        policy_risk_list_train = list()
        policy_risk_list_test = list()
        att_list_train = list()
        att_list_test = list()
        folder = 1
        for file in range(num):
            kwargs = {'folder_ind': folder, 'count': file}
            self.train_and_evaluate(policy_risk_list_train, policy_risk_list_test, att_list_train, att_list_test,
                                    **kwargs)
        return policy_risk_list_train, policy_risk_list_test, att_list_train, att_list_test

    def evaluate_performance_acic(self):
        """
         Evaluates performance of the model on the ACIC dataset.

         @return: A tuple containing two lists: `pehe_list_train` and `pehe_list_test`.
             `pehe_list_train` contains the PEHE values on the training set for each iteration,
             `pehe_list_test` contains the PEHE values on the test set for each iteration.
         """
        num = self.num
        pehe_list_train = list()
        pehe_list_test = list()
        ate_list_train = list()
        ate_list_test = list()
        for folder in range(1, num + 1):
            len_folder_files = len(os.listdir('./ACIC/' + str(folder) + '/'))
            file = 0
            kwargs = {'folder_ind': folder, 'count': file}
            self.train_and_evaluate(pehe_list_train, pehe_list_test, ate_list_train, ate_list_test, **kwargs)
        return pehe_list_train, pehe_list_test, ate_list_train, ate_list_test

    def elast(self, data, y, t):
        # line coeficient for the one variable linear regression
        return (np.sum((data[t] - data[t].mean()) * (data[y] - data[y].mean())) /
                np.sum((data[t] - data[t].mean()) ** 2))

    def cumulative_gain(self, dataset, prediction, y, t, min_periods=30, steps=100):
        size = dataset.shape[0]
        ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
        n_rows = list(range(min_periods, size, size // steps)) + [size]

        ## add (rows/size) as a normalizer.
        return np.array([self.elast(ordered_df.head(rows), y, t) * (rows / size) for rows in n_rows])


    def plot_cumulative_gain(self, gain_curve_test, gain_curve_train, data_test):
        plt.plot(gain_curve_test, color="C0", label="Test")
        plt.plot(gain_curve_train, color="C1", label="Train")
        plt.plot([0, 100], [0, self.elast(data_test, "y", "t")], linestyle="--", color="black", label="Baseline")
        plt.legend()
        plt.title(self.params['model_name'] + ' ' + self.params['dataset_name'] + " Cumulative Gain")
        plt.show()


    def find_pehe(self, y0_pred, y1_pred, data):
        """
          Calculates the PEHE and ATE metrics.

          @param y0_pred: The predicted y0 values.
          @param y1_pred: The predicted y1 values.
          @param data: The data dictionary.

          @return: A tuple containing two values: `sqrt_pehe` and `ate`.
              `sqrt_pehe` is the square root of the PEHE metric,
              `ate` is the absolute difference between the mean of the predicted CATE and the true ATE.
          """
        if self.binary:
            cate_pred = y1_pred - y0_pred
        else:
            y0_pred = data['y_scaler'].inverse_transform(y0_pred)
            y1_pred = data['y_scaler'].inverse_transform(y1_pred)
            cate_pred = (y1_pred - y0_pred).squeeze()
        cate_true = (data['mu_1'] - data['mu_0']).squeeze()
        pehe = np.mean(np.square((cate_true - cate_pred)))
        sqrt_pehe = np.sqrt(pehe)
        ate = np.abs(np.mean(cate_pred) - np.mean(cate_true))
        return sqrt_pehe, ate

    def find_policy_risk(self, y0_pred, y1_pred, data):
        """
        Calculates policy value, policy risk, policy curve, and epsilon ATT.

        @param y0_pred: The predicted y0 values.
        @param y1_pred: The predicted y1 values.
        @param data: The data dictionary.

        @return: A tuple containing four values:
            `policy_value` is the policy value metric,
            `policy_risk` is the policy risk metric,
            `policy_curve` is the policy curve,
            `eps_ATT` is the absolute difference between the true ATT and the mean of the predicted CATE for treated subjects.
        """
        if self.binary:
            cate_pred = y1_pred - y0_pred
        else:
            cate_pred = (y1_pred - y0_pred).squeeze()

        cate_true = data['tau']
        t = data['t']
        # find subjects untreated and assigned to be untreated
        t_e = t + data['tau']
        # untreated assigned to be untreated subjects
        t_c_e = t_e < 1.0

        ATT = np.mean(data['y'][t > 0]) - np.mean(data['y'][t_c_e])
        eps_ATT = np.abs(ATT - np.mean(cate_pred[t > 0]))

        policy_value, policy_curve = self.policy_val(data['t'][cate_true > 0], data['y'][cate_true > 0],
                                                     cate_pred[cate_true > 0], False)
        policy_risk = 1 - policy_value

        return policy_value, policy_risk, policy_curve, eps_ATT

    @staticmethod
    def policy_range(n, res=10):
        """
         Generates a range of policy thresholds.

         @param n: The number of units in the population.
         @param res: The desired number of thresholds.

         @return: A list of `res` thresholds, ranging from 0 to `n`.
         """
        step = int(float(n) / float(res))
        n_range = range(0, int(n + 1), step)
        if not n_range[-1] == n:
            n_range.append(n)

        # To make sure every curve is same length. Incurs a small error if res high.
        # Only occurs if number of units considered differs.
        # For example if resampling validation sets (with different number of
        # units in the randomized sub-population)

        while len(n_range) > res:
            k = np.random.randint(len(n_range) - 2) + 1
            del n_range[k]

        return n_range

    def policy_val(self, t, yf, eff_pred, compute_policy_curve=False):
        """
        Computes the value of the policy defined by predicted effect.

        @param self: The instance of the class.
        @param t: The treatment assignment indicator (1 if treated, 0 otherwise).
        @param yf: The outcome variable under treatment and control.
        @param eff_pred: The predicted treatment effect.
        @param compute_policy_curve: Whether to compute the policy curve.

        @return policy_value: The value of the policy.
        @return policy_curve: The policy curve (if `compute_policy_curve` is True).
        """

        pol_curve_res = 40

        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0

        if isinstance(policy, np.ndarray) and isinstance(t, np.ndarray):
            treat_overlap = (policy == t) * (t > 0)
            control_overlap = (policy == t) * (t < 1)
        else:
            treat_overlap = (policy == t).numpy() * (t > 0)
            control_overlap = (policy == t).numpy() * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        policy_curve = []

        if compute_policy_curve:
            n = t.shape[0]
            I_sort = np.argsort(-eff_pred)

            n_range = self.policy_range(n, pol_curve_res)

            for i in n_range:
                I = I_sort[0:i]

                policy_i = 0 * policy
                policy_i[I] = 1
                pit_i = np.mean(policy_i)

                treat_overlap = (policy_i > 0) * (t > 0)
                control_overlap = (policy_i < 1) * (t < 1)

                if np.sum(treat_overlap) == 0:
                    treat_value = 0
                else:
                    treat_value = np.mean(yf[treat_overlap])

                if np.sum(control_overlap) == 0:
                    control_value = 0
                else:
                    control_value = np.mean(yf[control_overlap])

                policy_curve.append(pit_i * treat_value + (1 - pit_i) * control_value)

        return policy_value, policy_curve

    def regression_loss(self, concat_true, concat_pred):
        """
        Computes the loss of a regression model used for causal inference.

        @param self: The instance of the class.
        @param concat_true: The concatenated true outcomes and treatment assignments.
        @param concat_pred: The concatenated predicted outcomes for control and treated groups.

        @return loss: The sum of the loss for untreated and treated samples.
        """
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]

        if self.params['binary']:
            y0_pred = tf.cast((y0_pred > 0.5), tf.float32)
            y1_pred = tf.cast((y1_pred > 0.5), tf.float32)
            loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=(1 - t_true) * y_true,
                                                                           logits=y0_pred))
            loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=t_true * y_true, logits=y1_pred))

        else:
            loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
            loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))
        return loss0 + loss1

    def load_data(self, **kwargs):
        """
         Loads the specified dataset.

         Parameters:
         ----------
         dataset_name: str
             The name of the dataset to load. Valid options include 'ihdp_a', 'ihdp_b', 'acic', 'twins', 'jobs', 'gnn', 'ihdp_g', and 'sum'.
         count: int, optional
             The number of samples to load from the dataset. Defaults to loading all samples.

         Returns:
         ----------
         data: numpy.ndarray
             The loaded dataset.

         Raises:
         ----------
         ValueError: If an invalid dataset name is specified.

         """
        if self.dataset_name == 'ihdp_a':
            path_data = "./IHDP_a"
            return self.load_ihdp_data(path_data, kwargs.get('count'))
        elif self.dataset_name == 'ihdp_b':
            path_data = "./IHDP_b"
            return self.load_ihdp_data(path_data, kwargs.get('count'))
        elif self.dataset_name == 'ihdp_g':
            path_data = "./IHDP_g"
            return self.load_ihdp_data(path_data, kwargs.get('count'))
        elif self.dataset_name == 'acic':
            return self.load_acic_data(kwargs.get('folder_ind'), kwargs.get('count'))
        elif self.dataset_name == 'twins':
            return self.load_twins_data(kwargs.get('count'))
        elif self.dataset_name == 'jobs':
            path_data = "./JOBS"
            return self.load_jobs_data(path_data, kwargs.get('count'))
        elif self.dataset_name == 'gnn':
            return self.load_gnn_data(kwargs.get('folder_ind'), kwargs.get('count'))
        elif self.dataset_name == 'sum':
            path_data = './SUM_' + str(self.params['num_layers']) + '/'
            file_name_train = 'sum_train_'
            file_name_test = 'sum_test_'
            return self.load_sum_data(path_data, file_name_train, file_name_test, kwargs.get('count'), size=self.sum_size)
        else:
            print('No such dataset. The available datasets are: ',
                  'ihdp_a, ihdp_b, acic, twins, jobs, gnn, ihdp_g, sum')

    @staticmethod
    def load_sum_data(path_data, file_name_train, file_name_test, i, size):
        """
           Loads a sum dataset.

           Parameters:
           ----------
           path_data: str
               The path to the dataset.
           file_name_train: str
               The prefix of the filename for the training data.
           file_name_test: str
               The prefix of the filename for the test data.
           i: int
               The index of the dataset.

           Returns:
           ----------
           data_train: dict
               The loaded training dataset.
           data_test: dict
               The loaded test dataset.

           """
        path_train_data = path_data + file_name_train + str(i) + '.csv'
        path_test_data = path_data + file_name_test + str(i) + '.csv'

        data_train_load = pd.read_csv(path_train_data)
        data_test_load = pd.read_csv(path_test_data)
        if size is not None:
            data_test_load = data_test_load.iloc[:size, :]
            data_train_load = data_train_load.iloc[:size, :]
        data_train = {}
        data_test = {}


        # we're just padding one dimensional vectors with an additional dimension
        x_train = np.asarray(data_train_load.iloc[:, 4:])
        x_test = np.asarray(data_test_load.iloc[:, 4:])
        # scale the data
        x_scaler = StandardScaler().fit(x_train)
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)
        data_train['x'] = x_train


        data_train['t'] = np.asarray(data_train_load['t']).reshape(-1, 1).astype('float32')
        data_train['y'] = np.asarray(data_train_load['y']).reshape(-1, 1).astype('float32')
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_train['y_scaler'] = StandardScaler().fit(data_train['y'])
        data_train['ys'] = data_train['y_scaler'].transform(data_train['y'])
        data_train['mu_1'] = np.asarray(data_train_load['mu_1'])
        data_train['mu_0'] = np.asarray(data_train_load['mu_0'])

        # we're just padding one dimensional vectors with an additional dimension
        data_test['x'] = x_test
        data_test['t'] = np.asarray(data_test_load['t']).reshape(-1, 1)
        data_test['y'] = np.asarray(data_test_load['y']).reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_test['y_scaler'] = data_train['y_scaler']
        data_test['ys'] = data_test['y_scaler'].transform(data_test['y'])

        data_test['mu_1'] = np.asarray(data_test_load['mu_1'])
        data_test['mu_0'] = np.asarray(data_test_load['mu_0'])

        return data_train, data_test

    @staticmethod
    def load_gnn_data(folder_ind, file_ind):
        path_data = "./GNN/GNN_" + str(folder_ind) + "/"
        file_name_train = "gnn_train_" + str(file_ind) + ".csv"
        file_name_test = "gnn_test_" + str(file_ind) + ".csv"
        path_train_data = path_data + file_name_train
        path_test_data = path_data + file_name_test

        data_train_load = pd.read_csv(path_train_data)
        data_test_load = pd.read_csv(path_test_data)
        # size = 100
        # data_train_load = data_train_load.iloc[:size, :]

        data_train = {}
        data_test = {}
        # we're just padding one dimensional vectors with an additional dimension
        data_train['x'] = np.asarray(np.asarray(data_train_load.iloc[:, 4:]))
        data_train['t'] = np.asarray(data_train_load['t']).reshape(-1, 1).astype('float32')
        data_train['y'] = np.asarray(data_train_load['y']).reshape(-1, 1).astype('float32')
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_train['y_scaler'] = StandardScaler().fit(data_train['y'])
        data_train['ys'] = data_train['y_scaler'].transform(data_train['y'])
        data_train['mu_1'] = np.asarray(data_train_load['mu_1'])
        data_train['mu_0'] = np.asarray(data_train_load['mu_0'])

        # we're just padding one dimensional vectors with an additional dimension
        data_test['x'] = np.asarray(np.asarray(data_test_load.iloc[:, 4:]))
        data_test['t'] = np.asarray(data_test_load['t']).reshape(-1, 1)
        data_test['y'] = np.asarray(data_test_load['y']).reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_test['y_scaler'] = data_train['y_scaler']
        data_test['ys'] = data_test['y_scaler'].transform(data_test['y'])

        data_test['mu_1'] = np.asarray(data_test_load['mu_1'])
        data_test['mu_0'] = np.asarray(data_test_load['mu_0'])

        return data_train, data_test

    @staticmethod
    def load_ihdp_data(path_data, i=7):
        """
         Load IHDP data for a specified fold number.

         Args:
             path_data (str): Path to the directory where the IHDP data is stored.
             i (int): Fold number to load (default is 7).

         Returns:
             tuple: A tuple of two dictionaries representing the training and testing data.
                 Each dictionary contains the following keys:
                 - 'x': A numpy array representing the covariates.
                 - 't': A numpy array representing the treatment.
                 - 'y': A numpy array representing the outcome.
                 - 'mu_0': A numpy array representing the potential outcome under control condition.
                 - 'mu_1': A numpy array representing the potential outcome under treatment condition.
                 Additionally, the training data dictionary contains:
                 - 'y_scaler': A sklearn StandardScaler object fitted on the training data 'y' values.
                 - 'ys': A numpy array representing the rescaled 'y' values using the 'y_scaler' fitted on the training data.
         """

        data_train = np.loadtxt(path_data + '/ihdp_npci_train_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)
        data_test = np.loadtxt(path_data + '/ihdp_npci_test_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)

        t_train, y_train = data_train[:, 0], data_train[:, 1][:, np.newaxis]
        mu_0_train, mu_1_train, x_train = data_train[:, 3][:, np.newaxis], data_train[:, 4][:, np.newaxis], data_train[
                                                                                                            :, 5:]

        t_test, y_test = data_test[:, 0].astype('float32'), data_test[:, 1][:, np.newaxis].astype('float32')
        mu_0_test, mu_1_test, x_test = data_test[:, 3][:, np.newaxis].astype('float32'), data_test[:, 4][:,
                                                                                         np.newaxis].astype('float32'), \
            data_test[:, 5:].astype('float32')
        # x_train_cont = (x_train[:, :6] + 5) / 10.
        # x_train[:, :6] = x_train_cont
        #
        # x_test_cont = (x_test[:, :6] + 5) / 10.
        # x_test[:, :6] = x_test_cont
        data_train = {'x': x_train, 't': t_train, 'y': y_train, 'mu_0': mu_0_train, 'mu_1': mu_1_train}
        # data_train = remove_anomalies(data_train)

        data_train['t'] = data_train['t'].reshape(-1,
                                                  1)  # we're just padding one dimensional vectors with an additional dimension
        data_train['y'] = data_train['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_train['y_scaler'] = StandardScaler().fit(data_train['y'])
        data_train['ys'] = data_train['y_scaler'].transform(data_train['y'])

        data_test = {'x': x_test, 't': t_test, 'y': y_test, 'mu_0': mu_0_test, 'mu_1': mu_1_test}
        data_test['t'] = data_test['t'].reshape(-1,
                                                1)  # we're just padding one dimensional vectors with an additional dimension
        data_test['y'] = data_test['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_test['y_scaler'] = data_train['y_scaler']
        data_test['ys'] = data_test['y_scaler'].transform(data_test['y'])

        return data_train, data_test

    @staticmethod
    def load_jobs_data(path_data, i=7):
        """
        Load the jobs dataset from the given file paths for a given fold.

        Parameters:
        -----------
        path_data : str
            The path to the folder containing the dataset files.
        i : int, optional
            The fold to use for the dataset. Default is 7.

        Returns:
        --------
        tuple
            A tuple containing two dictionaries, one for the training data and one for the testing data.
            Each dictionary contains the following key-value pairs:
                - 'x': ndarray, the confounding features for each sample
                - 't': ndarray, the factual observations for each sample
                - 'y': ndarray, the treatment values for each sample
                - 'tau': ndarray, the randomized trial values for each sample
        """
        data_file_train = path_data + f'/jobs_train_{i}.csv'

        data_file_test = path_data + f'/jobs_test_{i}.csv'

        df_train = np.loadtxt(data_file_train, delimiter=',', skiprows=1)

        x_train = np.squeeze(df_train[:, 0:17])  # confounders
        t_train = df_train[:, 17:18]  # factual observation
        y_train = df_train[:, 18:19].astype(np.float32)  # treatment
        e_train = df_train[:, 19:20]  # randomized trial

        data_mean = np.mean(x_train, axis=0, keepdims=True)
        data_std = np.std(x_train, axis=0, keepdims=True)

        x_train = (x_train - data_mean) / data_std

        data_train = {'x': x_train, 'y': y_train, 't': t_train, 'tau': e_train}

        df_test = np.loadtxt(data_file_test, delimiter=',', skiprows=1)

        x_test = np.squeeze(df_test[:, 0:17])  # confounders
        t_test = df_test[:, 17:18]  # factual observation
        y_test = df_test[:, 18:19].astype(np.float32)  # treatment
        e_test = df_test[:, 19:20]  # randomized trial

        x_test = (x_test - data_mean) / data_std

        data_test = {'x': x_test, 'y': y_test, 't': t_test, 'tau': e_test}

        return data_train, data_test

    @staticmethod
    def load_cfdata(file_dir):
        """
        Load the counterfactual data from a given directory.

        Args:
            file_dir (str): The directory containing the counterfactual data.

        Returns:
            dict: A dictionary containing the counterfactual data with keys 't', 'y', 'mu_0', 'mu_1'.
        """
        df = pd.read_csv(file_dir)
        z = df['z'].values[:, np.newaxis].astype('float32')
        y0 = df['y0'].values[:, np.newaxis].astype('float32')
        y1 = df['y1'].values[:, np.newaxis].astype('float32')
        y = y0 * (1 - z) + y1 * z
        mu_0, mu_1 = df['mu0'].values[:, np.newaxis].astype('float32'), df['mu1'].values[:, np.newaxis].astype(
            'float32')

        data_cf = {'t': z, 'y': y, 'mu_0': mu_0, 'mu_1': mu_1}
        return data_cf

    def load_acic_data(self, folder_ind=1, file_ind=1):

        data = pd.read_csv('./ACIC/x.csv')
        del data['x_2']
        del data['x_21']
        del data['x_24']

        data = data.dropna()
        data = data.values

        # load y and simulations
        folder_dir = './ACIC/' + str(folder_ind) + '/'
        filelist = os.listdir(folder_dir)
        data_cf = self.load_cfdata(folder_dir + filelist[file_ind])

        # number of observations
        n = data.shape[0]
        test_ind = 4000

        # create train data
        x_train = data[:test_ind, :]

        data_mean = np.mean(x_train, axis=0, keepdims=True)
        data_std = np.std(x_train, axis=0, keepdims=True)
        x_train = (x_train - data_mean) / data_std

        y_train = data_cf['y'][:test_ind, :]
        t_train = data_cf['t'][:test_ind, :]
        mu_0_train = data_cf['mu_0'][:test_ind]
        mu_1_train = data_cf['mu_1'][:test_ind]

        data_train = {'x': x_train, 't': t_train, 'y': y_train, 'mu_0': mu_0_train, 'mu_1': mu_1_train}

        data_train['t'] = data_train['t'].reshape(-1,
                                                  1)  # we're just padding one dimensional vectors with an additional dimension
        data_train['y'] = data_train['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_train['y_scaler'] = StandardScaler().fit(data_train['y'])
        data_train['ys'] = data_train['y_scaler'].transform(data_train['y'])
        # create test data

        x_test = data[test_ind:, :]
        x_test = (x_test - data_mean) / data_std
        y_test = data_cf['y'][test_ind:, :]
        t_test = data_cf['t'][test_ind:, :]
        mu_0_test = data_cf['mu_0'][test_ind:]
        mu_1_test = data_cf['mu_1'][test_ind:]

        data_test = {'x': x_test, 't': t_test, 'y': y_test, 'mu_0': mu_0_test, 'mu_1': mu_1_test}
        data_test['t'] = data_test['t'].reshape(-1,
                                                1)  # we're just padding one dimensional vectors with an additional dimension
        data_test['y'] = data_test['y'].reshape(-1, 1)
        # rescaling y between 0 and 1 often makes training of DL regressors easier
        data_test['y_scaler'] = data_train['y_scaler']
        data_test['ys'] = data_test['y_scaler'].transform(data_test['y'])

        return data_train, data_test

    @staticmethod
    def load_twins_data(count=1):
        train_rate = 0.8
        # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
        df = np.loadtxt("./TWINS/Twin_data.csv", delimiter=",", skiprows=1)

        # Define features
        x = df[:, :30]
        no, dim = x.shape
        # Define potential outcomes
        mu0_mu1 = df[:, 30:]
        # Die within 1 year = 1, otherwise = 0
        mu0_mu1 = np.array(mu0_mu1 < 9999, dtype=float)

        ## Assign treatment
        setSeed(count)
        np.random.seed(count)
        coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])

        prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

        prob_t = prob_temp / (2 * np.mean(prob_temp))
        prob_t[prob_t > 1] = 1

        t = np.random.binomial(1, prob_t, [no, 1])
        t = t.reshape([no, ])

        ## Define observable outcomes
        y = np.zeros([no, 1])
        y = np.transpose(t) * mu0_mu1[:, 1] + np.transpose(1 - t) * mu0_mu1[:, 0]
        y = np.reshape(np.transpose(y), [no, ])

        ## Train/test division
        idx = np.random.permutation(no)
        train_idx = idx[:int(train_rate * no)]
        test_idx = idx[int(train_rate * no):]

        x_train = x[train_idx, :]
        data_mean = np.mean(x_train, axis=0, keepdims=True)
        data_std = np.std(x_train, axis=0, keepdims=True)
        # x_train = (x_train - data_mean) / data_std

        t_train = t[train_idx]
        y_train = y[train_idx]
        mu0_mu1_train = mu0_mu1[train_idx, :]
        mu_0_train = mu0_mu1_train[:, 0]
        mu_1_train = mu0_mu1_train[:, 1]

        data_train = {'x': x_train, 't': t_train, 'y': y_train, 'mu_0': mu_0_train, 'mu_1': mu_1_train}
        data_train['t'] = data_train['t'].reshape(-1, 1)  # we're just padding one dimensional vectors with
        # an additional dimension
        data_train['y'] = data_train['y'].reshape(-1, 1)

        x_test = x[test_idx, :]
        # x_test = (x_test - data_mean) / data_std
        t_test = t[test_idx]
        y_test = y[test_idx]
        mu0_mu1_test = mu0_mu1[test_idx, :]
        mu_0_test = mu0_mu1_test[:, 0]
        mu_1_test = mu0_mu1_test[:, 1]

        data_test = {'x': x_test, 't': t_test, 'y': y_test, 'mu_0': mu_0_test, 'mu_1': mu_1_test}
        data_test['t'] = data_test['t'].reshape(-1, 1)  # we're just padding one dimensional vectors with
        # an additional dimension
        data_test['y'] = data_test['y'].reshape(-1, 1)

        return data_train, data_test

    def define_tuner(self, hypermodel, hp, objective, directory_name, project_name):
        if self.params['tuner_name'] == 'hyperband':
            tuner = self.params['tuner'](
                hypermodel=hypermodel,
                objective=objective,
                directory=directory_name,
                max_epochs=50,
                tuner_id='2',
                overwrite=False,
                hyperparameters=hp,
                project_name=project_name,
                seed=0)
        else:
            tuner = self.params['tuner'](
                hypermodel=hypermodel,
                objective=objective,
                directory=directory_name,
                tuner_id='2',
                overwrite=False,
                hyperparameters=hp,
                project_name=project_name,
                max_trials=self.params['max_trials'],
                seed=0)
        return tuner

    def get_trackers(self, count):
        folder_path = './Emissions/' + self.params['model_name'] + '/'
        if self.params['defaults']:
            folder_path += 'default/'
        else:
            folder_path += self.params['tuner_name'] + '/'

        file_path_train = self.params['model_name'] + '_' + self.params['dataset_name'] + '_train.csv'
        file_path_test = self.params['model_name'] + '_' + self.params['dataset_name'] + '_test.csv'

        if self.params['dataset_name'] == 'gnn':
            file_path_train = self.params['model_name'] + '_' + self.params['dataset_name'] + '_train_' + str(self.folder_ind) + '.csv'
            file_path_test = self.params['model_name'] + '_' + self.params['dataset_name'] + '_test_' + str(self.folder_ind) + '.csv'

        file_exists_train = exists(folder_path + file_path_train)
        file_exists_test = exists(folder_path + file_path_test)
        folder_exists = exists(folder_path)

        if file_exists_train:
            # delete file
            if count == 0:
                os.remove(folder_path + file_path_train)

        if file_exists_test:
            # delete file
            if count == 0:
                os.remove(folder_path + file_path_test)
        # check if folder exists
        if not folder_exists:
            os.makedirs(folder_path)
        tracker_train = EmissionsTracker(project_name=self.params['model_name'], output_dir=folder_path,
                                         output_file=file_path_train, log_level="critical")
        tracker_test = EmissionsTracker(project_name=self.params['model_name'], output_dir=folder_path,
                                        output_file=file_path_test, log_level="critical")

        return tracker_test, tracker_train
