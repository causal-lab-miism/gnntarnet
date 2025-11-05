from keras import Model
from models.CausalModel import *
from utils.layers import FullyConnected
from utils.callback import callbacks
from utils.set_seed import setSeed
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, EarlyStopping
from tensorflow.compat.v1.profiler import Profiler
from os.path import exists
import shutil
from codecarbon import track_emissions
from codecarbon import EmissionsTracker


class HyperSLearner(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = SModel(name='slearner', params=self.params, hp=hp)
        optimizer = Adam(learning_rate=self.params['lr'])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class SModel(Model):
    def __init__(self, params, hp, name='slearner', **kwargs):
        super(SModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.hidden_phi,
                                 final_activation=params['activation'], out_size=1,
                                 kernel_init=params['kernel_init'], kernel_reg=None, name='fc')

    def call(self, inputs):
        return self.fc(inputs)


class SLearner(CausalModel):
    """
    This class can be used to train and create stacked model
    for IHDP dataset setting "b"
    """

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.directory_name = None
        self.project_name = None
        self.best_hps = None

    def fit_tuner(self, x, y, t, seed):
        setSeed(seed)
        directory_name = 'params_' + self.params['tuner_name'] + '/' + self.params['dataset_name']

        # if self.dataset_name == 'acic':
        #     directory_name = directory_name + f'/{self.params["model_name"]}'
        #     project_name = str(self.folder_ind)
        # else:
        project_name = self.params["model_name"]

        self.directory_name = directory_name
        self.project_name = project_name

        # if exists(directory_name + '/' + project_name) and count == 0:
        #     shutil.rmtree(directory_name + '/' + project_name)

        hp = kt.HyperParameters()
        hypermodel = HyperSLearner(params=self.params)
        objective = kt.Objective("val_mse", direction="min")
        tuner = self.define_tuner(hypermodel, hp, objective, directory_name, project_name)

        x_t = tf.concat([x, t], axis=1)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_mse', patience=5)]
        tuner.search(x_t, y, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=self.params['verbose'])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if self.params['defaults']:
            best_hps.values = {'n_fc': self.params['n_fc'],
                               'hidden_phi': self.params['hidden_phi']}
        self.best_hps = best_hps

        return

    # @track_emissions(project_name='SLearner', output_dir='./Emissions', output_file='SLearner')
    def fit_model(self, x, y, t, count, seed):
        setSeed(seed)

        x_t = tf.concat([x, t], axis=1)
        setSeed(seed)

        tuner = kt.RandomSearch(
            HyperSLearner(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name,
            seed=0)

        best_hps = self.best_hps

        model = tuner.hypermodel.build(best_hps)
        stop_early = [
            ReduceLROnPlateau(monitor='mse', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=0., cooldown=0, min_lr=1e-8),
            EarlyStopping(monitor='mse', patience=40, restore_best_weights=True)]

        model.fit(x_t, y, epochs=self.params['epochs'], callbacks=stop_early,
                  batch_size=self.params['batch_size'], validation_split=0.0,
                  verbose=self.params['verbose'])
        if count == 0 and self.folder_ind == 0:
            self.sparams = f"""n_fc={best_hps.get('n_fc')} - hidden_phi = {best_hps.get('hidden_phi')}"""
            print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                  layer is {self.sparams}""")
        return model

    @staticmethod
    def evaluate(x_test, model):
        x_t0 = tf.concat([x_test, tf.zeros([x_test.shape[0], 1], dtype=tf.float64)], axis=1)
        x_t1 = tf.concat([x_test, tf.ones([x_test.shape[0], 1], dtype=tf.float64)], axis=1)

        out0 = model(x_t0)
        out1 = model(x_t1)
        return tf.concat((out0, out1), axis=1)

    def train_and_evaluate(self, metric_list_train, metric_list_test, average_metric_list_train,
                           average_metric_list_test, **kwargs):
        # kwargs['count'] = 87
        data_train, data_test = self.load_data(**kwargs)
        count = kwargs.get('count')
        self.folder_ind = kwargs.get('folder_ind') - 1

        tracker_test, tracker_train = self.get_trackers(count)

        if self.params['binary']:
            # fit tuner on the first dataset
            with tracker_train:
                self.fit_tuner(data_train['x'], data_train['y'], data_train['t'], seed=0)
                model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], count, seed=0)
            self.emission_train.append(tracker_train.final_emissions)
        else:
            # fit tuner on the first dataset
            with tracker_train:
                self.fit_tuner(data_train['x'], data_train['ys'], data_train['t'], seed=0)
                model = self.fit_model(data_train['x'], data_train['ys'], data_train['t'], count, seed=0)
            self.emission_train.append(tracker_train.final_emissions)


        # make a prediction
        with tracker_test:
            concat_pred_test = self.evaluate(data_test['x'], model)
            concat_pred_train = self.evaluate(data_train['x'], model)
        self.emission_test.append(tracker_test.final_emissions)

        y0_pred_test, y1_pred_test = concat_pred_test[:, 0], concat_pred_test[:, 1]
        y0_pred_test = tf.expand_dims(y0_pred_test, axis=1)
        y1_pred_test = tf.expand_dims(y1_pred_test, axis=1)

        y0_pred_train, y1_pred_train = concat_pred_train[:, 0], concat_pred_train[:, 1]
        y0_pred_train = tf.expand_dims(y0_pred_train, axis=1)
        y1_pred_train = tf.expand_dims(y1_pred_train, axis=1)

        if self.params['dataset_name'] == 'jobs':
            _, policy_risk_test, _, test_ATT = self.find_policy_risk(y0_pred_test, y1_pred_test, data_test)
            _, policy_risk_train, _, train_ATT = self.find_policy_risk(y0_pred_train, y1_pred_train, data_train)

            print(kwargs.get('count'), 'Policy Risk Test = ', policy_risk_test, '| Test ATT', test_ATT,
                  '| Policy Risk Train = ', policy_risk_train, '| Train ATT', train_ATT)
            metric_list_test.append(policy_risk_test)
            metric_list_train.append(policy_risk_train)

            average_metric_list_test.append(test_ATT)
            average_metric_list_train.append(train_ATT)

        else:
            pehe_test, ate_test = self.find_pehe(y0_pred_test, y1_pred_test, data_test)
            pehe_train, ate_train = self.find_pehe(y0_pred_train, y1_pred_train, data_train)

            if self.params['dataset_name'] == 'acic':
                print(kwargs.get('folder_ind'), kwargs.get('count'), 'Pehe Test = ', pehe_test, 'Pehe Train = ',
                      pehe_train)
            else:
                print(kwargs.get('count'), 'Pehe Test = ', pehe_test, ' Pehe Train = ', pehe_train, ' ATE test = ',
                      ate_test,
                      ' ATE train = ', ate_train)

            metric_list_test.append(pehe_test)
            metric_list_train.append(pehe_train)

            average_metric_list_test.append(ate_test)
            average_metric_list_train.append(ate_train)


