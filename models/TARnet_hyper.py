import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from models.CausalModel import *
from utils.layers import FullyConnected
import keras_tuner as kt
from utils.callback import callbacks
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from os.path import exists
import shutil
class HyperTarnet(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = TarnetModel(name='tarnet', params=self.params, hp=hp)
        optimizer = SGD(learning_rate=self.params['lr'], momentum=0.9)
        model.compile(optimizer=optimizer,
                      loss=self.regression_loss,
                      metrics=self.regression_loss)
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class TarnetModel(Model):
    def __init__(self, name, params, hp, **kwargs):
        super(TarnetModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.hidden_phi, final_activation='elu',
                                 out_size=self.hidden_phi, kernel_init=params['kernel_init'], kernel_reg=None,
                                 name='fc')
        self.n_fc_y0 = hp.Int('n_fc_y0', min_value=2, max_value=10, step=1)
        self.hidden_y0 = hp.Int('hidden_y0', min_value=16, max_value=512, step=16)
        self.pred_y0 = FullyConnected(n_fc=self.n_fc_y0, hidden_phi=self.hidden_y0,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y0')

        self.n_fc_y1 = hp.Int('n_fc_y1', min_value=2, max_value=10, step=1)
        self.hidden_y1 = hp.Int('hidden_y1', min_value=16, max_value=512, step=16)
        self.pred_y1 = FullyConnected(n_fc=self.n_fc_y1, hidden_phi=self.hidden_y1,
                                      final_activation=params['activation'], out_size=1,
                                      kernel_init=params['kernel_init'],
                                      kernel_reg=regularizers.l2(params['reg_l2']), name='y1')

    def call(self, inputs):
        x = self.fc(inputs)
        y0_pred = self.pred_y0(x)
        y1_pred = self.pred_y1(x)
        concat_pred = tf.concat([y0_pred, y1_pred], axis=-1)
        return concat_pred


class TARnetHyper(CausalModel):

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.directory_name = None
        self.project_name = None
        self.best_hps = None

    def fit_tuner(self, x, y, t, seed):
        setSeed(seed)
        t = tf.cast(t, dtype=tf.float32)
        directory_name = 'params_' + self.params['tuner_name'] + '/' + self.params['dataset_name']

        if self.dataset_name == 'gnn':
            directory_name = directory_name + f'/{self.params["model_name"]}'
            project_name = str(self.folder_ind)
        else:
            project_name = self.params["model_name"]

        hp = kt.HyperParameters()

        self.directory_name = directory_name
        self.project_name = project_name

        hypermodel = HyperTarnet(params=self.params)
        objective = kt.Objective("val_regression_loss", direction="min")
        tuner = self.define_tuner(hypermodel, hp, objective, directory_name, project_name)

        yt = tf.concat([y, t], axis=1)
        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_regression_loss', patience=5)]
        tuner.search(x, yt, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=self.params['verbose'])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if self.params['defaults']:
            best_hps.values = {'n_fc': self.params['n_fc'],
                                'hidden_phi': self.params['hidden_phi'],
                               'n_fc_y0': self.params['n_fc_y0'],
                               'hidden_y0': self.params['hidden_y0'],
                               'n_fc_y1': self.params['n_fc_y1'],
                               'hidden_y1': self.params['hidden_y1']}
        self.best_hps = best_hps

        return

    def fit_model(self, x, y, t, count, seed):
        setSeed(seed)
        t = tf.cast(t, dtype=tf.float32)

        tuner = kt.RandomSearch(
            HyperTarnet(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name,
            seed=0)

        best_hps = self.best_hps
        model = tuner.hypermodel.build(best_hps)
        yt = tf.concat([y, t], axis=1)
        model.fit(x=x, y=yt,
                  callbacks=callbacks('regression_loss'),
                  validation_split=0.0,
                  epochs=self.params['epochs'],
                  batch_size=self.params['batch_size'],
                  verbose=self.params['verbose'])

        self.sparams = f"""n_fc={best_hps.get('n_fc')} hidden_phi = {best_hps.get('hidden_phi')}
              hidden_y1 = {best_hps.get('hidden_y1')} n_fc_y1 = {best_hps.get('n_fc_y1')}
              hidden_y0 = {best_hps.get('hidden_y0')}  n_fc_y0 = {best_hps.get('n_fc_y0')}"""


        if count == 0 and self.folder_ind == 0:
            print(f"""The hyperparameter search is complete. The optimal hyperparameters are
                   {self.sparams}""")
            print(model.summary())

        return model

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)

    def train_and_evaluate(self, metric_list_train, metric_list_test, average_metric_list_train, average_metric_list_test, **kwargs):
        # kwargs['count'] = 87
        data_train, data_test = self.load_data(**kwargs)

        count = kwargs.get('count')
        self.folder_ind = kwargs.get('folder_ind') - 1

        tracker_test, tracker_train = self.get_trackers(count)

        if self.params['binary']:
            with tracker_train:
                # fit tuner on the first dataset
                self.fit_tuner(data_train['x'], data_train['y'], data_train['t'], seed=0)
                model = self.fit_model(data_train['x'], data_train['y'], data_train['t'], count, seed=0)
            self.emission_train.append(tracker_train.final_emissions)
        else:
            with tracker_train:
                # fit tuner on the first dataset
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

        # gain_curve_test = self.cumulative_gain(data_test_pred, "cate", y="y", t="t")
        # gain_curve_train = self.cumulative_gain(data_train_pred, "cate", y="y", t="t")
        # self.plot_cumulative_gain(gain_curve_test, gain_curve_train, data_test_pred)

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

            if self.params['dataset_name'] == 'acic' or self.params['dataset_name'] == 'gnn':
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
