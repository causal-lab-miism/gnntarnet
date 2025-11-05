from tensorflow.keras import Model
from models.CausalModel import *
from utils.layers import FullyConnected
from utils.callback import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from os.path import exists
import shutil

class HyperTLearner(kt.HyperModel, CausalModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build(self, hp):
        model = TModel(name='tlearner', params=self.params, hp=hp)
        lr = hp.Choice("lr", [1e-3, 1e-2, 1e-4])
        optimizer = Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [64, 128, 256, 512]),
            **kwargs,
        )


class TModel(Model):
    def __init__(self, params, hp, name='tlearner', **kwargs):
        super(TModel, self).__init__(name=name, **kwargs)
        self.params = params
        self.n_fc = hp.Int('n_fc', min_value=2, max_value=10, step=1)
        self.hidden_phi = hp.Int('hidden_phi', min_value=16, max_value=512, step=16)
        self.fc = FullyConnected(n_fc=self.n_fc, hidden_phi=self.n_fc,
                                 final_activation=params['activation'], out_size=1,
                                 kernel_init=params['kernel_init'],
                                 kernel_reg=regularizers.l2(params['reg_l2']), name='fc')

    def call(self, inputs):
        # for reproducibility
        x = self.fc(inputs)
        return x


class TLearner(CausalModel):
    """
    This class can be used to train and create stacked model
    for IHDP dataset setting "b"
    """

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.directory_name = None
        self.project_name1 = None
        self.project_name0 = None
        self.best_hps_0 = None
        self.best_hps_1 = None

    def fit_tuner(self, x, y, t, seed):
        directory_name = 'params_' + self.params['tuner_name'] + '/'\
                         + self.params['dataset_name'] + f'/{self.params["model_name"]}'
        setSeed(seed)

        # if self.dataset_name == 'acic':
        #     directory_name = directory_name + f'/{self.folder_ind}'

        self.directory_name = directory_name
        self.project_name1 = 'Model1'

        # if exists(directory_name + '/' + project_name1) and count == 0:
        #     shutil.rmtree(directory_name + '/' + project_name1)

        hp1 = kt.HyperParameters()

        hypermodel_1 = HyperTLearner(params=self.params)
        objective_1 = kt.Objective("val_mse", direction="min")
        tuner_1 = self.define_tuner(hypermodel_1, hp1, objective_1, self.directory_name, self.project_name1)


        self.project_name0 = 'Model0'

        # if exists(directory_name + '/' + project_name0) and count == 0:
        #     shutil.rmtree(directory_name + '/' + project_name0)

        hp0 = kt.HyperParameters()
        hypermodel_0 = HyperTLearner(params=self.params)
        objective_0 = kt.Objective("val_mse", direction="min")
        tuner_0 = self.define_tuner(hypermodel_0, hp0, objective_0, self.directory_name, self.project_name0)

        t0_ind = np.squeeze(t == 0)
        t1_ind = np.squeeze(t == 1)

        x0 = x[t0_ind]
        x1 = x[t1_ind]

        y0 = y[t0_ind]
        y1 = y[t1_ind]
        setSeed(seed)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='val_mse', patience=5)]
        tuner_0.search(x0, y0, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=self.params['verbose'])
        tuner_1.search(x1, y1, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=self.params['verbose'])

        # Get the optimal hyperparameters
        best_hps_0 = tuner_0.get_best_hyperparameters(num_trials=10)[0]
        best_hps_1 = tuner_1.get_best_hyperparameters(num_trials=10)[0]

        if self.params['defaults']:
            defaults = {'n_fc': self.params['n_fc'],
                        'hidden_phi': self.params['hidden_phi'],
                        'lr': self.params['lr'],
                        'batch_size': self.params['batch_size']}

            best_hps_0.values = defaults
            best_hps_1.values = defaults

        self.best_hps_0 = best_hps_0
        self.best_hps_1 = best_hps_1

        return


    def fit_model(self, x, y, t, count, seed):
        setSeed(seed)

        t0_ind = np.squeeze(t == 0)
        t1_ind = np.squeeze(t == 1)

        x0 = x[t0_ind]
        x1 = x[t1_ind]

        y0 = y[t0_ind]
        y1 = y[t1_ind]
        setSeed(seed)

        tuner_0 = kt.RandomSearch(
            HyperTLearner(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name0,
            seed=0)

        tuner_1 = kt.RandomSearch(
            HyperTLearner(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name1,
            seed=0)

        # Get the optimal hyperparameters
        best_hps_0 = self.best_hps_0
        best_hps_1 = self.best_hps_1

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model0 = tuner_0.hypermodel.build(best_hps_0)
        model1 = tuner_1.hypermodel.build(best_hps_1)

        model0.fit(x0, y0, epochs=self.params['epochs'], callbacks=callbacks('loss'),
                   batch_size=best_hps_0.get('batch_size'), validation_split=0.0,
                   verbose=self.params['verbose'])

        model1.fit(x1, y1, epochs=self.params['epochs'], callbacks=callbacks('loss'),
                   batch_size=best_hps_1.get('batch_size'), validation_split=0.0,
                   verbose=self.params['verbose'])

        hyperparameters0 = f"""n_fc_0={best_hps_0.get('n_fc')} - hidden_phi_0 = {best_hps_0.get('hidden_phi')} -
                  learning rate={best_hps_0.get('lr')} - batch size = {best_hps_0.get('batch_size')}"""
        hyperparameters1 = f"""np_fc_1={best_hps_1.get('n_fc')} - hidden_phi_1 = {best_hps_1.get('hidden_phi')} -
                  learning rate={best_hps_1.get('lr')} - batch size = {best_hps_1.get('batch_size')} """

        if count == 0 and self.folder_ind == 0:
            print("The hyperparameter search is complete. The optimal hyperparameters are " + hyperparameters0)

            print("The hyperparameter search is complete. The optimal hyperparameters are " + hyperparameters1)


        self.sparams = hyperparameters1 + hyperparameters0
        return [model0, model1]

    @staticmethod
    def evaluate(x_test, models):
        concat_pred = list()
        for model in models:
            y_pred = model.predict(x_test)
            concat_pred.append(y_pred)
        return tf.concat((concat_pred[0], concat_pred[1]), axis=1)

    def train_and_evaluate(self, metric_list_train, metric_list_test, average_metric_list_train, average_metric_list_test, **kwargs):
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

