def find_params(model_name, dataset_name):
    """SLEARNER"""

    params_SLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 300,
                              'epochs': 300, 'binary': False, 'n_fc': 3, 'verbose': 0, 'val_split': 0.0,
                              'kernel_init': 'RandomNormal', 'max_trials': 10}

    params_SLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 300,
                              'epochs': 300, 'binary': False, 'n_fc': 3, 'verbose': 0, 'val_split': 0.1,
                              'kernel_init': 'GlorotNormal', 'max_trials': 30}

    params_SLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40,
                            'batch_size': 128, 'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 300,
                            'val_split': 0.1, 'epochs': 50, 'binary': True, 'n_fc': 3, 'verbose': 0,
                            'kernel_init': 'RandomNormal', 'max_trials': 20}

    params_SLearner_sum = {'dataset_name': "sum", 'num': 100, 'lr': 1e-3, 'patience': 40,
                           'batch_size': 64, 'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 300,
                           'epochs': 300, 'binary': False, 'n_fc': 3, 'verbose': 0, 'val_split': 0.0,
                           'kernel_init': 'RandomNormal', 'max_trials': 10}

    """TLEARNER"""

    params_TLearner_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-2, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear',
                              'epochs': 300, 'binary': False, 'n_fc': 3, 'hidden_phi': 300,
                              'verbose': 0, 'model_name': 'TLearner',
                              'max_trials': 10, 'kernel_init': 'RandomNormal'}

    params_TLearner_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-2, 'patience': 40,
                              'batch_size': 64, 'reg_l2': .01, 'activation': 'linear',
                              'hidden_phi': 300, 'epochs': 1200, 'binary': False, 'n_fc': 3,
                              'verbose': 0, 'model_name': 'TLearner', 'max_trials': 10, 'kernel_init': 'GlorotNormal',
                              }

    params_TLearner_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-2, 'patience': 40,
                            'max_trials': 20, 'batch_size': 256, 'reg_l2': .01,
                            'activation': 'sigmoid', 'hidden_phi': 64, 'epochs': 30, 'binary': True, 'n_fc': 3,
                            'verbose': 0, 'kernel_init': 'RandomNormal'}

    params_TLearner_sum = {'dataset_name': "sum", 'num': 100, 'lr': 1e-2, 'patience': 40,
                           'batch_size': 32, 'reg_l2': .01, 'activation': 'linear',
                           'epochs': 300, 'binary': False, 'n_fc': 4, 'hidden_phi': 128,
                           'verbose': 0, 'model_name': 'TLearner',
                           'max_trials': 10, 'kernel_init': 'RandomNormal'}

    """TARNET"""

    params_TARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 32,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'hidden_y1': 100,
                            'hidden_y0': 100, 'epochs': 300, 'binary': False, 'n_fc': 3, 'n_fc_y1': 3,
                            'defaults': False,
                            'n_fc_y0': 3, 'verbose': 0, 'kernel_init': 'RandomNormal', 'params': 'params_ihdp_a',
                            'max_trials': 10, 'out_size': 1}

    params_TARnet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 5, 'batch_size': 32,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 200, 'hidden_y1': 100,
                            'defaults': False,
                            'hidden_y0': 100, 'epochs': 300, 'binary': False, 'n_fc': 3, 'n_fc_y0': 3, 'n_fc_y1': 3,
                            'verbose': 0, 'kernel_init': 'GlorotNormal', 'params': 'params_ihdp_b',
                            'max_trials': 10}

    params_TARnet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-2, 'patience': 40, 'batch_size': 256,
                          'reg_l2': .01, 'activation': 'sigmoid', 'hidden_phi': 200, 'hidden_y1': 200,
                          'defaults': False,
                          'hidden_y0': 200, 'epochs': 30, 'binary': True, 'n_fc': 3, 'n_fc_y1': 3, 'n_fc_y0': 3,
                          'val_split': 0.1, 'verbose': 0, 'kernel_init': 'RandomNormal', 'max_trials': 10}

    params_TARnet_sum = {'dataset_name': "sum", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 32,
                         'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 100, 'hidden_y1': 100,
                         'defaults': False,
                         'eye': False,
                         'json': True,
                         'drop': None,
                         'hidden_y0': 100, 'epochs': 100, 'binary': False, 'n_fc': 3, 'n_fc_y1': 3, 'n_fc_y0': 3,
                         'verbose': 0, 'kernel_init': 'RandomNormal', 'params': 'params_ihdp_a',
                         'max_trials': 10}

    """GNN-TARNET"""

    params_GNNTARnet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 64,
                               'gnn_hidden_units': 128, 'gnn_n_fc': 4, 'aggregation_type': "sum", 'eye': True,
                               'combination_type': "add", 'dropout_rate': 0.5, 'normalize': False, 'reg_l2': .01,
                               'activation': "linear", 'hidden_y1': 240, 'hidden_y0': 96, 'defaults': True,
                               'epochs': 300, 'binary': False, 'n_hidden_1': 6, 'n_hidden_0': 7,
                               'json': True,
                               'drop': None,
                               'verbose': 0, 'kernel_init': 'RandomNormal', 'params': 'params_ihdp_a', 'max_trials': 10}

    params_GNNTARnet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 32,
                               'gnn_hidden_units': 128, 'gnn_n_fc': 4, 'aggregation_type': "sum", 'eye': False,
                               'combination_type': "mlp", 'dropout_rate': 0.0, 'normalize': True, 'defaults': True,
                               'reg_l2': .01, 'activation': "linear", 'hidden_y1': 240,
                               'json': True,
                               'drop': None,
                               'infl_y': True,
                               'hidden_y0': 208, 'epochs': 350, 'binary': False, 'n_hidden_1': 6,
                               'n_hidden_0': 4, 'verbose': 0, 'kernel_init': 'GlorotNormal', 'params': 'params_ihdp_b',
                               'max_trials': 10}

    params_GNNTARnet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-2, 'patience': 40, 'batch_size': 64,
                             'gnn_hidden_units': 16, 'gnn_n_fc': 2, 'aggregation_type': "sum", 'eye': True,
                             'combination_type': "add", 'dropout_rate': 0.0, 'normalize': False, 'defaults': True,
                             'reg_l2': .01, 'activation': 'sigmoid', 'hidden_y1': 176,
                             'json': True,
                             'drop': None,
                             'infl_y': True,
                             'hidden_y0': 240, 'epochs': 100, 'binary': True, 'n_hidden_1': 6, 'n_hidden_0': 5,
                             'val_split': 0.1, 'verbose': 0, 'kernel_init': 'RandomNormal', 'max_trials': 10}

    params_GNNTARnet_sum = {'dataset_name': "sum", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 2,
                            'gnn_hidden_units': 16, 'gnn_n_fc': 2, 'aggregation_type': "sum", 'defaults': True,
                            'combination_type': "add", 'dropout_rate': 0.0, 'normalize': False, 'reg_l2': .01,
                            'activation': "linear", 'hidden_y1': 16, 'hidden_y0': 16, 'eye': False,
                            'json': True,
                            'drop': None,
                            'infl_y': False,
                            'epochs': 100, 'binary': False, 'n_hidden_1': 2, 'n_hidden_0': 2, 'notears': False,
                            'verbose': 0, 'kernel_init': 'RandomNormal', 'params': 'params_ihdp_a',
                            'max_trials': 10}

    """TEDVAE"""

    params_TEDVAE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 1024,
                            'reg_l2': .01, 'activation': 'linear', 'latent_dim_z': 20,
                            'num_bin': 19, 'num_cont': 6, 'latent_dim_zt': 10, 'max_trials': 10,
                            'latent_dim_zy': 10, 'epochs': 400, 'binary': False,
                            'verbose': 0, 'kernel_init': 'RandomNormal', 'defaults': True,
                            'n_fc_enc_x': 4, 'hidden_phi_enc_x': 500, 'n_fc_dec_x': 4, 'hidden_phi_dec_x': 500}

    params_TEDVAE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 1024,
                            'reg_l2': .01, 'activation': 'linear', 'hidden_phi': 32, 'latent_dim_z': 15, 'num_bin': 19,
                            'num_cont': 6, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 400, 'binary': False,
                            'n_fc': 5, 'verbose': 0, 'val_split': 0.1, 'kernel_init': 'GlorotNormal', 'defaults': True,
                            'max_trials': 10,
                            'n_fc_enc_x': 4, 'hidden_phi_enc_x': 500, 'n_fc_dec_x': 4, 'hidden_phi_dec_x': 500}

    params_TEDVAE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-3, 'patience': 40, 'batch_size': 256,
                          'reg_l2': .01, 'activation': 'sigmoid', 'latent_dim_z': 15, 'num_bin': 0,
                          'num_cont': 17, 'latent_dim_zt': 15, 'latent_dim_zy': 5, 'epochs': 30, 'binary': True,
                          'verbose': 0, 'kernel_init': 'GlorotNormal', 'defaults': True, 'max_trials': 10,
                          'n_fc_enc_x': 4, 'hidden_phi_enc_x': 500, 'n_fc_dec_x': 4, 'hidden_phi_dec_x': 500}

    params_TEDVAE_sum = {'dataset_name': "sum", 'num': 100, 'lr': 1e-4, 'patience': 40, 'batch_size': 1024,
                         'reg_l2': .01, 'activation': 'linear', 'latent_dim_z': 15,
                         'num_bin': 0, 'num_cont': 25, 'latent_dim_zt': 15,
                         'latent_dim_zy': 5, 'epochs': 400, 'binary': False, 'max_trials': 10,
                         'verbose': 0, 'kernel_init': 'RandomNormal', 'model_name': 'tedvae', 'defaults': True,
                         'n_fc_enc_x': 4, 'hidden_phi_enc_x': 500, 'n_fc_dec_x': 4, 'hidden_phi_dec_x': 500}

    """CFRNET"""

    params_CFRNet_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr': 1e-4, 'patience': 40, 'tuner_batch_size': 256,
                            'batch_size': 1024, 'reg_l2': .01, 'activation': 'linear', 'epochs': 300,
                            'binary': False, 'verbose': 0, 'kernel_init': 'RandomNormal', 'defaults': False,
                            'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3,
                            'hidden_y1': 100, 'max_trials': 10, 'n_fc_t': 3, 'hidden_t': 100
                            }

    params_CFRNet_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr': 1e-3, 'patience': 5,
                            'tuner_batch_size': 256, 'batch_size': 1024, 'reg_l2': .01, 'activation': 'linear',
                            'epochs': 300, 'binary': False, 'verbose': 0, 'kernel_init': 'GlorotNormal',
                            'defaults': False, 'max_trials': 10,
                            'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3,
                            'hidden_y1': 100, 'n_fc_t': 3, 'hidden_t': 100,
                            }

    params_CFRNet_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr': 1e-2, 'patience': 40, 'tuner_batch_size': 512,
                          'batch_size': 1024, 'reg_l2': .01, 'activation': 'sigmoid', 'epochs': 200, 'binary': True,
                          'verbose': 0, 'kernel_init': 'RandomNormal', 'defaults': False,
                          'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 2, 'hidden_y0': 100, 'n_fc_y1': 2,
                          'hidden_y1': 100, 'max_trials': 1,
                          'n_fc_t': 1, 'hidden_t': 1}

    params_CFRNet_sum = {'dataset_name': "sum", 'num': 1, 'lr': 1e-4, 'patience': 40,
                         'tuner_batch_size': 256,
                         'batch_size': 1024, 'reg_l2': .01, 'activation': 'linear', 'epochs': 300,
                         'binary': False, 'verbose': 0, 'kernel_init': 'RandomNormal', 'defaults': True,
                         'n_fc': 3, 'hidden_phi': 200, 'n_fc_y0': 3, 'hidden_y0': 100, 'n_fc_y1': 3,
                         'hidden_y1': 100, 'max_trials': 10,
                         'n_fc_t': 3, 'hidden_t': 100
                         }

    """GANITE"""

    params_GANITE_IHDP_a = {'dataset_name': "ihdp_a", 'num': 100, 'lr_g': 1e-3, 'lr_i': 1e-3, 'patience': 40,
                            'batch_size_g': 64, 'batch_size_i': 64, 'reg_l2': .01, 'activation': 'linear',
                            'defaults': True,
                            'binary': False, 'n_fc_i': 5, 'hidden_phi_i': 8, 'n_fc_g': 5, 'hidden_phi_g': 8,
                            "n_fc_d": 5, "hidden_phi_d": 5, 'epochs_g': 100, 'max_trials': 10,
                            'verbose': 0, 'epochs_i': 50, 'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    params_GANITE_IHDP_b = {'dataset_name': "ihdp_b", 'num': 100, 'lr_g': 1e-3, 'lr_i': 1e-3, 'patience': 40,
                            'batch_size_g': 64, 'batch_size_i': 64, 'reg_l2': .01, 'activation': 'linear',
                            'defaults': True,
                            'binary': False, 'n_fc_i': 5, 'hidden_phi_i': 8, 'n_fc_g': 5, 'hidden_phi_g': 8,
                            "n_fc_d": 5, "hidden_phi_d": 5, 'epochs_g': 100, 'max_trials': 10,
                            'verbose': 0, 'epochs_i': 50, 'val_split': 0.1, 'kernel_init': 'GlorotNormal'}

    params_GANITE_JOBS = {'dataset_name': "jobs", 'num': 100, 'lr_g': 1e-3, 'lr_i': 1e-3, 'patience': 40,
                          'batch_size_g': 128,
                          'batch_size_i': 128, 'reg_l2': .01, 'activation': 'sigmoid', 'binary': True,
                          'n_fc_i': 3, 'hidden_phi_i': 4, 'n_fc_g': 3, 'hidden_phi_g': 4, 'max_trials': 10,
                          "n_fc_d": 3, "hidden_phi_d": 4, 'epochs_g': 100, 'defaults': True,
                          'verbose': 0, 'epochs_i': 50, 'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    params_GANITE_sum = {'dataset_name': "sum", 'num': 100, 'lr_g': 1e-3, 'lr_i': 1e-3, 'patience': 40,
                         'batch_size_g': 64, 'batch_size_i': 64, 'reg_l2': .01, 'activation': 'linear',
                         'defaults': True,
                         'binary': False, 'n_fc_i': 5, 'hidden_phi_i': 8, 'n_fc_g': 5, 'hidden_phi_g': 8,
                         "n_fc_d": 5, "hidden_phi_d": 5, 'epochs_g': 100, 'max_trials': 10,
                         'verbose': 0, 'epochs_i': 50, 'val_split': 0.1, 'kernel_init': 'RandomNormal'}

    """-------------------------------------------------------------"""

    params_TARnet = {'ihdp_a': params_TARnet_IHDP_a, 'ihdp_b': params_TARnet_IHDP_b,
                     'jobs': params_TARnet_JOBS, 'sum': params_TARnet_sum}

    params_GNNTARnet = {'ihdp_a': params_GNNTARnet_IHDP_a, 'ihdp_b': params_GNNTARnet_IHDP_b,
                        'jobs': params_GNNTARnet_JOBS, 'sum': params_GNNTARnet_sum}

    params_TEDVAE = {'ihdp_a': params_TEDVAE_IHDP_a, 'ihdp_b': params_TEDVAE_IHDP_b,
                     'jobs': params_TEDVAE_JOBS, 'sum': params_TEDVAE_sum}

    params_GANITE = {'ihdp_a': params_GANITE_IHDP_a, 'ihdp_b': params_GANITE_IHDP_b,
                     'jobs': params_GANITE_JOBS, 'sum': params_GANITE_sum}

    params_TLearner = {'ihdp_a': params_TLearner_IHDP_a, 'ihdp_b': params_TLearner_IHDP_b,
                       'jobs': params_TLearner_JOBS, 'sum': params_TLearner_sum}

    params_SLearner = {'ihdp_a': params_SLearner_IHDP_a, 'ihdp_b': params_SLearner_IHDP_b,
                       'jobs': params_SLearner_JOBS, 'sum': params_SLearner_sum}

    params_CFRNet = {'ihdp_a': params_CFRNet_IHDP_a, 'ihdp_b': params_CFRNet_IHDP_b,
                     'jobs': params_CFRNet_JOBS, 'sum': params_CFRNet_sum}

    """-------------------------------------------------------------"""

    params = {'TARnet': params_TARnet, 'GNNTARnet': params_GNNTARnet,
              'TEDVAE': params_TEDVAE, 'TLearner': params_TLearner, 'SLearner': params_SLearner,
              'CFRNet': params_CFRNet, 'GANITE': params_GANITE}

    return params[model_name][dataset_name]
