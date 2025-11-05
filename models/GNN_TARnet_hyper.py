import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from keras import regularizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

from utils.layers import FullyConnected, VariationalFullyConnected, Convolutional1D, LocallyConnected
# from models.CausalModel import CausalModel
from models.CausalModel import *
import keras_tuner as kt
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, EarlyStopping
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
import os, sys
import tensorflow_probability as tfp
tf.get_logger().setLevel(logging.ERROR)
from causalnex.structure.notears import from_numpy
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_accuracy

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
import json
from os.path import exists

os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt

plt.show()
import json
from os.path import exists
import shutil


def callbacks(rlr_monitor):
    cbacks = [
        TerminateOnNaN(),
        ReduceLROnPlateau(monitor=rlr_monitor, factor=0.5, patience=5, verbose=0, mode='auto',
                          min_delta=0., cooldown=0, min_lr=1e-8),
        EarlyStopping(monitor='val_regression_loss', patience=40, min_delta=0., restore_best_weights=False)
    ]
    return cbacks


class HyperGNNTarnet(kt.HyperModel, CausalModel):
    """
    HyperGNNTarnet

    Keras Tuner HyperModel wrapper for constructing and compiling a GNNTARnetModel configured
    for causal regression tasks. This class adapts a set of static parameters together with a
    Keras Tuner HyperParameters object to produce a compiled Keras Model ready for tuning
    and training.

    Usage:
    - Instantiate with a params dictionary describing fixed training and model settings.
    - Pass an instance to a Keras Tuner (e.g., kt.RandomSearch) or call build(hp) directly to
        get a compiled GNNTARnetModel.
    - Use fit(hp, model, ...) to train the model; this method injects the configured batch_size.

    Args:
            params (dict):
                    Dictionary of configuration values required by the underlying GNNTARnetModel and
                    for training. Required keys:
                    - 'lr' (float): learning rate used to construct the SGD optimizer.
                    - 'batch_size' (int): batch size used by the fit method.
                    Additional keys supported by GNNTARnetModel may be provided and will be forwarded.
            name (str, optional):
                    Name assigned to the produced GNNTARnetModel. Defaults to 'gnn_tarnet'.

    Attributes:
            params (dict):
                    The same params dict passed at construction; stored for use when building/fitting.
            name (str):
                    Model name used when creating the GNNTARnetModel instance.

    Methods:
            build(hp):
                    Build and return a compiled GNNTARnetModel using the provided HyperParameters
                    instance (hp) together with self.params. The model is compiled using:
                    - optimizer: SGD(learning_rate=params['lr'], nesterov=True, momentum=0.9)
                    - loss: self.regression_loss (provided by CausalModel)
                    - metrics: self.regression_loss
                    - run_eagerly: False

                    Args:
                            hp: a keras_tuner.HyperParameters instance used to configure hyperparameters
                                    of the GNNTARnetModel.

                    Returns:
                            A compiled tf.keras.Model (GNNTARnetModel).

            fit(hp, model, *args, **kwargs):
                    Convenience wrapper around model.fit that enforces the configured batch_size.
                    It forwards all positional and keyword arguments to model.fit and injects
                    batch_size=self.params['batch_size'] unless overridden explicitly in kwargs.

                    Args:
                            hp: HyperParameters (kept for API compatibility; not used directly here).
                            model: A compiled tf.keras.Model returned by build(hp).
                            *args: Positional arguments forwarded to model.fit.
                            **kwargs: Keyword arguments forwarded to model.fit. If 'batch_size' is provided
                                                in kwargs it will override self.params['batch_size'].

    Notes:
    - This class inherits from kt.HyperModel to integrate with Keras Tuner and from
        CausalModel to reuse causal modeling utilities such as regression_loss.
    - The implementation assumes that GNNTARnetModel accepts 'name', 'hp', and 'params'
        arguments at construction.
    - The SGD optimizer is configured with momentum=0.9 and Nesterov momentum enabled.

    Example:
            params = {'lr': 1e-3, 'batch_size': 128, ...}
            hypermodel = HyperGNNTarnet(params)
            tuner = kt.RandomSearch(hypermodel, objective='val_loss', max_trials=10, ...)
            tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

    """
    def __init__(self, params, name='gnn_tarnet'):
        super().__init__()
        self.params = params
        self.name = name

    def build(self, hp):
        momentum = 0.9

        model = GNNTARnetModel(
            name=self.name,
            hp=hp,
            params=self.params,
        )

        model.compile(optimizer=SGD(learning_rate=self.params['lr'], nesterov=True, momentum=momentum),
                      loss=self.regression_loss,
                      metrics=self.regression_loss, run_eagerly=False
                      )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=self.params['batch_size'],
            **kwargs,
        )


class GraphConvLayer(layers.Layer):
    """
    GraphConvLayer

    A graph convolutional layer that prepares node messages with a small MLP,
    aggregates neighbor messages using unsorted segment operations, and updates
    node representations via a configurable combination function and an MLP.

    This layer is designed to work with inputs describing a graph's node features
    and an edge list. It supports different aggregation strategies ("sum", "mean",
    "max") and combination strategies ("concat", "add", "mlp"). An optional L2
    normalization can be applied to the updated node embeddings.

    Parameters
    ----------
    params : dict
        Configuration dictionary. Expected keys include:
          - 'aggregation_type' (str): One of "sum", "mean", "max".
          - 'combination_type' (str): One of "concat", "add", "mlp".
          - 'normalize' (bool): If True, L2-normalize the output embeddings.
          - 'kernel_init', 'dropout_rate', ... : Other keys passed to internal
            FullyConnected layers (used for initialization, dropout, etc.).
    g n n_n_fc : int
        Number of fully-connected layers (n_fc) used inside the internal MLPs.
    g n n_hidden_units : int
        Hidden dimension used by the internal MLPs and the final output
        representation size produced by this layer.

    Attributes
    ----------
    ffn_prepare : FullyConnected
        MLP used to map parent (source) node representations into messages
        that will be aggregated for each target node.
    update_fn : FullyConnected
        MLP used to combine node representations and aggregated messages into
        the updated node embeddings.

    Input and expected shapes
    -------------------------
    The layer expects a single argument to call() in the form of a tuple:
        (node_representations, edges, edge_weights)

    - node_representations : tf.Tensor
        Node feature tensor. The node axis is expected to be index 1 when a batch
        dimension is present (e.g. [batch, num_nodes, D]) or it may be a 2-D tensor
        [num_nodes, D]. The layer gathers parent node features along the node axis
        (axis=1 in the implementation) to form per-edge parent representations.
    - edges : tf.Tensor, shape [num_edges, 2]
        Each row is [parent_index, node_index] describing a directed edge from a
        parent (source) node to a child/target node. Indices are used with
        tf.gather and as segment ids for aggregation.
    - edge_weights : tf.Tensor or None
        Provided for API compatibility but not used by this implementation.
        (Reserved for future use if weighted aggregation is needed.)

    Internal method behaviors
    -------------------------
    - prepare(node_representations):
        Applies ffn_prepare to parent node representations (per-edge) to produce
        messages of shape matching the expected message embedding dimension.

    - aggregate(node_indices, neighbour_messages, node_representations):
        Aggregates messages per target node according to params['aggregation_type'].
        Uses tf.math.unsorted_segment_sum, _mean, or _max. The number of segments is
        inferred from the number of nodes along the node axis of node_representations.

    - update(node_representations, aggregated_messages):
        Combines the original node representations and the aggregated messages using
        one of:
          - "concat": concatenates along the feature axis,
          - "add": element-wise addition,
          - "mlp": element-wise multiplication (treated as an elementwise gating).
        The combined tensor is then passed through update_fn. If params['normalize']
        is True, the result is L2-normalized along the last axis.

    Returns
    -------
    tf.Tensor
        Updated node embeddings with the same leading dimensions and node axis as
        the input node_representations. The final embedding dimensionality is the
        output dimension of the internal update_fn (typically gnn_hidden_units).

    Raises
    ------
    ValueError
        If params['aggregation_type'] or params['combination_type'] is not one of
        the supported values.

    Notes
    -----
    - The implementation assumes that edges are indexed with respect to the node
      axis used by tf.gather (axis=1 in the current code). If you pass a batched
      node_representations, ensure edges are correct for that layout.
    - edge_weights are accepted but currently ignored by the aggregation logic.
    - The class composes small FullyConnected modules (ffn_prepare and update_fn)
      rather than writing explicit dense layers inline, so behavior (activation,
      dropout, batch-norm, initializers) is controlled via the params dictionary
      and gnn_n_fc / gnn_hidden_units constructor arguments.
      """
    def __init__(
            self,
            params,
            gnn_n_fc,
            gnn_hidden_units,
            *args,
            **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.params = params
        self.aggregation_type = params['aggregation_type']
        self.combination_type = params['combination_type']
        self.normalize = params['normalize']
        self.gnn_n_fc = gnn_n_fc
        self.gnn_hidden_units = gnn_hidden_units

        self.ffn_prepare = FullyConnected(n_fc=self.gnn_n_fc, hidden_phi=self.gnn_hidden_units,
                                          final_activation='elu', out_size=self.gnn_hidden_units,
                                          kernel_init=self.params['kernel_init'], use_bias=False,
                                          kernel_reg=None, dropout=False, dropout_rate=self.params['dropout_rate'],
                                          name='ffn_prepare')

        self.update_fn = FullyConnected(n_fc=self.gnn_n_fc, hidden_phi=self.gnn_hidden_units,
                                        final_activation=None, out_size=self.gnn_hidden_units, use_bias=True,
                                        kernel_init=self.params['kernel_init'], batch_norm=False,
                                        kernel_reg=None, dropout=False, dropout_rate=self.params['dropout_rate'],
                                        name='update_fn')

    def prepare(self, node_representations):
        # node_representations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_representations)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_representations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        num_nodes = node_representations.shape[1]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(tf.transpose(neighbour_messages, [1, 0, 2]),
                                                              node_indices, num_segments=num_nodes)
            aggregated_message = tf.transpose(aggregated_message, [1, 0, 2])
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(tf.transpose(neighbour_messages, [1, 0, 2]),
                                                               node_indices, num_segments=num_nodes)
            aggregated_message = tf.transpose(aggregated_message, [1, 0, 2])
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(neighbour_messages,
                                                              node_indices, num_segments=num_nodes)
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")
        return aggregated_message

    def update(self, node_representations, aggregated_messages):
        # node_representations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "concat":
            # Concatenate the node_representations and aggregated_messages.
            h = tf.concat([node_representations, aggregated_messages], axis=2)
        elif self.combination_type == "add":
            # Add node_representations and aggregated_messages.
            h = node_representations + aggregated_messages
        elif self.combination_type == "mlp":
            h = node_representations * aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        node_embeddings = self.update_fn(h)
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_representations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """
        node_representations, edges, edge_weights = inputs
        # Get node_indices (source) and parent_indices (target) from edges.
        parent_indices, node_indices = edges[:, 0], edges[:, 1]
        parents_repesentations = tf.gather(node_representations, parent_indices, axis=1)
        # Prepare the messages of the parents.
        parent_messages = self.prepare(parents_repesentations)
        # Aggregate the parents messages.
        aggregated_messages = self.aggregate(node_indices, parent_messages, node_representations)
        return self.update(node_representations, aggregated_messages)


class Embedding(Model):
    """
    Embedding(params, vector_size, num_neurons)

    A TensorFlow Model that constructs a separate small fully-connected subnetwork for
    each dimension of an input vector and produces a per-dimension embedding.

    This class is intended to be a lightweight way to learn an independent
    transformation for each input feature/dimension. For each i in [0, vector_size),
    a FullyConnected subnetwork (created with the supplied hyperparameters) is
    applied to the i-th column of the input batch. The per-dimension outputs are
    then stacked to form a batched tensor of embeddings.

    Args:
        params (dict):
            Dictionary of hyperparameters forwarded to each FullyConnected instance.
            Expected keys (examples): 'kernel_init' (initializer spec). Other keys
            may be referenced by the FullyConnected implementation.
        vector_size (int):
            Number of independent input dimensions. The model will create one
            FullyConnected subnetwork per input dimension.
        num_neurons (int):
            Output size (number of neurons) of each FullyConnected subnetwork. Each
            input dimension i is mapped to a vector of length num_neurons.

    Attributes:
        vector_size (int): same as the vector_size argument.
        num_neurons (int): same as the num_neurons argument.
        params (dict): same as the params argument.
        networks (list): list of FullyConnected instances, length == vector_size.
            networks[i] is applied to the i-th input feature.

    Call signature:
        call(inputs) -> tf.Tensor

    Inputs:
        inputs (tf.Tensor): a 2-D tensor with shape (batch_size, vector_size)
            (or a shape where the second dimension equals vector_size). Each column
            corresponds to one input feature that will be fed into its matching
            FullyConnected subnetwork. The dtype should be compatible with the
            FullyConnected layer (e.g. tf.float32).

    Returns:
        tf.Tensor: a 3-D tensor of shape (batch_size, vector_size, num_neurons)
            where output[:, i, :] is the embedding produced by networks[i] for the
            i-th input feature across the batch.

    Example:
        Given vector_size=3 and num_neurons=8, calling the model with an input
        tensor of shape (B, 3) returns a tensor of shape (B, 3, 8), containing
        per-feature learned embeddings.

    Notes:
        - Each feature has its own separate subnetwork and parameters; no weight
          sharing occurs across input dimensions.
        - The class subclasses tf.keras.Model and is therefore compatible with the
          Keras training/evaluation APIs (compile, fit, evaluate, predict).
        - The user is responsible for ensuring the FullyConnected layer accepts
          the 1-D per-feature slices passed during call (implementation-specific
          shape behavior of FullyConnected may require inputs of shape (batch, 1)
          instead of (batch,), in which case a reshape or explicitly shaped
          inputs should be used).
    """
    def __init__(self, params, vector_size, num_neurons):
        super(Embedding, self).__init__()
        self.vector_size = vector_size
        self.num_neurons = num_neurons
        self.params = params
        self.networks = []
        for i in range(vector_size):
            x = FullyConnected(n_fc=1, hidden_phi=1,
                                      final_activation=None, out_size=self.num_neurons,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y'+str(i))
            self.networks.append(x)

    def call(self, inputs):
        outputs = []
        for i, layer in enumerate(self.networks):
            x_i = inputs[:, i]
            x = layer(x_i)
            outputs.append(x)
        outputs = tf.stack(outputs, axis=1)
        return outputs

class GNNTARnetModel(Model):
    """
    GNNTARnetModel(name, params, hp, *args, **kwargs)

    Keras Model implementing a two-stage graph convolutional TARNet-style architecture
    that produces two potential outcome predictions (y0 and y1) per example.

    High-level behavior
    - Embeds an input node-index tensor into node feature vectors.
    - Applies two GraphConvLayer blocks with residual (skip) connections.
    - Selects a subset of nodes believed to influence the outcome (params['influence_y']).
    - Flattens the selected node features and feeds them into two separate
        fully-connected heads to predict y0 and y1.
    - Concatenates the two scalar predictions into a single output tensor of shape
        (batch_size, 2).

    Args
    - name (str): A name identifier for the model instance.
    - params (dict): Dictionary of model configuration values. Required keys:
            - 'edges': Graph connectivity information consumed by GraphConvLayer
                 (format/type expected by your GraphConvLayer implementation).
            - 'weights': Weights or configuration used by the graph conv layers.
            - 'num_nodes' (int): Number of nodes per example (used by Embedding).
            - 'activation' (str or callable): Activation used in prediction heads.
            - 'kernel_init' (str or callable): Kernel initializer used in prediction heads.
            - 'influence_y' (list or 1-D array of int): Indices of nodes to select
                 before flattening and prediction.
        The GraphConvLayer and Embedding used here may also expect other fields
        in params; provide whatever those components require.
    - hp: A hyperparameter object (e.g., keras-tuner HyperParameters) used to
        sample architecture hyperparameters. This class reads the following hyperparameters:
            - gnn_n_fc: number of fully-connected layers inside graph conv block
                (hp.Int('gnn_n_fc', min_value=2, max_value=10, step=1)).
            - gnn_hidden_units: hidden size used inside graph conv blocks
                (hp.Int('gnn_hidden_units', min_value=16, max_value=256, step=16)).
            - n_hidden_0, hidden_y0: depth and width of the y0 fully-connected head.
            - n_hidden_1, hidden_y1: depth and width of the y1 fully-connected head.
    - *args, **kwargs: forwarded to the base Model initializer.

    Attributes (not exhaustive)
    - params (dict): stored params dict passed at init.
    - model_name (str): same as name argument.
    - edges: convenience reference to params['edges'].
    - gnn_weights: convenience reference to params['weights'].
    - gnn_n_fc, gnn_hidden_units, n_hidden_0, hidden_y0, n_hidden_1, hidden_y1:
        hyperparameters sampled from hp.
    - conv1, conv2: two GraphConvLayer instances used in sequence with residuals.
    - embedding: Embedding layer mapping node indices to node feature vectors.
    - pred_y0, pred_y1: FullyConnected head modules that each output a scalar.
        These heads use L2 kernel regularization with weight 0.01 by default.
    - flatten: layers.Flatten used to flatten selected node features before heads.

    Call signature and tensor shapes
    - call(inputs)
        - inputs: Tensor or array representing per-example node indices or node inputs
            expected by the Embedding layer. Typical shape: (batch_size, num_nodes).
        - Embedding produces: (batch_size, num_nodes, gnn_hidden_units).
        - After each GraphConvLayer: (batch_size, num_nodes, gnn_hidden_units).
        - Node selection: tf.gather(..., axis=1) selects indices from params['influence_y'],
            resulting in shape (batch_size, n_selected_nodes, gnn_hidden_units).
        - Flatten: produces (batch_size, n_selected_nodes * gnn_hidden_units).
        - pred_y0 and pred_y1 each produce (batch_size, 1).
        - Final output: tf.concat([y0_pred, y1_pred], axis=-1) -> (batch_size, 2).

    Notes and implementation details
    - Two residual connections are used: output of each GraphConvLayer is added
        to its input (skip connection) before feeding to the next layer.
    - The model assumes GraphConvLayer accepts a tuple (node_features, edges, None)
        consistent with this code's call pattern.
    - The prediction heads apply L2 regularization (regularizers.l2(0.01)). The
        activation and initializer come from params.
    - This class is designed for counterfactual / treatment effect style tasks
        where two potential outcomes are predicted per sample.
    - Ensure params['influence_y'] contains valid node indices (0 <= idx < num_nodes).
    """
    def __init__(
            self,
            name,
            params,
            hp,
            *args,
            **kwargs,
    ):
        super(GNNTARnetModel, self).__init__(*args, **kwargs)
        self.params = params
        self.model_name = name
        self.edges = params['edges']
        self.gnn_weights = params['weights']
        # self.gnn_n_fc = self.params['gnn_n_fc']
        # self.gnn_hidden_units = self.params['gnn_hidden_units']
        self.gnn_n_fc = hp.Int('gnn_n_fc', min_value=2, max_value=10, step=1)
        self.gnn_hidden_units = hp.Int('gnn_hidden_units', min_value=16, max_value=256, step=16)
        self.n_hidden_0 = hp.Int('n_hidden_0', min_value=2, max_value=10, step=1)
        self.hidden_y0 = hp.Int('hidden_y0', min_value=16, max_value=256, step=16)
        self.n_hidden_1 = hp.Int('n_hidden_1', min_value=2, max_value=10, step=1)
        self.hidden_y1 = hp.Int('hidden_y1', min_value=16, max_value=256, step=16)

        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            params=self.params,
            gnn_n_fc=self.gnn_n_fc,
            gnn_hidden_units=self.gnn_hidden_units,
            name="graph_conv1"
        )

        # # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            params=self.params,
            gnn_n_fc=self.gnn_n_fc,
            gnn_hidden_units=self.gnn_hidden_units,
            name="graph_conv2"
        )

        self.embedding = Embedding(params=self.params, vector_size=self.params['num_nodes'], num_neurons=self.gnn_hidden_units)

        self.pred_y0 = FullyConnected(n_fc=self.n_hidden_0, hidden_phi=self.hidden_y0,
                                      final_activation=self.params['activation'], out_size=1,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y0')

        self.pred_y1 = FullyConnected(n_fc=self.n_hidden_1, hidden_phi=self.hidden_y1,
                                      final_activation=self.params['activation'], out_size=1,
                                      kernel_init=self.params['kernel_init'],
                                      kernel_reg=regularizers.l2(.01), name='pred_y1')


        self.flatten = layers.Flatten()

    def call(self, inputs):
        x = inputs
        x = self.embedding(x)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, None))
        # # Skip connection.
        x = x1 + x
        # # # # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, None))
        x = x2 + x
        # # # use info about nodes influencing the outcome
        x = tf.gather(x, self.params['influence_y'], axis=1)
        # # Flatten
        x = self.flatten(x)
        # Make a prediction
        y0_pred = self.pred_y0(x)
        y1_pred = self.pred_y1(x)
        # Concatenate the result and return
        concat_pred = tf.concat([y0_pred, y1_pred], axis=-1)
        return concat_pred


class GNNTARnetHyper(CausalModel):
    """
    GNNTARnetHyper

    High-level wrapper for creating, tuning, training and evaluating a GNN-based TARNet causal model.
    This class extends a generic CausalModel and integrates graph loading, data preparation, hyperparameter
    tuning via Keras Tuner, model training, and evaluation utilities tailored for GNN-TARnet experiments.

    Attributes
    ----------
    params : dict
        Configuration dictionary controlling model architecture, training, dataset and tuner behavior.
        Common/expected keys (not exhaustive):
          - 'tuner_name', 'dataset_name', 'model_name' : str identifiers for tuning and checkpointing
          - 'json' : bool, whether graphs are stored as JSON (True) or CSV (False)
          - 'eye' : bool, if True use identity adjacency (no edges)
          - 'binary' : bool, whether outcome is binary (affects which y is used)
          - 'defaults' : bool, if True skip tuner results and use provided default hparams
          - 'tuner' : tuner class or callable used in fit_model to reload tuner
          - training hyperparams: 'epochs', 'batch_size', 'verbose'
          - architecture placeholders: 'gnn_n_fc', 'gnn_hidden_units', 'n_hidden_0', 'hidden_y0', 'n_hidden_1', 'hidden_y1'
        The object will also set and update entries such as:
          - 'edges', 'weights', 'num_edges', 'num_nodes', 'influence_y'

    directory_name : str or None
        Path prefix used by the tuner when saving experiments (set during fit_tuner).

    project_name : str or None
        Project name used by the tuner (set during fit_tuner).

    sparams : str
        Human-readable string summarizing the selected hyperparameters after tuning.

    Notes on expected data shapes and graph formats
    -----------------------------------------------
    - x arrays: typically of shape (num_samples, num_nodes) before expansion; many methods expand to
      (num_samples, num_nodes, 1) for model input.
    - t arrays: treatment arrays of shape (num_samples, 1) or compatible shape cast to float32.
    - y / ys: outcome arrays. If params['binary'] is True, data_train['y'] is used; else data_train['ys'].
    - edges: when JSON format is used, expected keys include 'from', 'to', 'influence_y', and 'weights'.
      When CSV/array format is used, edges are an array-like representation of edge index pairs.
    - edge_weights: expected as an array of shape (num_edges, 1) (weights normalized per-file codepath).

    Public Methods
    --------------
    fit_tuner(seed=0, **kwargs)
        Run a hyperparameter search (Keras Tuner) for the HyperGNNTarnet hypermodel.
        Required kwargs:
          - x: input features for tuning (typically training x)
          - y: outcome(s)
          - t: treatment(s)
          - edges: edge list/array for the graph
          - weights: per-edge weights array
        Behavior:
          - Concatenates y and t for multi-output regression training target.
          - Sets various self.params entries (edges, weights, num_edges, num_nodes).
          - Constructs a HyperGNNTarnet hypermodel and invokes a tuner (via define_tuner).
          - Uses callbacks TerminateOnNaN and EarlyStopping during search.
        Side effects:
          - Sets self.directory_name and self.project_name (used later by fit_model).
          - Uses setSeed for reproducibility.

    fit_model(seed=0, count=0, **kwargs)
        Build and train a final model using the best hyperparameters found by the tuner.
        Required kwargs same as fit_tuner (x, y, t, edges, weights).
        Behavior:
          - Reloads the tuner from self.params['tuner'] with directory and project name set earlier.
          - Fetches best hyperparameters; if params['defaults'] is True, overrides tuner values with
            provided defaults in self.params.
          - Builds the model from the hypermodel and trains it with ReduceLROnPlateau and EarlyStopping.
          - Prints a summary and records a concise hyperparameter summary in self.sparams (for the first run).
        Returns
          - Trained Keras model instance.
        Side effects:
          - Updates self.params with chosen hyperparameter values when defaults=False.

    load_graph(path)
        Load a graph description from disk. The format depends on params['json']:
          - If True: expects a JSON file and returns the parsed dict.
          - If False: reads a CSV into a pandas.DataFrame (header=None).
        Returns parsed graph object (dict or DataFrame depending on format).

    get_graph_info(graph)
        Convert a loaded graph object into a standardized dict with keys:
          - 'edges': numpy array of shape (num_edges, 2) listing (from, to) pairs
          - 'edge_weights': array-like of shape (num_edges, 1), normalized
          - 'influence_y': array-like mapping edges to influenced node(s) for outcome computation
        Behavior differs by storage format:
          - JSON: reads 'from', 'to', 'weights', and 'influence_y' fields and normalizes weights.
          - CSV/array: interprets rows as edge pairs and constructs default weights of ones.
        Returns graph_info dict.

    load_graphs(x_train, count)
        Build a graph_info structure for the current dataset and example index.
        Behavior:
          - Determines a path based on dataset_name (special-case 'sum').
          - Chooses file extension based on params['json'] and loads using load_graph.
          - If params['eye'] is True, creates an identity adjacency and sets influence_y to range(num_nodes).
        Returns graph_info (dict as described in get_graph_info).

    evaluate(x_test, model) -> np.ndarray
        Static helper that returns model.predict(x_test). Intended to be used with trained Keras models.

    prepare_data(**kwargs) -> (x_train, x_test, args_train, data_train, data_test)
        Load and prepare data for training and evaluation.
        Expected behavior:
          - Calls load_data (inherited) to obtain data_train and data_test dicts with keys 'x', 't', 'y'/'ys', etc.
          - Expands input x arrays to (num_samples, num_nodes, 1).
          - Loads graph_info via load_graphs and updates params['num_edges'] and params['influence_y'].
          - Selects correct y_train depending on params['binary'].
        Returns:
          - x_train (expanded), x_test (expanded), args_train dict suitable for fit_tuner/fit_model,
            data_train, data_test.

    train_and_evaluate(metric_list_train, metric_list_test, average_metric_list_train, average_metric_list_test, **kwargs)
        End-to-end orchestration for a single experiment run:
          - Sets a seed for reproducibility and obtains tracker contexts for training/testing profiling.
          - Prepares data via prepare_data and executes hyperparameter search and model training.
          - Performs predictions on train/test sets and computes evaluation metrics:
              - For dataset_name == 'jobs': computes policy risk and ATT using find_policy_risk.
              - Otherwise: computes PEHE and ATE using find_pehe.
          - Appends metrics to the provided lists (metric_list_* and average_metric_list_*).
        Required kwargs (optional/used):
          - count: integer index used when loading graph files and printing progress
          - folder_ind: optional folder identifier printed for some datasets
        Side effects:
          - Appends emission/tracker information to self.emission_train/self.emission_test.
          - Prints progress and metric summaries.

    Behavioral and implementation notes
    -----------------------------------
    - The class expects an external HyperGNNTarnet hypermodel and a define_tuner(...) method to be available
      (likely provided by the parent class or module) which return a configured Keras Tuner instance.
    - Uses setSeed(seed) at multiple points to enable reproducible runs.
    - Relies on inherited utilities: load_data, get_trackers, find_pehe, find_policy_risk, and possibly others
      provided by the CausalModel base class.
    - The tuning/training targets are constructed as concatenation of y and t (yt = concat([y, t])),
      so model outputs are expected to produce both potential outcome predictions (y0,y1) and/or
      treatment-related outputs depending on the architecture.
    - Many methods mutate self.params and other instance attributes; callers should be aware that the
      params dict acts as both configuration input and runtime state.

    Example (conceptual)
    --------------------
    >>> # After creating an instance with appropriate params:
    >>> g = GNNTARnetHyper(params)
    >>> x_train, x_test, args_train, data_train, data_test = g.prepare_data(count=0)
    >>> g.fit_tuner(**args_train)
    >>> model = g.fit_model(**args_train, count=0)
    >>> preds = g.evaluate(x_test, model)

    Exceptions
    ----------
    - FileNotFoundError / parsing errors can be raised by load_graph depending on storage and path conventions.
    - Tuner-related errors may occur if the tuner or HyperGNNTarnet are misconfigured.

    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.directory_name = None
        self.project_name = None

    def fit_tuner(self, seed=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        edges = kwargs['edges']
        weights = kwargs['weights']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)

        directory_name = 'params_' + self.params['tuner_name'] + '/' + self.params['dataset_name']
        setSeed(seed)

        project_name = self.params["model_name"]

        self.params['edges'] = edges
        self.params['weights'] = weights
        self.params['num_edges'] = edges.shape[0]
        self.params['num_nodes'] = x.shape[1]

        hp = kt.HyperParameters()

        self.directory_name = directory_name
        self.project_name = project_name

        hypermodel = HyperGNNTarnet(params=self.params, name='gnn_tarnet_search')
        objective = kt.Objective("val_regression_loss", direction="min")
        tuner = self.define_tuner(hypermodel, hp, objective, directory_name, project_name)

        stop_early = [TerminateOnNaN(), EarlyStopping(monitor='regression_loss', patience=5)]
        tuner.search(x, yt, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=self.params['verbose'])

        return

    def fit_model(self, seed=0, count=0, **kwargs):
        x = kwargs['x']
        y = kwargs['y']
        t = kwargs['t']
        edges = kwargs['edges']
        weights = kwargs['weights']
        t = tf.cast(t, dtype=tf.float32)
        yt = tf.concat([y, t], axis=1)
        setSeed(seed)

        self.params['edges'] = edges
        self.params['weights'] = weights
        self.params['num_edges'] = edges.shape[0]

        tuner = self.params['tuner'](
            HyperGNNTarnet(params=self.params),
            directory=self.directory_name,
            project_name=self.project_name,
            seed=0)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if self.params['defaults']:
            best_hps.values = {'gnn_n_fc': self.params['gnn_n_fc'],
                                'gnn_hidden_units': self.params['gnn_hidden_units'],
                                'n_hidden_0': self.params['n_hidden_0'],
                               'hidden_y0': self.params['hidden_y0'],
                               'n_hidden_1': self.params['n_hidden_1'],
                               'hidden_y1': self.params['hidden_y1']}
        else:
            self.params['gnn_n_fc'] = best_hps.get('gnn_n_fc')
            self.params['gnn_hidden_units'] = best_hps.get('gnn_hidden_units')
            self.params['n_hidden_0'] = best_hps.get('n_hidden_0')
            self.params['hidden_y0'] = best_hps.get('hidden_y0')
            self.params['n_hidden_1'] = best_hps.get('n_hidden_1')
            self.params['hidden_y1'] = best_hps.get('hidden_y1')

        model = tuner.hypermodel.build(best_hps)
        stop_early = [
            ReduceLROnPlateau(monitor='regression_loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=0., cooldown=0, min_lr=1e-8),
            EarlyStopping(monitor='regression_loss', patience=40, restore_best_weights=True)]

        model.fit(x=x, y=yt,
                  validation_split=0.0,
                  callbacks=stop_early,
                  epochs=self.params['epochs'],
                  verbose=self.params['verbose'],
                  batch_size=self.params['batch_size'])

        if count == 0:
            print(model.summary())
            self.sparams = f""" gnn_n_fc = {best_hps.get('gnn_n_fc')} gnn_hidden_units = {best_hps.get('gnn_hidden_units')}
             n_hidden_0 = {best_hps.get('n_hidden_0')} n_hidden_1 = {best_hps.get('n_hidden_1')}
             hidden_y0 = {best_hps.get('hidden_y0')}  hidden_y1 = {best_hps.get('hidden_y1')}"""
            print(f"""The hyperparameter search is complete. the optimal hyperparameters are
                              {self.sparams}""")
        return model

    def load_graph(self, path):
        if self.params['json']:
            with open(path) as f:
                graph = json.load(f)
        else:
            graph = pd.read_csv(path, header=None)
        return graph

    def get_graph_info(self, graph):
        if self.params['json']:
            edges = np.concatenate([np.asarray(graph['from']).reshape(-1, 1),
                                    np.asarray(graph['to']).reshape(-1, 1)], axis=1)
            influence_y = np.asarray(graph['influence_y'])
            edge_weights = np.asarray(graph['weights'])
            edge_weights = np.expand_dims(edge_weights / np.sum(edge_weights, axis=0), axis=-1)
            graph_info = {'edges': edges, 'edge_weights': edge_weights, 'influence_y': influence_y}
            return graph_info
        else:
            edges = np.asarray(graph)
            influence_y = []
            """Get non-zero elements from acyclic_W for edges and stack them to match the num of patients.
            Create an edges array (sparse adjacency matrix) of shape [num_samples, 2, num_edges]."""

            """Create an edge weights array of ones."""
            edge_weights = np.ones(shape=(edges.shape[0]))
            edge_weights = np.expand_dims(edge_weights / np.sum(edge_weights, axis=0), axis=-1)

            graph_info = {'edges': edges, 'edge_weights': edge_weights, 'influence_y': influence_y}
            return graph_info
    def load_graphs(self, x_train, count):

        if self.dataset_name == 'sum':
                path = 'graphs/sum_graph_' + str(self.params['num_layers'])
        else:
            path = 'graphs/' + self.params['dataset_name']

        if not self.params['json']:
            file_name = '/graph_' + str(count) + '.csv'
        else:
            file_name = '/graph_' + str(count) + '.json'

        graph = self.load_graph(path + file_name)
        graph_info = self.get_graph_info(graph)

        if self.params['eye']:
            acyclic_W = np.eye(x_train.shape[1])
            graph = np.asarray(np.nonzero(acyclic_W))
            edges = np.transpose(graph)
            graph_info['influence_y'] = np.arange(x_train.shape[1])
            graph_info['edges'] = edges

        return graph_info

    @staticmethod
    def evaluate(x_test, model):
        return model.predict(x_test)

    def prepare_data(self, **kwargs):
        data_train, data_test = self.load_data(**kwargs)
        x_train, t_train = data_train['x'], data_train['t']
        self.params['num_nodes'] = x_train.shape[1]
        self.folder_ind = kwargs.get('folder_ind')
        x_test = np.expand_dims(data_test['x'], axis=-1)
        x_train = np.expand_dims(x_train, axis=-1)

        graph_info = self.load_graphs(x_train=x_train, count=kwargs.get('count'))

        influence_y = graph_info['influence_y']
        self.params['num_edges'] = len(graph_info['edges'][1])
        self.params['influence_y'] = np.append(influence_y, np.expand_dims(x_train.shape[1], axis=0))

        edges = graph_info['edges']
        weights = graph_info['edge_weights']
        if self.params['binary']:
            y_train = data_train['y']
        else:
            y_train = data_train['ys']

        args_train = {'x': x_train, 't': t_train, 'y': y_train, 'edges': edges, 'weights': weights}

        return x_train, x_test, args_train, data_train, data_test

    def train_and_evaluate(self, metric_list_train, metric_list_test, average_metric_list_train,
                           average_metric_list_test, **kwargs):
        setSeed(seed=42)
        tracker_test, tracker_train = self.get_trackers(count=kwargs.get('count'))
        x_train, x_test, args_train, data_train, data_test = self.prepare_data(**kwargs)

        with tracker_train:
            self.fit_tuner(seed=0, **args_train)
            model = self.fit_model(**args_train, count=kwargs.get('count'))
        self.emission_train.append(tracker_train.final_emissions)

        # make a prediction
        with tracker_test:
            concat_pred_test = self.evaluate(x_test, model)
            concat_pred_train = self.evaluate(x_train, model)
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

