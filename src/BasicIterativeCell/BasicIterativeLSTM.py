import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import floor
from tensorflow.python.ops.rnn_cell import linear
from rnn_cell import *

class IterativeCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, internal_nn, iteration_activation_nn=None, max_iterations=5., initial_iterate_prob=0.5,
                 iterate_prob_decay=0.75, allow_cell_reactivation=True, add_summaries=False, device_to_run_at=None):
        self._device_to_run_at = device_to_run_at

        if internal_nn is None:
            raise "You must define an internal NN to iterate"

        if internal_nn.input_size != internal_nn.output_size:
            raise "Input and output sizes of the internal NN must be the same in order to iterate over them"

        if iteration_activation_nn is not None and (iteration_activation_nn.output_size != internal_nn.output_size):
            raise "Input and output sizes of the iteration activation NN should be the same as the ones from " \
                  "internal NN"

        if max_iterations < 1:
            raise "The maximum amount of iterations to perform must be a natural value"

        if initial_iterate_prob <= 0 or initial_iterate_prob >= 1:
            raise "iteration_prob must be a value between 0 and 1"

        self._internal_nn = internal_nn
        self._iteration_activation_nn = iteration_activation_nn
        self._max_iteration_constant = max_iterations
        self._initial_iterate_prob_constant = initial_iterate_prob
        self._iterate_prob_decay_constant = iterate_prob_decay
        self._allow_reactivation = allow_cell_reactivation
        self._should_add_summaries = add_summaries
        self._already_added_summaries = []

    @property
    def input_size(self):
        return self._internal_nn._input_size

    @property
    def output_size(self):
        return self._internal_nn.output_size

    @property
    def state_size(self):
        return self._internal_nn.state_size

    def __call__(self, input, state, scope=None):
        if self._should_add_summaries:
            self.add_pre_execution_summaries(input, state)

        with vs.variable_scope(scope or type(self).__name__):
            loop_vars = [input, state, tf.zeros([self.output_size]), tf.constant(0.0), tf.constant(0.0),
                         tf.constant(self._max_iteration_constant), tf.constant(self._initial_iterate_prob_constant),
                         tf.constant(self._iterate_prob_decay_constant), tf.ones(tf.shape(input)), tf.zeros([input.get_shape().dims[0].value, self.output_size]),
                         tf.constant(True)]
            loop_vars[0], loop_vars[1], loop_vars[2], loop_vars[3], loop_vars[4], loop_vars[5], loop_vars[6], loop_vars[
                7], loop_vars[8], loop_vars[9], loop_vars[10] = tf.while_loop(iterativeLSTM_LoopCondition, iterativeLSTM_Iteration, loop_vars)

        #_, loop_vars[0] = array_ops.split(1, 2, loop_vars[1])

        if self._should_add_summaries:
            self.add_post_execution_summaries(input, state, loop_vars[0] , loop_vars[1], loop_vars[4], None, loop_vars[9], None)

        return loop_vars[0], loop_vars[1]

    def add_pre_execution_summaries(self, input, state):
        if not self._already_added_summaries.__contains__(tf.get_variable_scope().name +
                                                                  "/pre_execution_input_entropy"):
            variable_summaries(calculate_feature_entropy(input),
                               tf.get_variable_scope().name + "/pre_execution_input_entropy", add_histogram=False)
            self._already_added_summaries.append(tf.get_variable_scope().name + "/pre_execution_input_entropy")

    def add_post_execution_summaries(self, initial_input, initial_state, final_output, final_state, number_of_iterations_performed,
                                     final_iterate_prob, final_iterations_counts, final_iteration_activations):
        if not self._already_added_summaries.__contains__(tf.get_variable_scope().name+"/iterations_performed"):
            variable_summaries(number_of_iterations_performed, tf.get_variable_scope().name+"/iterations_performed")
            self._already_added_summaries.append(tf.get_variable_scope().name+"/iterations_performed")

            self._already_added_summaries.append(tf.get_variable_scope().name+"/iterations_counts")
            variable_summaries(final_iterations_counts, tf.get_variable_scope().name+"/iterations_counts")

            variable_summaries(calculate_feature_entropy(final_output),
                               tf.get_variable_scope().name + "/post_execution_output_entropy", add_histogram=False)
            self._already_added_summaries.append(tf.get_variable_scope().name + "/post_execution_output_entropy")

            variable_summaries(calculate_feature_vectors_kl_divergence(initial_input, final_output),
                               tf.get_variable_scope().name + "/improved_from_former_kl_divergence", add_histogram=False)
            self._already_added_summaries.append(tf.get_variable_scope().name + "/improved_from_former_kl_divergence")

            variable_summaries(calculate_feature_vectors_kl_divergence(final_output, initial_input),
                               tf.get_variable_scope().name + "/former_from_improved_kl_divergence", add_histogram=False)
            self._already_added_summaries.append(tf.get_variable_scope().name + "/former_from_improved_kl_divergence")

def calculate_feature_entropy(feature_vector):
    return - feature_vector * tf.log(feature_vector) - (1 - feature_vector) * tf.log(1 - feature_vector)

def calculate_feature_vectors_kl_divergence(former_feature_vector, updated_feature_vector):
    return former_feature_vector * (tf.log(former_feature_vector) - tf.log(updated_feature_vector)) + (1 - former_feature_vector) * (tf.log(1 - former_feature_vector) - tf.log(1 - updated_feature_vector))


def iterativeLSTM_Iteration(inputs, state, num_units, forget_bias, iteration_number, max_iterations,
                            iteration_prob, iteration_prob_decay, iteration_activation, iteration_count, 
                            keep_looping):

    new_output, new_state, new_iteration_activation = iterativeLSTM(inputs, state, num_units.get_shape().dims[0].value,
                                                                                        forget_bias, iteration_activation,
                                                                                        iteration_count, iteration_prob)
    iteration_flag = tf.reduce_max(new_iteration_activation)

    iteration_count = iteration_count + iteration_flag

    new_iteration_number = iteration_number + iteration_flag

    do_keep_looping = tf.logical_and(tf.less(new_iteration_number, max_iterations), tf.equal(iteration_flag, tf.constant(1.0)))

    new_iteration_prob = iteration_prob * iteration_prob_decay

    new_c, new_h = array_ops.split(1, 2, new_state)

    c, h = array_ops.split(1, 2, state)
    new_c = tf.cond(do_keep_looping, lambda: c, lambda: new_c)
    #new_h = tf.cond(do_keep_looping, lambda: h, lambda: new_h)
    new_state = array_ops.concat(1, [new_c, new_h])

    new_output = tf.cond(do_keep_looping, lambda: inputs, lambda: new_output)

    return new_output, new_state, num_units, forget_bias, new_iteration_number, max_iterations, new_iteration_prob, iteration_prob_decay, new_iteration_activation, iteration_count, do_keep_looping

def iterativeLSTM_LoopCondition(inputs, state, num_units, forget_bias, iteration_number, max_iterations,
                                iteration_prob, iteration_prob_decay, iteration_activation, iteration_count, 
                                keep_looping):
    return keep_looping


def iterativeLSTM(inputs, state, num_units, forget_bias, iteration_activation, iteration_count, iteration_prob):
    # This function applies the standard LSTM calculation plus the calculation of the evidence to infer if another iteration is needed.

    # "BasicLSTM"
    # Parameters of gates are concatenated into one multiply for efficiency.
    c, h = array_ops.split(1, 2, state)
    concat = linear([inputs, h], 4 * num_units, True)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(1, 4, concat)

    new_c = c * sigmoid(f + forget_bias) + sigmoid(i) * tanh(j)
    new_h = tanh(new_c) * sigmoid(o)

    # Only a new state is exposed if the iteration gate in this unit of this batch activated the extra iteration.
    new_h = new_h * iteration_activation + h * (1 - iteration_activation)
    new_c = new_c * iteration_activation + c * (1 - iteration_activation)

    new_state = array_ops.concat(1, [new_c, new_h])
    new_output = (new_h+inputs) * iteration_activation + inputs * (1 - iteration_activation)

    # In this approach the evidence of the iteration gate is based on the inputs that doesn't change over iterations and its state
    #p = linear([j], num_units, True, scope= "iteration_activation")

    new_iteration_activation = update_iteration_activations(iteration_activation, tf.ones(tf.shape(inputs)))

    return new_output, new_state, new_iteration_activation

def update_iteration_activations(current_iteration_activations, new_iteration_activations):
    # It is possible that other instances of the batch activate this cell, hence we need to avoid this
    # by activate only those activations were this instance of the batch is actually activated
    batch_iteration_activations = tf.reduce_max(current_iteration_activations, 1, True)
    batch_iteration_activations_extended = tf.tile(batch_iteration_activations,
        [1, int(current_iteration_activations.get_shape().dims[1].value
            or new_iteration_activations.get_shape().dims[1].value)])

    return new_iteration_activations * batch_iteration_activations_extended


def variable_summaries(var, name, add_distribution=True, add_range=True, add_histogram=True):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        if add_distribution:
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
                tf.scalar_summary('sttdev/' + name, stddev)

        if add_range:
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))

        if add_histogram:
            tf.histogram_summary(name, var)
