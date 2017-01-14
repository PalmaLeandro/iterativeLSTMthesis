from pexpect.expect import searcher_re

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell import linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import floor
from tensorflow.python.ops import variable_scope as vs


class IterativeCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, internal_nn, max_iterations=10., iterate_prob=0.5, iterate_prob_decay=0.5, allow_cell_reactivation=True, add_summaries=False):
        if internal_nn is None:
            raise "You must define an internal NN to iterate"
        if internal_nn.input_size!=internal_nn.output_size:
            raise "Input and output sizes of the internal NN must be the same in order to iterate over them"
        if max_iterations<1:
            raise "The maximum amount of iterations to perform mus be a natural value"
        if iterate_prob<=0 or iterate_prob>=1:
            raise "iteration_prob must be a value between 0 and 1"
        self._internal_nn = internal_nn
        self._max_iteration_constant = max_iterations
        self._initial_iterate_prob_constant = iterate_prob
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
        loop_variables = [input,                                                    # previous layer output
                          state,                                                    # last cell's state
                          tf.zeros([input.get_shape()[0],input.get_shape()[1]]),    # number of the current iteration
                          tf.constant(self._initial_iterate_prob_constant),         # initial iteration probability
                          tf.ones([input.get_shape()[0],input.get_shape()[1]])]     # calculation of the first iteration's activation
        final_output, \
        final_state, \
        number_of_iterations_performed, \
        final_iterate_prob, \
        final_iteration_activations = tf.while_loop(self.loop_condition(), self.loop_body(), loop_variables)

        if self._should_add_summaries:
            self.add_summaries(final_output, final_state, number_of_iterations_performed, final_iterate_prob, final_iteration_activations)

        return final_output, final_state

    def add_summaries(self,final_output, final_state, number_of_iterations_performed, final_iterate_prob, final_iteration_activations):
        if not self._already_added_summaries.__contains__(tf.get_variable_scope().name+"/iterations_performed"):
            variable_summaries(number_of_iterations_performed, tf.get_variable_scope().name+"/iterations_performed")
            self._already_added_summaries.append(tf.get_variable_scope().name+"/iterations_performed")

    def loop_condition(self):
        return lambda input, state, iteration_number, iterate_prob, iteration_activations: \
            tf.logical_and(tf.equal(tf.reduce_max(iteration_activations), tf.constant(1.)),
                              tf.less(tf.reduce_max(iteration_number), tf.constant(self._max_iteration_constant)))

    def loop_body(self):
        return lambda input, state, iteration_number, iterate_prob, iteration_activations: \
                self.resolve_iteration_calculation(input, state, iteration_number, iterate_prob, iteration_activations)


    def resolve_iteration_calculation(self, input, state, number_of_iterations_performed, iterate_prob, current_iteration_activations):
        output, new_state = self._internal_nn(input, state)

        # Only a new state is exposed if the iteration gate in this unit of this batch activated the extra iteration.
        output = output * current_iteration_activations + input * (1 - current_iteration_activations)

        new_state_activation_extended = tf.tile(current_iteration_activations,[1,2])
        new_state = new_state * new_state_activation_extended + state * (1 - new_state_activation_extended)

        number_of_iterations_performed += current_iteration_activations
        iteration_activations = self.resolve_iteration_activations(input, state, output, new_state, iterate_prob, current_iteration_activations)
        output = tf.cond(self.loop_condition()(output, new_state, number_of_iterations_performed, iterate_prob, iteration_activations), lambda: input, lambda: output)
        new_iterate_prob = tf.pow(iterate_prob, 2)
        return output, new_state, number_of_iterations_performed, new_iterate_prob, iteration_activations

    def resolve_iteration_activations(self, input, old_state, output, new_state, iterate_prob, current_iteration_activations):
        iteration_gate_logits = linear([output], self.output_size, True, scope=tf.get_variable_scope())
        tf.get_variable_scope().reuse_variables()

        #iteration_activations = sigmoid(iteration_gate_logits)
        iteration_activations = tf.floor(sigmoid(iteration_gate_logits) + iterate_prob)
        return self.update_iteration_activations(current_iteration_activations, iteration_activations, iterate_prob)

    def update_iteration_activations(self, current_iteration_activations, new_iteration_activations, iterate_prob):
        # It is possible that other instances of the batch activate this cell, hence we need to avoid this by activate only those activations were this instance of the batch is actually activated
        batch_iteration_activations = tf.reduce_max(current_iteration_activations, 1, True)
        batch_iteration_activations_extended = tf.tile(batch_iteration_activations,[1, self.output_size])

        if self._allow_reactivation:
            return new_iteration_activations * batch_iteration_activations_extended
        else:
            return new_iteration_activations * current_iteration_activations * batch_iteration_activations_extended


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)