import tensorflow as tf
from tensorflow.python.ops.rnn_cell import linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import floor


class IterativeLayer(tf.nn.rnn_cell.RNNCell):
    def __init__(self, internal_nn, max_iterations=20, iterate_prob=0.75):
        if internal_nn is None:
            raise "You must define an internal NN to iterate"
        if internal_nn.input_size!=internal_nn.output_size:
            raise "Input and output sizes of the internal NN must be the same in order to iterate over them"
        if max_iterations<1:
            raise "The maximum amount of iterations to perform mus be a natural value"
        if iterate_prob<=0 or iterate_prob>=1:
            raise "iteration_prob must be a value between 0 and 1"
        self._internal_nn = internal_nn
        self._max_iterations = max_iterations
        self._iterate_prob = tf.constant(iterate_prob)
        self._number_of_iterations_built = 0

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
        should_add_summary = tf.get_variable_scope().reuse is True and self._number_of_iterations_built is 0
        output, new_state, number_of_iterations_performed = self.resolve_iteration_calculation(input, state, tf.zeros([]), scope)
        if should_add_summary:
            tf.histogram_summary("iterations_performed", number_of_iterations_performed,
                                 name="iterations_performed_summary")
        return output, new_state

    def resolve_iteration_calculation(self, input, state, number_of_iterations_performed, scope):
        output, new_state = self._internal_nn(input, state, scope)
        number_of_iterations_performed += 1
        self._number_of_iterations_built += 1
        if self._number_of_iterations_built < self._max_iterations:
            return tf.cond(
                self.resolve_iteration_activation(input, state, output, new_state),
                           lambda: self.resolve_iteration_calculation(output,
                                                                      new_state,
                                                                      number_of_iterations_performed=
                                                                        number_of_iterations_performed,
                                                                      scope=scope),
                           lambda: [output, new_state, number_of_iterations_performed])
        return output, new_state, number_of_iterations_performed

    def resolve_iteration_activation(self, input, old_state, output, new_state):
        iteration_gate_logits = linear([output], 1, True,
                                       scope=tf.get_variable_scope())
        tf.get_variable_scope().reuse_variables()
        iteration_gate_activation = floor(tf.reduce_max(sigmoid(iteration_gate_logits)) + self._iterate_prob)
        self._iterate_prob *= self._iterate_prob
        return tf.equal(iteration_gate_activation, tf.constant(1.))
