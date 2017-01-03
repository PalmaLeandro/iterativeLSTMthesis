import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell import linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import floor


class IterativeCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, internal_nn, max_iterations=10, iterate_prob=0.5, iterate_prob_decay=0.5):
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
        self._iterate_prob = tf.Variable(iterate_prob,trainable=False)
        self._iterate_prob_decay = tf.Variable(iterate_prob_decay,trainable=False)
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
        self._iteration_activations = self.resolve_iteration_activations(input,state,input,state)
        output, new_state, number_of_iterations_performed = self.resolve_iteration_calculation(input, state, tf.zeros([]), scope)
        if should_add_summary:
            tf.histogram_summary("iterations_performed", number_of_iterations_performed,
                                 name="iterations_performed_summary")
        return output, new_state

    def resolve_iteration_calculation(self, input, state, number_of_iterations_performed, scope):
        self._number_of_iterations_built += 1

        old_c, old_h = array_ops.split(1, 2, state)
        output, new_state = self._internal_nn(input, state, scope)
        tf.get_variable_scope().reuse_variables()
        new_c, new_h = array_ops.split(1, 2, new_state)
        # Only a new state is exposed if the iteration gate in this unit of this batch activated the extra iteration.
        new_h = new_h * self._iteration_activations + old_h * (1 - self._iteration_activations)
        new_c = new_c * self._iteration_activations + old_c * (1 - self._iteration_activations)
        output = new_h
        new_state_to_next_iteration = array_ops.concat(1, [new_c, old_h])
        new_state_to_output = array_ops.concat(1, [new_c, new_h])
        if self._number_of_iterations_built < self._max_iterations:
            iteration_activation_flag = floor(tf.reduce_max(self._iteration_activations) + self._iterate_prob)
            number_of_iterations_performed += iteration_activation_flag
            self._iteration_activations = self.resolve_iteration_activations(input, state, output, new_state_to_output)  * self._iteration_activations
            return tf.cond(tf.equal(iteration_activation_flag, tf.constant(1.)),
                           lambda: self.resolve_iteration_calculation(output,
                                                                      new_state_to_output,
                                                                      number_of_iterations_performed=
                                                                        number_of_iterations_performed,
                                                                      scope=scope),
                           lambda: [output, new_state_to_output, number_of_iterations_performed])
        return output, new_state_to_output, tf.constant(self._max_iterations)

    def resolve_iteration_activations(self, input, old_state, output, new_state):
        iteration_gate_logits = linear([input, old_state, output, new_state], self.output_size, True,
                                       scope=tf.get_variable_scope())
        iteration_activations = sigmoid(iteration_gate_logits)
        self._iterate_prob *= self._iterate_prob_decay
        return iteration_activations

