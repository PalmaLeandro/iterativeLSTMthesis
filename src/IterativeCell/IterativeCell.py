import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell import linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import floor


class IterativeCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, internal_nn, max_iterations=10, iterate_prob=0.5, iterate_prob_decay=0.75, allow_cell_reactivation=True):
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
        self._allow_reactivation = allow_cell_reactivation
        self._number_of_iterations_built = 0
        self._iteration_activations = None

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
            tf.histogram_summary(tf.get_variable_scope().name+"/iterations_performed", number_of_iterations_performed,
                                 name="iterations_performed_summary")
        return output, new_state

    def resolve_iteration_calculation(self, input, state, number_of_iterations_performed, scope):
        self._number_of_iterations_built += 1
        output, new_state = self._internal_nn(input, state, scope)
        tf.get_variable_scope().reuse_variables()

        # Only a new state is exposed if the iteration gate in this unit of this batch activated the extra iteration.
        output = output * self._iteration_activations + input * (1 - self._iteration_activations)
        new_state_activation_extended = array_ops.concat(1, [self._iteration_activations for dim in range(0, self._internal_nn.output_size)]) #TODO: change to extend vector or concat as function of state size
        new_state = new_state * new_state_activation_extended + state * (1 - new_state_activation_extended)

        if self._number_of_iterations_built < self._max_iterations:

            iteration_activation_flag = floor(tf.reduce_max(self._iteration_activations) + self._iterate_prob)

            number_of_iterations_performed += iteration_activation_flag

            self.resolve_iteration_activations(input, state, output, new_state) * self._iteration_activations

            return tf.cond(tf.equal(iteration_activation_flag, tf.constant(1.)),
                           lambda: self.resolve_iteration_calculation(output,
                                                                      new_state,
                                                                      number_of_iterations_performed=
                                                                        number_of_iterations_performed,
                                                                      scope=scope),
                           lambda: [output, new_state, number_of_iterations_performed])
        return output, new_state, tf.constant(self._max_iterations,tf.float32)

    def resolve_iteration_activations(self, input, old_state, output, new_state):
        iteration_gate_logits = linear([input, output], self.output_size, True,
                                       scope=tf.get_variable_scope())
        iteration_activations = sigmoid(iteration_gate_logits)
        self._iterate_prob *= self._iterate_prob_decay
        self.update_iteration_activations(iteration_activations)
        return self._iteration_activations

    def update_iteration_activations(self, new_iteration_activations):
        # It is possible that other instances of the batch activate this cell, hence we need to avoid this by activate only those activations were this instance of the batch is actually activated
        batch_iteration_activations = tf.reduce_max(new_iteration_activations, 1, True)
        batch_iteration_activations_extended = array_ops.concat(1, [batch_iteration_activations for dim in range(0, self._internal_nn.output_size)]) #TODO: change to extend vector or concat as function of state size
        if self._iteration_activations is None or self._allow_reactivation is True:
            self._iteration_activations = new_iteration_activations * batch_iteration_activations_extended
        else:
            self._iteration_activations *= new_iteration_activations * batch_iteration_activations_extended
