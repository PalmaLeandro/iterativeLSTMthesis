import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import floor
from tensorflow.python.ops.rnn_cell import linear
from rnn_cell import *

class NarrowIterativeLSTM(RNNCell):
    def __init__(self, max_iterations=50.0, iterate_prob=0.5, iterate_prob_decay=0.5, num_units=1, forget_bias=0.0, input_size=None):
        self._iterate_prob = iterate_prob
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._input_size = num_units if input_size is None else input_size
        self._max_iterations = max_iterations
        self._iterate_prob_decay = iterate_prob_decay

    def __call__(self, inputs, state, scope=None):
        "Iterative Long short-term memory cell (LSTM)."
        with vs.variable_scope(scope or type(self).__name__):
            loop_vars = [inputs, state, tf.zeros([self.output_size]), tf.constant(self._forget_bias), tf.constant(0.0),
                         tf.constant(self._max_iterations), tf.constant(self._iterate_prob), tf.constant(self._iterate_prob_decay), tf.ones(inputs.get_shape()), tf.constant(True)]
            loop_vars[0], loop_vars[1], loop_vars[2], loop_vars[3], loop_vars[4], loop_vars[5], loop_vars[6], loop_vars[
                7], loop_vars[8], loop_vars[9] = tf.while_loop(iterativeLSTM_LoopCondition, iterativeLSTM_Iteration, loop_vars)

            # This basic approach doesn't apply any last layer to the unit.
        return loop_vars[0] , loop_vars[1], loop_vars[4]

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units * 2


def iterativeLSTM_CellCalculation(inputs, state, num_units, forget_bias, iteration_prob, iteration_activation):
    number_of_units = num_units.get_shape().dims[0].value

    new_output, new_state, p = iterativeLSTM(inputs, state, number_of_units, forget_bias, iteration_activation)

    new_iteration_activation = floor(tf.nn.sigmoid(p) + iteration_prob) * iteration_activation

    return new_output,  new_state, new_iteration_activation


def iterativeLSTM_Iteration(inputs, state, num_units, forget_bias, iteration_number, max_iterations,
                            iteration_prob, iteration_prob_decay, iteration_activation, keep_looping):

    new_output, new_state, new_iteration_activation = iterativeLSTM_CellCalculation(inputs, state, num_units,
                                                                                        forget_bias,
                                                                                        iteration_prob, iteration_activation)
    iteration_flag = tf.reduce_max(new_iteration_activation)

    new_iteration_number = iteration_number + iteration_flag

    do_keep_looping = tf.logical_and(tf.less(new_iteration_number, max_iterations), tf.equal(iteration_flag, tf.constant(1.0)))

    new_iteration_prob = iteration_prob * iteration_prob_decay

    # Here the current output is selected. If there will be another iteration, then the inputs remain. Otherwise, the last output will be used.
    new_output = tf.cond(do_keep_looping, lambda:  inputs, lambda: new_output)
    # Here the current state is selected. All i have to do in order to keep the iteration within the cell gates is to update c but not to update h if it's bit the last iteration.
    new_state = tf.cond(do_keep_looping, lambda:  array_ops.concat(1, [array_ops.split(1, 2, new_state)[0], array_ops.split(1, 2, state)[1]]), lambda: new_state)

    return new_output, new_state, num_units, forget_bias, new_iteration_number, max_iterations, new_iteration_prob, iteration_prob_decay, new_iteration_activation, do_keep_looping


def iterativeLSTM_LoopCondition(inputs, state, num_units, forget_bias, iteration_number, max_iterations,
                                iteration_prob, iteration_prob_decay, iteration_activation, keep_looping):
    return keep_looping


def iterativeLSTM(inputs, state, num_units, forget_bias, iteration_activation):
    # This function aplies the standard LSTM calculation plus the calculation of the evidence to infer if another iteration is needed.

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

    # In this approach the evidence of the iteration gate is based on the inputs that doesn't change over iterations and its state
    p = linear([ inputs, new_state], num_units, True,scope= "iteration_activation")
    return new_h, new_state,p
