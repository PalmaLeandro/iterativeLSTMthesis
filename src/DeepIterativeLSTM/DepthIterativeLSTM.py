import tensorflow as tf
from numpy.lib.function_base import iterable
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import floor
from tensorflow.python.ops.rnn_cell import linear


class DepthIterativeLSTM(tf.nn.rnn_cell.RNNCell):
    def __init__(self, max_iterations=50.0, iterate_prob=0.5, num_units=1, num_layers=1,forget_bias=0.0, input_size=None, output_keep_prob=1.):
        self._iterate_prob = iterate_prob
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._input_size = num_units if input_size is None else input_size
        self._max_iterations = max_iterations
        self._num_layers = num_layers
        self._output_keep_prob = output_keep_prob

    def __call__(self, inputs, state, scope=None):
        "Iterative Long short-term memory cell (LSTM)."
        with vs.variable_scope(scope or type(self).__name__):
            loop_vars = [inputs, state, tf.zeros([self.output_size]), tf.zeros([self._num_layers]), tf.constant(self._forget_bias), tf.constant(0.0),
                         tf.constant(self._max_iterations), tf.constant(self._iterate_prob), tf.ones(inputs.get_shape()), tf.constant(self._output_keep_prob), tf.constant(True)]
            loop_vars[0], loop_vars[1], loop_vars[2], loop_vars[3], loop_vars[4], loop_vars[5], loop_vars[6], loop_vars[
                7], loop_vars[8], loop_vars[9], loop_vars[10] = tf.while_loop(iterativeLSTM_LoopCondition, iterativeLSTM_Iteration, loop_vars)
            #hasta aca la implementacion en run. de aca en mas la implementacion en debugger.

        return loop_vars[0], loop_vars[1], loop_vars[5]

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_layers * self._num_units * 2


def iterativeLSTM_CellCalculation(inputs, state, num_units, num_layers, forget_bias, iteration_prob, iteration_activation, keep_prob):
    # This function calculates a single iteration over the entire network.

    number_of_units = num_units.get_shape().dims[0].value
    number_of_layers = num_layers.get_shape().dims[0].value
    cur_state_pos = 0
    cur_inp = inputs
    new_states = []
    cs =[]
    new_cs =[]
    for i in range(number_of_layers):
        with vs.variable_scope("Cell%d" % i):
          cur_state = array_ops.slice(state, [0, cur_state_pos], [-1, number_of_units * 2])
          c, h = array_ops.split(1, 2, cur_state)
          cs.append(c)
          cur_state_pos += number_of_units * 2
          cur_inp, new_state = LSTM(cur_inp, cur_state, number_of_units, forget_bias, vs.get_variable_scope())
          new_c, new_h = array_ops.split(1, 2, new_state)

          # Only a new state is exposed if the iteration gate in this unit of this batch activated an extra iteration.
          new_c = new_c * iteration_activation + c * (1 - iteration_activation)
          new_h = (new_h) * iteration_activation + h * (1 - iteration_activation)
          new_h = tf.nn.dropout(new_h, keep_prob)

          # Here the new state is the entirely update.
          new_state = array_ops.concat(1, [new_c, new_h])
          new_states.append(new_state)
          new_cs.append(new_c)
          cur_inp = new_h
    new_output = cur_inp
    new_state = array_ops.concat(1, new_states)
    new_c = array_ops.concat(1, new_cs)
    c = array_ops.concat(1, cs)

    p = linear([ inputs, new_state], number_of_units, True, scope="iteration_activation")

    new_iteration_activation = floor(tf.nn.sigmoid(p) + iteration_prob) * iteration_activation

    return new_output,  new_state, new_iteration_activation


def iterativeLSTM_Iteration(inputs, state, num_units, num_layers, forget_bias, iteration_number, max_iterations,
                            iteration_prob, iteration_activation, keep_prob, keep_looping):

    new_output, new_state, new_iteration_activation = iterativeLSTM_CellCalculation(inputs, state, num_units, num_layers,
                                                                                        forget_bias,
                                                                                        iteration_prob, iteration_activation, keep_prob)
    iteration_flag = tf.reduce_max(new_iteration_activation)

    new_iteration_number = iteration_number + iteration_flag

    do_keep_looping = tf.logical_and(tf.less(new_iteration_number, max_iterations), tf.equal(iteration_flag, tf.constant(1.0)))

    new_iteration_prob = iteration_prob * iteration_prob


    # Here the current output is selected. If there will be another iteration, then the residual calculatios is used. Otherwise, the last output will be used.
    new_output = tf.cond(do_keep_looping, lambda:  inputs + new_output, lambda: new_output)

    return new_output, new_state, num_units, num_layers, forget_bias, new_iteration_number, max_iterations, new_iteration_prob, new_iteration_activation, keep_prob, do_keep_looping


def iterativeLSTM_LoopCondition(inputs, state, num_units, num_layers, forget_bias, iteration_number, max_iterations,
                                iteration_prob, iteration_activation, keep_prob, keep_looping):
    return keep_looping


def LSTM(inputs, state, num_units, forget_bias, scope):
    # This function aplies the standard LSTM calculation.

    # "BasicLSTM"
    # Parameters of gates are concatenated into one multiply for efficiency.
    c, h = array_ops.split(1, 2, state)
    concat = linear([inputs, h], 4 * num_units, bias=True, scope=scope)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(1, 4, concat)

    new_c = c * sigmoid(f + forget_bias) + sigmoid(i) * tanh(j)
    new_h = tanh(new_c) * sigmoid(o)
    new_state = array_ops.concat(1, [new_c, new_h])

    return new_h, new_state
