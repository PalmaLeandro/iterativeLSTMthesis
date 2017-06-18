
# coding: utf-8

# At this notebook i am going to expose the ability of an iterative approach to recognize complex patterns aginst conventional feedfoward approach

# In[1]:


import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import floor
from rnn_cell import *

class IterativeCell(object):

    def __init__(self, size, max_iterations=50., initial_iterate_prob=0.5,
                 iterate_prob_decay=0.5, allow_cell_reactivation=True, add_summaries=False, device_to_run_at=None):
        self._device_to_run_at = device_to_run_at
        self._sixe = size

        if max_iterations < 1:
            raise "The maximum amount of iterations to perform must be a natural value"

        if initial_iterate_prob <= 0 or initial_iterate_prob >= 1:
            raise "iteration_prob must be a value between 0 and 1"

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

    def __call__(self, input, scope=None):
        if self._should_add_summaries:
            self.add_pre_execution_summaries(input)

        with vs.variable_scope(scope or type(self).__name__):
            loop_vars = [input, tf.zeros([self.output_size]), tf.constant(0.0),
                         tf.constant(self._max_iteration_constant), tf.constant(self._initial_iterate_prob_constant),
                         tf.constant(self._iterate_prob_decay_constant), tf.ones(tf.shape(input)), tf.zeros([input.get_shape().dims[0].value, self.output_size]),
                         tf.constant(True)]
            loop_vars[0], loop_vars[1], loop_vars[2], loop_vars[3], loop_vars[4], loop_vars[5], loop_vars[6], loop_vars[
                7], loop_vars[8], loop_vars[9], loop_vars[10] = tf.while_loop(iterativeLSTM_LoopCondition, iterativeLSTM_Iteration, loop_vars)

        #_, loop_vars[0] = array_ops.split(1, 2, loop_vars[1])

        if self._should_add_summaries:
            self.add_post_execution_summaries(input, state, loop_vars[0] , loop_vars[1], loop_vars[4], None, loop_vars[9], None)

        return loop_vars[0]

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


def iterativeLSTM_Iteration(inputs, num_units, iteration_number, max_iterations,
                            iteration_prob, iteration_prob_decay, iteration_activation, iteration_count, 
                            keep_looping):

    output, new_iteration_activation = iterativeLSTM(inputs, state, num_units.get_shape().dims[0].value,
                                                                                        forget_bias, iteration_activation,
                                                                                        iteration_count, iteration_prob)
    iteration_flag = tf.reduce_max(new_iteration_activation)

    iteration_count = iteration_count + iteration_flag

    new_iteration_number = iteration_number + iteration_flag

    do_keep_looping = tf.logical_and(tf.less(new_iteration_number, max_iterations), tf.equal(iteration_flag, tf.constant(1.0)))

    new_iteration_prob = iteration_prob * iteration_prob_decay

    #new_c, new_h = array_ops.split(1, 2, new_state)

    new_output = tf.cond(do_keep_looping, lambda: inputs, lambda: output)

    return output, new_state, num_units, forget_bias, new_iteration_number, max_iterations, new_iteration_prob, iteration_prob_decay, new_iteration_activation, iteration_count, do_keep_looping

def iterativeLSTM_LoopCondition(inputs, num_units, iteration_number, max_iterations,
                                iteration_prob, iteration_prob_decay, iteration_activation, iteration_count, 
                                keep_looping):
    return keep_looping


def iterative_cell(inputs, num_units, iteration_activation, iteration_count, iteration_prob):
    j_logits = linear([inputs], num_units, False, scope="j_logits")
    j_displacement = linear([iteration_count], num_units, True, scope="j_displacement")
    opposed_j = tanh( - j_logits + j_displacement)
    new_info = tanh(tanh(j_logits + j_displacement) + tanh(opposed_j * sigmoid(opposed_j)))
    logits = linear([new_info], num_units, True)

    new_output = logits * iteration_activation + inputs * (1 - iteration_activation)

    # In this approach the evidence of the iteration gate is based on the inputs that doesn't change over iterations and its state
    p = linear([inputs, new_output], num_units, True, scope= "iteration_activation")


    new_iteration_activation = update_iteration_activations(iteration_activation, floor(sigmoid(p) + iteration_prob))

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



# In[ ]:



import tensorflow as tf
import pandas as pd
import numpy as np
import math


flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate. Default will be 0.01 .')
flags.DEFINE_integer('max_epochs', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 4, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 5, 'Number of units in hidden layer 2.')
flags.DEFINE_string('train_dir', '/Users/leandro/Documents/Repositories/FIUBA/determination/data', 
                    'Directory to put the training data.')
flags.DEFINE_string('log_dir', '/Users/leandro/Documents/Repositories/FIUBA/determination/logs', 
                    'Directory to put the log.')

flags.DEFINE_boolean(
    "erase_logs_dir", True,
    "An option to erase summaries in the logdir.")

FIRST_LAYER_HIDDEN_UNITS = flags.FLAGS.hidden1
SECONG_LAYER_HIDDEN_UNITS = flags.FLAGS.hidden2
LEARNING_RATE = flags.FLAGS.learning_rate
MAX_EPOCHS = flags.FLAGS.max_epochs
DATA_DIR = flags.FLAGS.train_dir
LOG_DIR = flags.FLAGS.log_dir
NUM_CLASSES = 2
ERASE_LOGS_DIR = flags.FLAGS.erase_logs_dir


def init_dir(dir_path):
  if tf.gfile.Exists(dir_path) and ERASE_LOGS_DIR is True:
    tf.gfile.DeleteRecursively(dir_path)
  tf.gfile.MakeDirs(dir_path)

init_dir(LOG_DIR)


# In[2]:


df = pd.DataFrame({'X1':[0.0, 0.0, 1.0, 1.0], 'X2':[0.0, 1.0, 0.0, 1.0], 'X3':[0.0, 1.0, 1.0, 1.0], 'Y':[0.0, 1.0, 1.0, 2.0]})


# In[3]:


df.to_csv(DATA_DIR + '/input.csv', index=None)


# In[4]:


data = pd.read_csv(DATA_DIR + '/input.csv')
NUM_CLASSES = len(df.Y.unique())
BATCH_SIZE = len(data)


# In[5]:


print 'Number of values to predict: {}'.format(NUM_CLASSES)
print 'Number of examples to learn simultaneously: {}'.format(len(data))


# In[6]:


inputs_data = data.ix[:, 0:len(data.columns)-1].as_matrix()
if inputs_data is None:
   inputs_data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


# In[7]:


label_data = np.squeeze(data.ix[:, len(data.columns) - 1:len(data.columns)].as_matrix())
if label_data is None:
   label_data = [0.0, 1.0, 1.0, 0.0]


# In[8]:


def dictionary_from_proto_buffer(proto_buffer):
    summaries = {}
    for val in proto_buffer.value:
        # Assuming all summaries are scalars.
        summaries[val.tag] = val.simple_value
    return summaries

def summarize_layer_determinations(layer_output):
    number_of_components = layer_output.get_shape().dims[1].value
    det_coefs = []
    for i in range(number_of_components):  
        unit_det_coeffs = []
        for j in range(number_of_components):
            corr_coef, update_op = tf.contrib.metrics.streaming_pearson_correlation(layer_output[:, i], 
                                                                                    layer_output[:, j])
            det_coef = update_op**2
            tf.summary.scalar('determination_' + str(i) + '_and_' + str(j), det_coef)
            if j != i:
                unit_det_coeffs.append(det_coef)
        total_determination_of_unit = tf.reduce_sum(unit_det_coeffs)
        det_coefs.append(total_determination_of_unit)
        tf.summary.scalar('determination_of_' + str(i) + '_unit', total_determination_of_unit)
    

def build_model_layer(inputs, number_of_units, scope_name, use_relu, add_summaries=False):
    with tf.name_scope(scope_name):
        weights = tf.Variable(tf.truncated_normal([inputs.get_shape().dims[1].value, number_of_units], 
                                                 stddev=1.0 / math.sqrt(float(number_of_units))), name='weights')
        
        biases = tf.Variable(tf.zeros([number_of_units]), name='biases')
        
        logits = tf.matmul(inputs, weights) + biases
        
        layer_output = tf.nn.relu(logits) if use_relu is True else logits

        if add_summaries is True:
            summarize_layer_determinations(layer_output)

    return layer_output
        

def build_model(inputs):
    first_layer_output = build_model_layer(inputs=inputs, 
                                          number_of_units=FIRST_LAYER_HIDDEN_UNITS,
                                          scope_name='first_hidden_layer',
                                          use_relu=True,
                                          add_summaries=True)
    
    second_layer_output = build_model_layer(inputs=first_layer_output, 
                                            number_of_units=SECONG_LAYER_HIDDEN_UNITS,
                                            scope_name='second_hidden_layer',
                                            use_relu=True,
                                            add_summaries=True)
    
    output = build_model_layer(inputs=second_layer_output, 
                               number_of_units=NUM_CLASSES,
                               scope_name='logits',
                               use_relu=False,
                               add_summaries=False)
    return output


# In[9]:


with tf.Session() as sess:
    inputs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])
    label = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE])
    
    model_output = build_model(inputs)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(model_output, tf.to_int64(label)), 
                          name='xentropy_mean')
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    training = optimizer.minimize(loss, global_step=global_step)
    
    merge_summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(MAX_EPOCHS):
        model_feed = {inputs:inputs_data, label:label_data}

        _, loss_value, summaries = sess.run([training, loss, merge_summaries], feed_dict=model_feed)
        if epoch // 100 == 0:
            summary_writer.add_summary(summaries, global_step=epoch)
            summary_writer.flush()
            print 'epoch {}, loss = {}'.format(epoch, loss_value)
    
    summary_writer.close()
    sess.close()


# In[10]:


dictionary = dictionary_from_proto_buffer(tf.Summary.FromString(summaries))


# In[11]:


def separate_layer_units_from_dictionary(dictionary):
    layers = {}
    for annotation in dictionary.keys():
        if 'unit' in annotation:
            layer_name, unit_name = annotation.rsplit('/', 1)
            if layer_name in layers:
                layers[layer_name].append(dict([('name', unit_name), ('value', dictionary[annotation])]))
            else:
                layers[layer_name] = [dict([('name', unit_name), ('value', dictionary[annotation])])]
    return layers


# In[12]:


layers_units_annotations = separate_layer_units_from_dictionary(dictionary)
layers_units_annotations


# In[13]:


def min_and_max_units_determinations_from_layer(layer_units_annotations):
    max_determination_unit = {'name':'', 'value':0.0}
    min_determination_unit = {'name':'', 'value':1.0}
    for unit_annotation in layer_units_annotations:
        if 'determination' in unit_annotation['name']:
            if max_determination_unit['value'] < unit_annotation['value']:
                max_determination_unit = unit_annotation
            if min_determination_unit['value'] > unit_annotation['value']:
                min_determination_unit = unit_annotation
    return max_determination_unit, min_determination_unit


# In[14]:


def layers_candidates_to_substraction_from_nn_dictionary(dictionary):
    layers_candidates_to_substraction = {}
    layer_candidate_to_reduction = {'name': None, 'value':0., 'cell':None}
    for layer in dictionary:
        max_determination_unit, min_determination_unit = min_and_max_units_determinations_from_layer(dictionary[layer])
        layers_candidates_to_substraction[layer] = max_determination_unit
        if layer_candidate_to_reduction['value'] < max_determination_unit['value']/len(dictionary[layer]):
           layer_candidate_to_reduction = {'name':layer, 
                                           'value':max_determination_unit['value'], 
                                           'cell':max_determination_unit} 
    return layers_candidates_to_substraction, layer_candidate_to_reduction


# In[15]:


layers_candidates_to_substraction, layer_candidate_to_reduction = layers_candidates_to_substraction_from_nn_dictionary(layers_units_annotations)


# #### Cell of first layer with lowest Shapley value

# In[16]:


layers_candidates_to_substraction['first_hidden_layer']['name']


# #### Cell of second layer with lowest Shapley value

# In[17]:


layers_candidates_to_substraction['second_hidden_layer']['name']


# #### Layer with cell with lowest Shapley value

# In[18]:


layer_candidate_to_reduction


# In[19]:


from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


# In[20]:


show_graph(tf.get_default_graph().as_graph_def())

