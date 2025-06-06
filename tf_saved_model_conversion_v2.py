# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convert Tensorflow SavedModel to TensorFlow.js web format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import tempfile
from zipfile import ZipFile

# Required to load saved models that use TFDF.
# import tensorflow_decision_forests
import tensorflow as tf
# ... other imports above ...

# Attempt to import tensorflow_decision_forests, but don't fail if it's not found
# or if it has issues loading its own dependencies.
try:
    import tensorflow_decision_forests
except (ImportError, tf.errors.NotFoundError, AttributeError) as e: # Catch multiple potential errors
    tensorflow_decision_forests = None # Define it as None if import fails
    print(f"Warning: Optional module 'tensorflow_decision_forests' could not be imported or has issues. Error: {e}. Proceeding without it.")

# ... other code below ...
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.io import gfile
from tensorflow.python.checkpoint.trackable_view import TrackableView
from tensorflow.python.eager import context
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.keras.saving.saving_utils import trace_model_call
from tensorflow.python.keras.saving.saving_utils import def_function
from tensorflow.python.keras.saving.saving_utils import model_input_signature
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model import loader
from tensorflow.python.tools.saved_model_utils import get_meta_graph_def
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.saved_model.experimental import TrackableResource
from google.protobuf.json_format import MessageToDict
import tensorflow_hub as hub
from packaging import version

from tensorflowjs import write_weights
from tensorflowjs.converters import common
from tensorflowjs.converters import normalize_bias_add
from tensorflowjs.converters import fold_batch_norms
from tensorflowjs.converters import fuse_prelu
from tensorflowjs.converters import fuse_depthwise_conv2d
from tensorflowjs.converters import graph_rewrite_util
from tensorflowjs import resource_loader

CLEARED_TENSOR_FIELDS = (
    'tensor_content', 'half_val', 'float_val', 'double_val', 'int_val',
    'string_val', 'scomplex_val', 'int64_val', 'bool_val',
    'resource_handle_val', 'variant_val', 'uint32_val', 'uint64_val')

_HUB_V1_MODULE_PB = "tfhub_module.pb"

def load_graph(graph_filename):
  """Loads GraphDef. Returns Python Graph object.

  Args:
    graph_filename: string File name for the frozen graph.
  """
  with tf.compat.v1.gfile.Open(graph_filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    # Set name to empty to avoid using the default name 'import'.
    tf.import_graph_def(graph_def, name='')

  return graph

def get_cluster():
  """Grappler optimization configuration for GPU."""
  named_device = device_properties_pb2.NamedDevice()
  named_device.name = '/GPU:0'
  named_device.properties.type = 'GPU'
  named_device.properties.environment['architecture'] = '4'
  cluster = gcluster.Cluster(devices=[named_device])
  return cluster

def validate(graph_def, skip_op_check, strip_debug_ops):
  """Validate if the node's op is compatible with TensorFlow.js.

  Args:
    graph_def: tf.GraphDef TensorFlow GraphDef proto object, which represents
      the model topology.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to allow unsupported debug ops.
  """
  nodes = [] + list(graph_def.node)
  for func in graph_def.library.function:
    nodes.extend(list(func.node_def))

  if skip_op_check:
    return set()
  ops = []
  for filename in resource_loader.list_dir('op_list'):
    if os.path.splitext(filename)[1] == '.json':
      with resource_loader.open_file(os.path.join('op_list',
                                                  filename)) as json_data:
        ops += json.load(json_data)

  names = {x['tfOpName'] for x in ops}
  if strip_debug_ops:
    names = names.union({'Assert', 'CheckNumerics', 'Print'})
  not_supported = {x.op for x in [x for x in nodes if x.op not in names]}
  return not_supported

def _run_grappler(config, graph_def, graph, signature_def):
  meta_graph = export_meta_graph(
      graph_def=graph_def, graph=graph)

  meta_graph.signature_def["not_used_key"].CopyFrom(signature_def)

  return tf_optimizer.OptimizeGraph(
      config, meta_graph, cluster=get_cluster())

def optimize_graph(graph, signature_def,
                   skip_op_check=False, strip_debug_ops=False,
                   experiments=False):
  """Takes a Python Graph object and optimizes the graph.

  Args:
    graph: The frozen graph to optimize.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
  """

  # Add a collection 'train_op' so that Grappler knows the outputs.
  for _, output in signature_def.outputs.items():
    name = output.name.split(':')[0]
    graph.add_to_collection('train_op', graph.get_operation_by_name(name))

  graph_def = graph.as_graph_def()

  unsupported = validate(graph_def, skip_op_check,
                         strip_debug_ops)
  if unsupported:
    raise ValueError('Unsupported Ops in the model before optimization\n' +
                     ', '.join(unsupported))

  # first pass of grappler optimization, this is needed for batch norm folding.
  config = config_pb2.ConfigProto()
  rewriter_config = config.graph_options.rewrite_options
  rewriter_config.optimizers[:] = [
      'pruning', 'constfold', 'arithmetic', 'dependency', 'pruning',
      'constfold', 'arithmetic', 'dependency'
  ]
  if experiments:
    rewriter_config.experimental_disable_compressed_tensor_optimization = True

  if strip_debug_ops:
    rewriter_config.optimizers.insert(0, 'debug_stripper')

  optimized_graph = _run_grappler(config, graph_def, graph, signature_def)

  # batch norm folding
  optimized_graph = fold_batch_norms.fold_batch_norms(optimized_graph)

  optimized_graph = normalize_bias_add.normalize_bias_add_op(optimized_graph)

  # set the device to CPU for all Conv2d and MatMul nodes, since grappler
  # remap optimizer only support FusedConv2D and FusedMatMul for CPU.
  for node in optimized_graph.node:
    if node.op == 'Conv2D' or node.op == 'MatMul':
      node.device = '/device:CPU:0'

  # rerun grappler to fuse conv2d/matmul
  config.graph_options.rewrite_options.optimizers[:] = [
      'remap',
      'constfold', 'arithmetic', 'dependency'
  ]

  optimized_graph = _run_grappler(config, optimized_graph, graph, signature_def)

  optimized_graph = _remove_unused_control_flow_inputs(optimized_graph)

  # Because TF break the Prelu op into 6 ops, for performance we are
  # fusing those ops into a single prelu
  optimized_graph = fuse_prelu.fuse_ops_for_prelu(optimized_graph)

  # Because grappler does not support DepthwiseConv2d fusing, we have
  # implemented it here.
  optimized_graph = fuse_depthwise_conv2d.fuse_depthwise_conv2d(optimized_graph)

  # Since the grappler remap optimizer doe snot support prelu as the activation
  # function for _FusedConv2D op, we are doing it manually here.
  optimized_graph = fuse_prelu.fuse_prelu_with_fused_conv2d_or_matmul(
      optimized_graph)

  unsupported = validate(optimized_graph, skip_op_check,
                         strip_debug_ops)
  if unsupported:
    raise ValueError('Unsupported Ops in the model after optimization\n' +
                     ', '.join(unsupported))

  return optimized_graph

def extract_const_nodes(nodes):
  """Takes a list of nodes and extract the weights. Return weight manifest
  object.

  Args:
    nodes: list of tf.NodeDef TensorFlow NodeDef proto object.
  """
  constants = [node for node in nodes if node.op == 'Const']
  const_inputs = {}
  # removed the conditional inputs for constants
  for const in constants:
    const_inputs[const.name] = const.input[:]
    del const.input[:]

  const_manifest = []

  for const in constants:
    const_manifest.append({
        'name': const.name,
        'data': graph_rewrite_util.values_from_const(const)
    })
    # Restore the conditional inputs
    const.input[:] = const_inputs[const.name]

    # Remove the binary array from tensor and save it to the external file.
    for field_name in CLEARED_TENSOR_FIELDS:
      const.attr["value"].tensor.ClearField(field_name)

  return const_manifest

def extract_weights(graph_def, initializer_graph_def=None):
  """Takes a Python GraphDef object and extract the weights.

  Args:
    graph_def: tf.GraphDef TensorFlow GraphDef proto object, which represents
      the model topology.
    initializer_graph_def: tf.GraphDef proto object for initializer graph.
  """
  global_manifest = extract_const_nodes(graph_def.node)

  function_manifests = []
  for func in graph_def.library.function:
    nodes = graph_rewrite_util.rename_constants(
        func.node_def, func.signature.name)
    del func.node_def[:]
    func.node_def.extend(nodes)
    function_manifests += extract_const_nodes(func.node_def)

  initializer_manifests = []
  if initializer_graph_def:
    initializer_manifests = extract_const_nodes(initializer_graph_def.node)

  return [global_manifest + function_manifests + initializer_manifests]

def write_artifacts(topology,
                    weights,
                    output_graph,
                    tf_version,
                    signature_def,
                    quantization_dtype_map=None,
                    weight_shard_size_bytes=1024 * 1024 * 4,
                    initializer_graph_def=None,
                    initializer_signature_def=None,
                    resource_ids_maps=None,
                    metadata=None):
  """Writes weights and topology to the output_dir.

  If `topology` is Falsy (e.g., `None`), only emit weights to output_dir.

  Args:
    topology: tf.GraphDef TensorFlow GraphDef proto object, which represents
      the model topology.
    weights: an array of weight groups (as defined in tfjs write_weights).
    output_graph: the output file name to hold all the contents.
    tf_version: Tensorflow version of the input graph.
    signature_def: the SignatureDef of the inference graph.
    quantization_dtype_map: A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    initializer_graph_def: tf.GraphDef proto object for initializer graph.
    initializer_signature_def: the SignatureDef of the initializer graph.
    resource_ids_maps: Tuple of two dictionaries, one
      mapping inference input names to resource id, and the other
      mapping initializer output names to resource id.
    metadata: User defined metadata map.
  """
  model_json = {
      common.FORMAT_KEY: common.TFJS_GRAPH_MODEL_FORMAT,
      # TODO(piyu): Add tensorflow version below by using `meta_info_def`.
      common.GENERATED_BY_KEY: tf_version,
      common.CONVERTED_BY_KEY: common.get_converted_by(),
      common.SIGNATURE_KEY: MessageToDict(signature_def),
  }
  model_json[common.ARTIFACT_MODEL_TOPOLOGY_KEY] = topology or None

  if metadata:
    model_json[common.USER_DEFINED_METADATA_KEY] = metadata

  if initializer_graph_def and initializer_graph_def.node:
    model_json[common.ARTIFACT_MODEL_INITIALIZER] = MessageToDict(
        initializer_graph_def)
    if initializer_signature_def:
      model_json[common.INITIALIZER_SIGNATURE_KEY] = MessageToDict(
          initializer_signature_def)

  # Assign resource ids to inference inputs and initializer outputs. In
  # TensorFlow, both inference and initializer graphs have a reference
  # to the common resource (so initializer runs on reference, and then inference
  # graph uses it). We are doing something similar but instead of assigning
  # a reference to the resource in the serialized graph, we assign the id
  # of the resource, and then we can recreate the common reference in javascript
  # by matching resource ids.
  if resource_ids_maps is not None:
    model_input_to_resource_id, init_output_to_resource_id = resource_ids_maps
    signature_inputs = model_json[common.SIGNATURE_KEY]['inputs']
    initializer_signature_outputs = model_json[common.INITIALIZER_SIGNATURE_KEY]['outputs']

    for (input, resource_id) in model_input_to_resource_id.items():
      if input in signature_inputs:
        signature_inputs[input][common.RESOURCE_ID_KEY] = resource_id

    for (output, resource_id) in init_output_to_resource_id.items():
      if output in initializer_signature_outputs:
        initializer_signature_outputs[output][common.RESOURCE_ID_KEY] = resource_id


  weights_manifest = write_weights.write_weights(
      weights, os.path.dirname(output_graph), write_manifest=False,
      quantization_dtype_map=quantization_dtype_map,
      shard_size_bytes=weight_shard_size_bytes)
  assert isinstance(weights_manifest, list)
  model_json[common.ARTIFACT_WEIGHTS_MANIFEST_KEY] = weights_manifest

  with gfile.GFile(output_graph, 'w') as f:
    json.dump(model_json, f)

def _remove_unused_control_flow_inputs(input_graph_def):
  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if (node.op == 'Placeholder' and
        node.name.startswith('unused_control_flow_input')):
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    result_graph_def.node.extend([new_node])
  result_graph_def.library.CopyFrom(input_graph_def.library)
  result_graph_def.versions.CopyFrom(input_graph_def.versions)
  return result_graph_def

def _check_signature_in_model(saved_model, signature_name):
  if signature_name not in saved_model.signatures:
    raise ValueError("Signature '%s' does not exist. The following signatures "
                     "are available: %s" % (signature_name,
                                            saved_model.signatures.keys()))

def _copy_assets(saved_model_dir, output_dir):
  input_assets_path = os.path.join(saved_model_dir, common.ASSETS_DIRECTORY_NAME)

  if gfile.exists(input_assets_path) and gfile.isdir(input_assets_path):

    tmp_dir = tempfile.mkdtemp()
    zip_path = gfile.join(tmp_dir, common.ASSETS_DIRECTORY_NAME + '.zip')

    with ZipFile(zip_path, 'w') as archive:
      for (input_dir_path, _, file_names) in gfile.walk(input_assets_path):

        relative_dir_path = os.path.relpath(input_dir_path, input_assets_path)

        for file_name in file_names:

          input_file_path = gfile.join(input_dir_path, file_name)
          relative_file_path = gfile.join(relative_dir_path, file_name)

          with gfile.GFile(input_file_path, 'rb') as input_file:
            with archive.open(relative_file_path, 'w') as relative_file:
              shutil.copyfileobj(input_file, relative_file)

    output_assets_path = gfile.join(output_dir, common.ASSETS_DIRECTORY_NAME + '.zip')
    gfile.copy(zip_path, output_assets_path, overwrite=True)

    if gfile.isdir(tmp_dir):
      gfile.rmtree(tmp_dir)

def _is_assets_required(model_ops):
  # TFDF stores the necessary files for its binary in the assets folder.
  # Check if any TFDF ops are used in the model.
  with resource_loader.open_file('op_list/tfdf.json') as tfdf_json:
    ops = json.load(tfdf_json)
    opNames = frozenset([x['tfOpName'] for x in ops])
    return not opNames.isdisjoint(model_ops)

def _get_frozen_graph_ops(frozen_graph):
  if frozen_graph is None:
    return []
  return [node.op for node in frozen_graph.as_graph_def().node]


def _freeze_saved_model_v1(saved_model_dir, saved_model_tags,
                           output_node_names):
  """Freeze the graph by converting variables to constants for 1.x saved model.

  Args:
    saved_model_dir: dir where saved model files are stored.
    saved_model_tags: inference graph tag.
    output_node_names: List of name strings for the result nodes of the graph.

  Returns:
    A freezed and optimized graph.
    Nullable. A freezed and optimized initializer graph.
    Nullable. A list of output node names of initializer.
  """
  # v1 loader need empty list if there are no saved_model tags.
  if not saved_model_tags:
    saved_model_tags = []

  g = tf.Graph()
  with g.as_default():
    with tf.compat.v1.Session() as sess:
      meta_graph = loader.load(sess, saved_model_tags, saved_model_dir)

      meta_graph_def = g.as_graph_def()

      frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
          sess, meta_graph_def, output_node_names)

      frozen_graph = tf.Graph()
      with frozen_graph.as_default():
        tf.import_graph_def(frozen_graph_def, name='')

      frozen_initializer_graph = None
      initializer_output_names = None
      # Only support table initializers for now.
      if meta_graph.collection_def and meta_graph.collection_def[
          'table_initializer']:
        initializer_output_names = meta_graph.collection_def[
            'table_initializer'].node_list.value
        # This will use grappler to extract a subgraph with the
        # table initializer ops as the outputs.
        frozen_initializer_graph_def = (tf.compat.v1.graph_util
                                        .convert_variables_to_constants(
                                            sess, meta_graph_def,
                                            initializer_output_names))
        frozen_initializer_graph = tf.Graph()
        with frozen_initializer_graph.as_default():
          tf.import_graph_def(frozen_initializer_graph_def, name='')

      return frozen_graph, frozen_initializer_graph

def _freeze_saved_model_v2(concrete_func, control_flow_v2=False):
  if version.parse(tf.__version__) < version.parse('2.2.0'):
    return convert_to_constants.convert_variables_to_constants_v2(
        concrete_func, lower_control_flow=not control_flow_v2).graph

  return convert_to_constants.convert_variables_to_constants_v2(
      concrete_func, lower_control_flow=not control_flow_v2,
      aggressive_inlining=True).graph

def _find_signature_def_name(tensor, signature_map):
  if not signature_map:
    return tensor.name

  tensor_shape_str = tensor.shape.as_proto().SerializeToString()
  names = []
  for key in signature_map:
    tensor_info = signature_map[key]
    signature_shape_str = tensor_info.tensor_shape.SerializeToString()
    if (tensor_info.dtype == tensor.dtype and
        tensor_shape_str == signature_shape_str):
      names.append(key)

  if not names or len(names) > 1:
    return tensor.name
  else:
    return names[0]

def _build_signature_def(frozen_graph, input_nodes, output_nodes,
                         signature_def=None):
  signature = meta_graph_pb2.SignatureDef()
  for input_tensor in input_nodes:
    op_name = input_tensor.name.split(':')[0]
    # The graph freezing may turn the original inputs into constants, or remove
    # them from the graph, so we need to ignore those.
    try:
      op = frozen_graph.get_operation_by_name(op_name)
      if op.type != 'Const':
        name = input_tensor.name
        if hasattr(signature_def, 'inputs'):
          name = _find_signature_def_name(input_tensor, signature_def.inputs)
        signature.inputs[name].name = input_tensor.name
        signature.inputs[name].dtype = input_tensor.dtype.as_datatype_enum
        signature.inputs[name].tensor_shape.CopyFrom(
            input_tensor.shape.as_proto())
    except KeyError:
      # The original input was removed when the graph was frozen.
      continue
  for output_tensor in output_nodes:
    if hasattr(output_tensor, 'name'):
      name = output_tensor.name
      if hasattr(signature_def, 'inputs'):
        name = _find_signature_def_name(output_tensor, signature_def.outputs)
      signature.outputs[name].name = output_tensor.name
      signature.outputs[name].dtype = output_tensor.dtype.as_datatype_enum
      signature.outputs[name].tensor_shape.CopyFrom(
          output_tensor.shape.as_proto())
    else: #just the tensor name string array
      signature.outputs[output_tensor].name = output_tensor
  return signature

def convert_tf_frozen_model(frozen_model_path,
                            output_node_names,
                            output_dir,
                            quantization_dtype_map=None,
                            skip_op_check=False,
                            strip_debug_ops=False,
                            weight_shard_size_bytes=1024 * 1024 * 4,
                            experiments=False,
                            metadata=None):
  """Convert frozen model and check the model compatibility with Tensorflow.js.
  Optimize and convert the model to Tensorflow.js format, when the model passes
  the compatiblity check.
  Args:
    frozen_model_path: string The path to frozen model.
    output_node_names: string The names of the output nodes, comma separated.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    quantization_dtype_map: A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    experiments: Bool enable experimental features.
    metadata: User defined metadata map.
  """

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  output_graph = os.path.join(output_dir, common.ARTIFACT_MODEL_JSON_FILE_NAME)

  graph = load_graph(frozen_model_path)
  signature = _build_signature_def(
      graph, [], output_node_names.split(','))

  optimized_graph = optimize_graph(graph, signature,
                                   skip_op_check=skip_op_check,
                                   strip_debug_ops=strip_debug_ops,
                                   experiments=experiments)

  weights = extract_weights(optimized_graph)

  write_artifacts(MessageToDict(optimized_graph),
                 weights,
                 output_graph, tf.__version__,
                 signature,
                 quantization_dtype_map=quantization_dtype_map,
                 weight_shard_size_bytes=weight_shard_size_bytes,
                 metadata=metadata)

def _load_model(saved_model_dir, saved_model_tags):
  model = None
  # Ensure any graphs created in eager mode are able to run.
  with context.eager_mode():
    if saved_model_tags:
      model = load(saved_model_dir, saved_model_tags)
    else:
      model = load(saved_model_dir)
  return model

def _find_signature(saved_model_dir, saved_model_tags, signature_def):
  meta_graph = get_meta_graph_def(saved_model_dir, saved_model_tags)
  signature_def_map = meta_graph.signature_def
  if signature_def not in signature_def_map.keys():
    raise ValueError('Signature "%s" does not exist in the saved model'
                     % (signature_def))

  return signature_def_map[signature_def]

def _get_resource_initializer_concrete_function(model):
  """Create a tf.function that creates and initializes all the resources used by the model.
  For more information on resources, please see the TensorFlow code:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/trackable/resource.py#L232
  Args:
    model: Loaded saved model.

  Returns:
    Nullable. A concrete function.
  """
  trackable_view = TrackableView(model)
  model_resources = [obj for obj in trackable_view.descendants() if isinstance(obj, TrackableResource)]

  if not model_resources:
    return None

  # A list holding tuples of (TrackableResource, captured_input_index) where
  # TrackableResource represents one resource in the model
  # (a hash table for example), and captured_input_index is the resource
  # initialization function's captured input index corresponding
  # to the TrackableResource. Captured inputs are simply inputs not provided
  # directly be user, but by the model.
  model_resources_with_captured_input_index = []
  for model_resource in model_resources:
    # A runtime id that is unique across different resources, and constant
    # across graphs.
    resource_handle_id = model_resource.resource_handle._id
    # the _initialize function initializes the resource, so one of its captured
    # inputs must be the resource, so search for that input.
    captured_inputs = model_resource._initialize.get_concrete_function()._captured_inputs
    for captured_input_index in range(len(captured_inputs)):
      if captured_inputs[captured_input_index]._id == resource_handle_id:
        model_resources_with_captured_input_index.append((model_resource, captured_input_index))

  @tf.function()
  def resource_initializer():
    # Recreate resources to capture them in this tf.function.
    new_resources = []
    for (model_resource, captured_input_index) in model_resources_with_captured_input_index:
      # Make a new resource (that is identical to the old, but captured in
      # this functon only).
      new_resource = model_resource._create_resource()
      new_resources.append(new_resource)

      # Since we precomputed the captured input corresponding to this resource,
      # we can directly replace it with the copy new_resource. If we don't do
      # this, then _initialize will not get capture in this graph since the
      # old resource was already initialized in TF model load.
      model_resource._initialize.get_concrete_function()._captured_inputs[captured_input_index] = new_resource
      model_resource._initialize()

    return new_resources

  # Add resource_initializer to the output graph.
  return resource_initializer.get_concrete_function()

def _get_resource_ids_maps(model, concrete_func, resource_init_concrete_func):
  """Generates dictionaries that map tensor names to the loaded saved model resource id,
     allowing for matching of initializer outputs to inference inputs.

  Args:
    model: Loaded saved model.
    concrete_func: Concrete function of the inference graph.
    resource_init_concrete_func: Concrete function of the initializer graph.

  Returns:
    A dictionary mapping inference input names to resource id.
    A dictionary mapping initializer output names to resource id.
  """
  trackable_view = TrackableView(model)
  model_resources = [obj for obj in trackable_view.descendants() if isinstance(obj, TrackableResource)]


  # Each resource has a unique runtime resource id associated with it which
  # can be used across graphs, so we extract it here from inference
  # graph for use later.
  resource_id_to_captured_input_index = {
    captured_input._id : captured_input_index for \
    captured_input_index, captured_input in \
    enumerate(concrete_func._captured_inputs)
  }
  # Captured inputs always come after user provided inputs.
  captured_input_index_offset = len(concrete_func.inputs) - len(concrete_func._captured_inputs)

  model_input_to_resource_id = {}
  init_output_to_resource_id = {}
  for i, resource in enumerate(model_resources):
    _id = resource.resource_handle._id
    # Get input from inference graph corresponding to this resource.
    captured_input_index = resource_id_to_captured_input_index[_id]
    model_input = concrete_func.inputs[captured_input_index + captured_input_index_offset]

    # Get output from initializer graph corresponding to this resource.
    init_output = resource_init_concrete_func.outputs[i]

    # Match both with the same id (initializer output will be passed in to
    # corresponding input in inference input).
    model_input_to_resource_id[model_input.name] = _id
    init_output_to_resource_id[init_output.name] = _id

  return (model_input_to_resource_id, init_output_to_resource_id)

def _convert_tf_saved_model(output_dir,
                            saved_model_dir=None,
                            keras_model=None,
                            signature_def='serving_default',
                            saved_model_tags='serve',
                            quantization_dtype_map=None,
                            skip_op_check=False,
                            strip_debug_ops=False,
                            use_structured_outputs_names=False,
                            weight_shard_size_bytes=1024 * 1024 * 4,
                            control_flow_v2=False,
                            experiments=False,
                            metadata=None,
                            frozen_graph_dir=None):
  """Take a SavedModel or KerasModel and convert to Tensorflow.js graph model.

  Args:
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    saved_model_dir: string The saved model directory.
    : string The names of the output nodes, comma separated.
    keras_model: An in-memory Keras model object.
    signature_def: string Tagset of the SignatureDef to load. Defaults to
      'serving_default'.
    saved_model_tags: tags of the GraphDef to load. Defaults to 'serve'.
    quantization_dtype_map: A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
    use_structured_outputs_names: Bool whether output of graph model will follow
      the structured_outputs format.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    control_flow_v2: Bool whether to enable control flow v2 ops.
    experiments: Bool enable experimental features.
    metadata: User defined metadata map.
    frozen_graph_dir: The directory to keep the intermediate frozen graph of
      model.
  """
  if signature_def is None:
    signature_def = 'serving_default'

  if not gfile.exists(output_dir):
    gfile.makedirs(output_dir)
  output_graph = os.path.join(
      output_dir, common.ARTIFACT_MODEL_JSON_FILE_NAME)

  saved_model_tags_list = None
  if saved_model_tags:
    saved_model_tags_list = saved_model_tags.split(',')

  model = None
  concrete_func = None
  saved_model_sigature = None
  if saved_model_dir:
    saved_model_sigature = _find_signature(saved_model_dir, saved_model_tags,
                                           signature_def)
    model = _load_model(saved_model_dir, saved_model_tags_list)
    _check_signature_in_model(model, signature_def)
    concrete_func = model.signatures[signature_def]
  elif keras_model:
    model = keras_model
    input_signature = None
    # If the model's call is not a `tf.function`, then we need to first get its
    # input signature from `model_input_signature` method. We can't directly
    # call `trace_model_call` because otherwise the batch dimension is set
    # to None.
    if not isinstance(model.call, def_function.Function):
      # Pass `keep_original_batch_size=True` will ensure that we get an input
      # signature including the batch dimension specified by the user.
      input_signature = model_input_signature(
          model, keep_original_batch_size=True)
    func = trace_model_call(model, input_signature)
    concrete_func = func.get_concrete_function()
  else:
    raise Exception('Provide either a saved model or keras model to convert.')

  output_node_names = []
  for output_tensor in concrete_func.outputs:
    output_node_names.append(output_tensor.name.split(':')[0])

  num_outputs = len(output_node_names)
  structured_outputs = concrete_func.structured_outputs
  if use_structured_outputs_names and structured_outputs is not None:
    if not isinstance(structured_outputs, dict):
      raise Exception('Converter only supports dict structured_outputs.')

    # As per tensorflow/python/util/nest.py: "If `structure` is or contains a
    # dict instance, the keys will be sorted to pack the flat sequence
    # in deterministic order."
    sorted_keys = sorted(structured_outputs.keys())

    # Check if structure is a simple dictionary.
    # We don't support anything more complex due to the GraphModel.predict
    # function return type in typescript.
    test_sequence = list(range(num_outputs))
    actual_structure = tf.nest.pack_sequence_as(
        structured_outputs, test_sequence, True)
    expected_structure = dict(zip(sorted_keys, test_sequence))
    if actual_structure != expected_structure:
      raise Exception('Converter only supports structured_outputs of form '
                      '{"key1": value1, "key2":value2 ... })')

    metadata = metadata or {}
    metadata[common.STRUCTURED_OUTPUTS_KEYS_KEY] = sorted_keys

  # TensorFlow doesn't encode the saved model version in the graph in a
  # reliable way. Try to freeze the graph using V2 utils. If that fails, freeze
  # the graph using V1 utils.
  frozen_initializer_graph = None
  resource_ids_maps = None
  try:
    frozen_graph = _freeze_saved_model_v2(concrete_func, control_flow_v2)
    resource_initializer_concrete_func = _get_resource_initializer_concrete_function(model)

    if resource_initializer_concrete_func:
      frozen_initializer_graph = _freeze_saved_model_v2(resource_initializer_concrete_func, control_flow_v2)
      resource_ids_maps = _get_resource_ids_maps(model, concrete_func, resource_initializer_concrete_func)

  except BaseException:
    if saved_model_dir:
      (frozen_graph,
       frozen_initializer_graph) = _freeze_saved_model_v1(saved_model_dir,
                                                          saved_model_tags_list,
                                                          output_node_names)
    else:
      print('Can not freeze saved model v1.')
      return

  if frozen_graph_dir:
    output_graph = os.path.join(frozen_graph_dir,
                                common.ARTIFACT_MODEL_JSON_FILE_NAME)
    frozen_file = output_graph + '.frozen'
    with tf.compat.v1.gfile.GFile(frozen_file, 'wb') as f:
      f.write(frozen_graph.as_graph_def().SerializeToString())

  signature = _build_signature_def(
      frozen_graph, concrete_func.inputs, concrete_func.outputs, saved_model_sigature)

  define_transform_graph_func()

  tf_version = None
  try:
    tf_version = model.tensorflow_version
  except: # pylint: disable=W0702
    # keras model does not have tensorflow_version, hard code to the latest
    # tensorflow version.
    tf_version = tf.__version__

  if saved_model_dir:
      model_ops = set(_get_frozen_graph_ops(frozen_graph)) |\
                  set(_get_frozen_graph_ops(frozen_initializer_graph))
      if _is_assets_required(model_ops):
        _copy_assets(saved_model_dir, output_dir)

  optimized_graph = optimize_graph(frozen_graph, signature,
                                   skip_op_check=skip_op_check,
                                   strip_debug_ops=strip_debug_ops,
                                   experiments=experiments)

  initializer_graph_def = None
  initializer_signature_def = None
  if frozen_initializer_graph:
    initializer_graph_def = frozen_initializer_graph.as_graph_def()
    if hasattr(frozen_initializer_graph, 'outputs'):
      initializer_signature_def = _build_signature_def(frozen_initializer_graph, [], frozen_initializer_graph.outputs)

  weights = extract_weights(optimized_graph, initializer_graph_def)

  write_artifacts(MessageToDict(optimized_graph),
      weights,
      output_graph,
      tf_version, signature,
      quantization_dtype_map=quantization_dtype_map,
      weight_shard_size_bytes=weight_shard_size_bytes,
      initializer_graph_def=initializer_graph_def,
      initializer_signature_def=initializer_signature_def,
      resource_ids_maps=resource_ids_maps,
      metadata=metadata)

def define_transform_graph_func():
  """Check if the TransformGraph is available to be imported, this package is
  available in g3 but not in oss version of TensorFlow.
  """

  transform_graph_available = True
  try:
    from tensorflow.tools.graph_transforms import TransformGraph # pylint: disable=C0415
  except: # pylint: disable=W0702
    transform_graph_available = False

  # Define the strip graph functions when TransformGraph is available, this will
  # strip the unused nodes from the graph.
  if transform_graph_available:
    def _strip_unused_nodes(frozen_graph, concrete_func, output_node_names):
      # Find the names of the input nodes needed to extract the minimal
      # inference graph. This is particularly useful for cases when the concrete
      # function contains nodes that do not contribute the inference computation
      # defined by the input/output pair. This would also eliminate op
      # unsupported error caused by nodes outside of the minial infrerence
      # graph.
      input_node_names = []
      input_tensors = {}
      for input_tensor in concrete_func.inputs:
        if input_tensor.dtype != 'resource':
          op_name = input_tensor.name.split(':')[0]
          # The graph freezing may turn the original inputs into constants, or
          # remove them from the graph, so we need to ignore those.
          try:
            op = frozen_graph.get_operation_by_name(op_name)
            if op.type != 'Const':
              input_node_names.append(op_name)
              input_tensors[op_name] = input_tensor
          except KeyError:
            # The original input was removed when the graph was frozen.
            continue

      graph_transformations = ['strip_unused_nodes']
      stripped_graph_def = TransformGraph(
          frozen_graph.as_graph_def(), input_node_names, output_node_names,
          graph_transformations)

      # The transform graph library cannot support input nodes that has dynamic
      # shape, this code will update the dtype and shape based on the
      # input tensor manually.
      for node in stripped_graph_def.node:
        if node.name in input_tensors:
          if node.attr['shape'] and node.attr['shape'].shape:
            node.attr['shape'].shape.CopyFrom(
                input_tensors[node.name].shape.as_proto())
          if node.attr['dtype'] and node.attr['dtype'].type:
            node.attr['dtype'].type = input_tensors[
                node.name].dtype.as_datatype_enum

      with tf.Graph().as_default() as stripped_graph:
        tf.import_graph_def(stripped_graph_def, name='')
        return stripped_graph

def convert_tf_saved_model(saved_model_dir,
                           output_dir, signature_def='serving_default',
                           saved_model_tags='serve',
                           quantization_dtype_map=None,
                           skip_op_check=False,
                           strip_debug_ops=False,
                           use_structured_outputs_names=False,
                           weight_shard_size_bytes=1024 * 1024 * 4,
                           control_flow_v2=False,
                           experiments=False,
                           metadata=None,
                           frozen_graph_dir=None):
  """Freeze the SavedModel and check the model compatibility with Tensorflow.js.

  Optimize and convert the model to Tensorflow.js format, when the model passes
  the compatiblity check.

  Args:
    saved_model_dir: string The saved model directory.
    : string The names of the output nodes, comma separated.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    signature_def: string Tagset of the SignatureDef to load. Defaults to
      'serving_default'.
    saved_model_tags: tags of the GraphDef to load. Defaults to 'serve'.
    quantization_dtype_map: A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
    use_structured_outputs_names: Bool whether output of graph model will follow
      the structured_outputs format.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    control_flow_v2: Bool whether to enable control flow v2 ops.
    experiments: Bool enable experimental features.
    metadata: User defined metadata map.
    frozen_graph_dir: The directory to keep the intermediate frozen graph of
      model.
  """
  _convert_tf_saved_model(output_dir, saved_model_dir=saved_model_dir,
                          signature_def=signature_def,
                          saved_model_tags=saved_model_tags,
                          quantization_dtype_map=quantization_dtype_map,
                          skip_op_check=skip_op_check,
                          strip_debug_ops=strip_debug_ops,
                          use_structured_outputs_names=
                          use_structured_outputs_names,
                          weight_shard_size_bytes=weight_shard_size_bytes,
                          control_flow_v2=control_flow_v2,
                          experiments=experiments,
                          metadata=metadata,
                          frozen_graph_dir=frozen_graph_dir)

def load_and_initialize_hub_module(module_path, signature='default'):
  """Loads graph of a TF-Hub module and initializes it into a session.

  Args:
    module_path: string Path to TF-Hub module.
    signature: string Signature to use when creating the apply graph.

  Return:
    graph: tf.Graph Graph of the module.
    session: tf.Session Session with initialized variables and tables.
    inputs: dict Dictionary of input tensors.
    outputs: dict Dictionary of output tensors.

  Raises:
    ValueError: If signature contains a SparseTensor on input or output.
  """
  graph = tf.Graph()
  with graph.as_default():
    tf.compat.v1.logging.info('Importing %s', module_path)
    module = hub.Module(module_path)

    signature_inputs = module.get_input_info_dict(signature)
    signature_outputs = module.get_output_info_dict(signature)
    # First check there are no SparseTensors in input or output.
    for key, info in list(signature_inputs.items()) + list(
        signature_outputs.items()):
      if info.is_sparse:
        raise ValueError(
            'Signature "%s" has a SparseTensor on input/output "%s".'
            ' SparseTensors are not supported.' % (signature, key))

    # Create placeholders to represent the input of the provided signature.
    inputs = {}
    for input_key, input_info in signature_inputs.items():
      inputs[input_key] = tf.compat.v1.placeholder(
          shape=input_info.get_shape(), dtype=input_info.dtype, name=input_key)

    outputs = module(inputs=inputs, signature=signature, as_dict=True)

    session = tf.compat.v1.Session(graph=graph)
    session.run(tf.compat.v1.global_variables_initializer())
    session.run(tf.compat.v1.tables_initializer())

  return graph, session, inputs, outputs


def convert_tf_hub_module_v1(module_path, output_dir,
                             signature='default', quantization_dtype_map=None,
                             skip_op_check=False, strip_debug_ops=False,
                             weight_shard_size_bytes=1024 * 1024 * 4,
                             experiments=False,
                             metadata=None):
  """Freeze the TF-Hub module and check compatibility with Tensorflow.js.

  Optimize and convert the TF-Hub module to Tensorflow.js format, if it passes
  the compatiblity check.

  Args:
    module_path: string Path to the module.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    signature: string Signature to load.
    quantization_dtype_map: A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    experiments: Bool enable experimental features.
    metadata: User defined metadata map.
  """

  if signature is None:
    signature = 'default'

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  graph, sess, inputs, outputs = load_and_initialize_hub_module(
      module_path, signature)

  input_node_names = []
  output_node_names = []

  for input_tensor in inputs.values():
    input_node_names.append(input_tensor.name.split(':')[0])
  for output_tensor in outputs.values():
    output_node_names.append(output_tensor.name.split(':')[0])

  print('Creating a model with inputs %s and outputs %s.' % (input_node_names,
                                                             output_node_names))

  frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), output_node_names)

  output_graph = os.path.join(output_dir, common.ARTIFACT_MODEL_JSON_FILE_NAME)
  frozen_file = output_graph + '.frozen'
  try:
    with tf.compat.v1.gfile.GFile(frozen_file, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

    frozen_graph = load_graph(frozen_file)
    signature = _build_signature_def(frozen_graph,
                                     inputs.values(), outputs.values())

    optimized_graph = optimize_graph(frozen_graph, signature,
                                     skip_op_check=skip_op_check,
                                     strip_debug_ops=strip_debug_ops,
                                     experiments=experiments)

    weights = extract_weights(optimized_graph)

    write_artifacts(MessageToDict(optimized_graph), weights,
                   output_graph, tf.__version__, signature,
                   quantization_dtype_map=quantization_dtype_map,
                   weight_shard_size_bytes=weight_shard_size_bytes,
                   metadata=metadata)
  finally:
    # Clean up the temp files.
    if os.path.exists(frozen_file):
      os.remove(frozen_file)


def convert_tf_hub_module(module_handle, output_dir,
                          signature='default', saved_model_tags='serve',
                          quantization_dtype_map=None,
                          skip_op_check=False, strip_debug_ops=False,
                          use_structured_outputs_names=False,
                          weight_shard_size_bytes=1024 * 1024 * 4,
                          control_flow_v2=False,
                          experiments=False,
                          metadata=None):
  """Conversion for TF Hub modules V1 and V2.

  See convert_tf_hub_module and convert_tf_saved_model.

  Args:
    module_path: string Path to the module.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    signature: string Signature to load.
    saved_model_tags: tags of the GraphDef to load. Defaults to ''.
    quantization_dtype_map: A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
    use_structured_outputs_names: Bool whether output of graph model will follow
      the structured_outputs format.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    control_flow_v2: Bool whether to enable control flow v2 ops.
    experiments: Bool enable experimental features.
    metadata: User defined metadata map.
  """
  module_path = hub.resolve(module_handle)
  # TODO(vbardiovskyg): We can remove this v1 code path once loading of all v1
  # modules is fixed on the TF side, or once the modules we cannot load become
  # replaced with newer versions.
  if gfile.exists(os.path.join(module_path, _HUB_V1_MODULE_PB)):
    print("Loading the module using TF 1.X interface from %s." % module_path)
    convert_tf_hub_module_v1(module_path, output_dir, signature,
                             quantization_dtype_map,
                             skip_op_check, strip_debug_ops,
                             weight_shard_size_bytes,
                             experiments=experiments,
                             metadata=metadata)
  else:
    print("Loading the module using TF 2.X interface from %s." % module_path)
    if signature is None:
      signature = 'default'
    convert_tf_saved_model(saved_model_dir=module_path,
                           output_dir=output_dir,
                           signature_def=signature,
                           saved_model_tags=saved_model_tags,
                           quantization_dtype_map=quantization_dtype_map,
                           skip_op_check=skip_op_check,
                           strip_debug_ops=strip_debug_ops,
                           use_structured_outputs_names=
                           use_structured_outputs_names,
                           weight_shard_size_bytes=weight_shard_size_bytes,
                           control_flow_v2=control_flow_v2,
                           experiments=experiments,
                           metadata=metadata)

def convert_keras_model_to_graph_model(keras_model,
                                       output_dir,
                                       saved_model_tags='serve',
                                       quantization_dtype_map=None,
                                       skip_op_check=False,
                                       strip_debug_ops=False,
                                       use_structured_outputs_names=False,
                                       weight_shard_size_bytes=1024 * 1024 * 4,
                                       control_flow_v2=False,
                                       experiments=False,
                                       metadata=None):
  """Convert an in-memory keras model to Tensorflow.js graph model format.

  Args:
    keras_model: Keras Model object.
    output_dir: string The name of the output directory. The directory
      will consist of
      - a file named 'model.json'
      - possibly sharded binary weight files.
    saved_model_tags: tags of the GraphDef to load. Defaults to 'serve'.
    quantization_dtype_map: A mapping from dtype
      (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
      supports wildcard substitution.
    skip_op_check: Bool whether to skip the op check.
    strip_debug_ops: Bool whether to strip debug ops.
    use_structured_outputs_names: Bool whether output of graph model will follow
      the structured_outputs format.
    weight_shard_size_bytes: Shard size (in bytes) of the weight files.
      The size of each weight file will be <= this value.
    control_flow_v2: Bool whether to enable control flow v2 ops.
    experiments: Bool enable experimental features.
    metadata: User defined metadata map.
  """
  _convert_tf_saved_model(output_dir, keras_model=keras_model,
                          saved_model_tags=saved_model_tags,
                          quantization_dtype_map=quantization_dtype_map,
                          skip_op_check=skip_op_check,
                          strip_debug_ops=strip_debug_ops,
                          use_structured_outputs_names=
                          use_structured_outputs_names,
                          weight_shard_size_bytes=weight_shard_size_bytes,
                          control_flow_v2=control_flow_v2,
                          experiments=experiments,
                          metadata=metadata)
