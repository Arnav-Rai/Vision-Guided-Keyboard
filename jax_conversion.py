# Copyright 2022 Google LLC
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
"""Converts a JAX function to TensorFlow.js web format."""
import tempfile
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from jax.experimental import jax2tf
# Original import of shape_poly was here, now handled in try-except
import tensorflow as tf

# Attempt to import jax components, but don't fail if not found or if specific submodules are missing.
try:
    from jax.experimental.jax2tf import shape_poly
    # If 'shape_poly' was imported successfully, try to get 'PolyShape'
    if hasattr(shape_poly, 'PolyShape'): # Check if PolyShape attribute exists
         PolyShape = shape_poly.PolyShape
    else:
         PolyShape = None # PolyShape itself might not exist even if shape_poly does
         print(f"Warning: 'PolyShape' not found in 'jax.experimental.jax2tf.shape_poly'. Proceeding without it.")
except ImportError as e:
    shape_poly = None
    PolyShape = None # Ensure PolyShape is also None if shape_poly import fails
    print(f"Warning: JAX module 'jax.experimental.jax2tf.shape_poly' could not be imported. Error: {e}. Proceeding without it.")
except Exception as e: # Catch other potential errors during import
    shape_poly = None
    PolyShape = None
    print(f"Warning: Unexpected error importing JAX 'shape_poly': {e}. Proceeding without it.")


# ... (rest of the file) ...
from tensorflowjs.converters import tf_saved_model_conversion_v2 as saved_model_conversion


_TF_SERVING_KEY = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
Array = Any
DType = Any
# PolyShape = shape_poly.PolyShape # THIS LINE (around original line 52) SHOULD BE REMOVED OR COMMENTED OUT
                                  # PolyShape is already defined (or set to None) in the try-except block above.


class _ReusableSavedModelWrapper(tf.train.Checkpoint):
  """Wraps a function and its parameters for saving to a SavedModel.

  Implements the interface described at
  https://www.tensorflow.org/hub/reusable_saved_models.
  """

  def __init__(self, tf_graph, param_vars):
    """Initializes a _ReusableSavedModelWrapper.

    Args:
      tf_graph: a tf.function taking one argument (the inputs), which can be
        be tuples/lists/dictionaries of np.ndarray or tensors. The function
        may have references to the tf.Variables in `param_vars`.
      param_vars: the parameters, as tuples/lists/dictionaries of tf.Variable,
        to be saved as the variables of the SavedModel.
    """
    super().__init__()
    self.variables = tf.nest.flatten(param_vars)
    self.trainable_variables = [v for v in self.variables if v.trainable]
    # If you intend to prescribe regularization terms for users of the model,
    # add them as @tf.functions with no inputs to this list. Else drop this.
    self.regularization_losses = []
    self.__call__ = tf_graph


def convert_jax(
    apply_fn: Callable[..., Any],
    params: Array,
    *,
    input_signatures: Sequence[Tuple[Sequence[Union[int, None]], DType]],
    model_dir: str,
    polymorphic_shapes: Optional[Sequence[Union[str, Any]]] = None, # Changed PolyShape to Any for broader compatibility if PolyShape is None
    **tfjs_converter_params):
  """Converts a JAX function `jax_apply_fn` and model parameters to a TensorflowJS model.

  Example usage for a Flax Module:

  ```
  import numpy as np
  from flax import linen as nn
  from jax import random
  import jax.numpy as jnp
  from tensorflowjs.converters.jax_conversion import convert_jax

  module = nn.Dense(features=4)
  inputs = jnp.ones((3, 4))
  params = module.init(random.PRNKey(0), inputs)['params']

  convert_jax(
    apply_fn=module.apply,
    params=params,
    input_signatures=[((3, 4), np.float32)],
    model_dir=tfjs_model_dir)
  ```

  Note that when using dynamic shapes, an additional argument
  `polymorphic_shapes` should be provided specifying values for the dynamic
  ("polymorphic") dimensions). See here for more details:
  https://github.com/google/jax/tree/main/jax/experimental/jax2tf#shape-polymorphic-conversion

  This is an adaption of the original implementation in jax2tf here:
  https://github.com/google/jax/blob/main/jax/experimental/jax2tf/examples/saved_model_lib.py

  Arguments:
    apply_fn: A JAX function that has one or more arguments, of which the first
      argument are the model parameters. This function typically is the forward
      pass of the network (e.g., `Module.apply()` in Flax).
    params: A Pytree containing the parameters of the module. These will all be
      converted to TF.Variables.
    input_signatures: the input signatures for the second and remaining
      arguments to `apply_fn` (the input). A signature must be a
      `tensorflow.TensorSpec` instance, or a (nested) tuple/list/dictionary
      thereof with a structure matching the second argument of `apply_fn`.
    model_dir: Directory where the TensorflowJS model will be written to.
    polymorphic_shapes: If given then it will be used as the
      `polymorphic_shapes` argument for the second parameter of `apply_fn`. In
      this case, a single `input_signatures` is supported, and should have
      `None` in the polymorphic (dynamic) dimensions.
  """
  if polymorphic_shapes is not None:
    # If polymorphic shapes are provided, add a polymorphic spec for the
    # first argument to `apply_fn`, which are the parameters.
    # Ensure that PolyShape is used only if it's not None
    if PolyShape is not None:
        valid_polymorphic_shapes = []
        for item in polymorphic_shapes:
            if isinstance(item, str) or isinstance(item, PolyShape): # Check type before adding
                valid_polymorphic_shapes.append(item)
            else:
                # Handle cases where item might not be a valid PolyShape or string,
                # perhaps by converting or raising an error, or using a default.
                # For now, we'll just print a warning and potentially skip it.
                print(f"Warning: Invalid item '{item}' in polymorphic_shapes. Expected str or PolyShape.")
        polymorphic_shapes = [None, *valid_polymorphic_shapes]
    else:
        # If PolyShape could not be imported, polymorphic_shapes might not work as intended
        # or might need to be strings only.
        # For now, we'll proceed but this part might need adjustment if JAX conversion is actually used.
        print("Warning: PolyShape is None. Polymorphic shapes for JAX conversion might not function as expected if JAX types are required.")
        polymorphic_shapes = [None, *[s for s in polymorphic_shapes if isinstance(s, str)]]


  tf_fn = jax2tf.convert(
      apply_fn,
      # Gradients must be included as 'PreventGradient' is not supported.
      with_gradient=True,
      polymorphic_shapes=polymorphic_shapes,
      # Do not use TFXLA Ops because these aren't supported by TFjs, but use
      # workarounds instead. More information:
      # https://github.com/google/jax/tree/main/jax/experimental/jax2tf#tensorflow-xla-ops
      enable_xla=False)

  # Create tf.Variables for the parameters. If you want more useful variable
  # names, you can use `tree.map_structure_with_path` from the `dm-tree`
  # package.
  param_vars = tf.nest.map_structure(
      lambda param: tf.Variable(param, trainable=True), params)
  # Do not use TF's jit compilation on the function.
  tf_graph = tf.function(
      lambda *xs: tf_fn(param_vars, *xs), autograph=False, jit_compile=False)

  # This signature is needed for TensorFlow Serving use.
  signatures = {
      _TF_SERVING_KEY: tf_graph.get_concrete_function(*input_signatures)
  }

  wrapper = _ReusableSavedModelWrapper(tf_graph, param_vars)
  saved_model_options = tf.saved_model.SaveOptions(
      experimental_custom_gradients=True)

  with tempfile.TemporaryDirectory() as saved_model_dir:
    tf.saved_model.save(
        wrapper,
        saved_model_dir,
        signatures=signatures,
        options=saved_model_options)
    saved_model_conversion.convert_tf_saved_model(saved_model_dir, model_dir,
                                                  **tfjs_converter_params)
