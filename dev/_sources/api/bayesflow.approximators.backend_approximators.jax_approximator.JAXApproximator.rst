bayesflow.approximators.backend\_approximators.jax\_approximator.JAXApproximator
================================================================================

.. currentmodule:: bayesflow.approximators.backend_approximators.jax_approximator

.. autoclass:: JAXApproximator
   :members:                                    
   :show-inheritance:                           
   :inherited-members:                          

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~JAXApproximator.__init__
      ~JAXApproximator.add_loss
      ~JAXApproximator.add_metric
      ~JAXApproximator.add_variable
      ~JAXApproximator.add_weight
      ~JAXApproximator.build
      ~JAXApproximator.build_from_config
      ~JAXApproximator.call
      ~JAXApproximator.compile
      ~JAXApproximator.compile_from_config
      ~JAXApproximator.compiled_loss
      ~JAXApproximator.compute_loss
      ~JAXApproximator.compute_mask
      ~JAXApproximator.compute_metrics
      ~JAXApproximator.compute_output_shape
      ~JAXApproximator.compute_output_spec
      ~JAXApproximator.count_params
      ~JAXApproximator.evaluate
      ~JAXApproximator.export
      ~JAXApproximator.fit
      ~JAXApproximator.from_config
      ~JAXApproximator.get_build_config
      ~JAXApproximator.get_compile_config
      ~JAXApproximator.get_config
      ~JAXApproximator.get_layer
      ~JAXApproximator.get_metrics_result
      ~JAXApproximator.get_state_tree
      ~JAXApproximator.get_weights
      ~JAXApproximator.load_own_variables
      ~JAXApproximator.load_weights
      ~JAXApproximator.loss
      ~JAXApproximator.make_predict_function
      ~JAXApproximator.make_test_function
      ~JAXApproximator.make_train_function
      ~JAXApproximator.predict
      ~JAXApproximator.predict_on_batch
      ~JAXApproximator.predict_step
      ~JAXApproximator.quantize
      ~JAXApproximator.quantized_build
      ~JAXApproximator.quantized_call
      ~JAXApproximator.reset_metrics
      ~JAXApproximator.save
      ~JAXApproximator.save_own_variables
      ~JAXApproximator.save_weights
      ~JAXApproximator.set_state_tree
      ~JAXApproximator.set_weights
      ~JAXApproximator.stateless_call
      ~JAXApproximator.stateless_compute_loss
      ~JAXApproximator.stateless_compute_metrics
      ~JAXApproximator.stateless_test_step
      ~JAXApproximator.stateless_train_step
      ~JAXApproximator.summary
      ~JAXApproximator.symbolic_call
      ~JAXApproximator.test_on_batch
      ~JAXApproximator.test_step
      ~JAXApproximator.to_json
      ~JAXApproximator.train_on_batch
      ~JAXApproximator.train_step
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~JAXApproximator.compiled_metrics
      ~JAXApproximator.compute_dtype
      ~JAXApproximator.distribute_reduction_method
      ~JAXApproximator.distribute_strategy
      ~JAXApproximator.dtype
      ~JAXApproximator.dtype_policy
      ~JAXApproximator.input
      ~JAXApproximator.input_dtype
      ~JAXApproximator.input_spec
      ~JAXApproximator.jit_compile
      ~JAXApproximator.layers
      ~JAXApproximator.losses
      ~JAXApproximator.metrics
      ~JAXApproximator.metrics_names
      ~JAXApproximator.metrics_variables
      ~JAXApproximator.non_trainable_variables
      ~JAXApproximator.non_trainable_weights
      ~JAXApproximator.output
      ~JAXApproximator.path
      ~JAXApproximator.quantization_mode
      ~JAXApproximator.run_eagerly
      ~JAXApproximator.supports_masking
      ~JAXApproximator.trainable
      ~JAXApproximator.trainable_variables
      ~JAXApproximator.trainable_weights
      ~JAXApproximator.variable_dtype
      ~JAXApproximator.variables
      ~JAXApproximator.weights
   
   