bayesflow.approximators.backend\_approximators.backend\_approximator.BaseBackendApproximator
============================================================================================

.. currentmodule:: bayesflow.approximators.backend_approximators.backend_approximator

.. autoclass:: BaseBackendApproximator
   :members:                                    
   :show-inheritance:                           
   :inherited-members:                          

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~BaseBackendApproximator.__init__
      ~BaseBackendApproximator.add_loss
      ~BaseBackendApproximator.add_metric
      ~BaseBackendApproximator.add_variable
      ~BaseBackendApproximator.add_weight
      ~BaseBackendApproximator.build
      ~BaseBackendApproximator.build_from_config
      ~BaseBackendApproximator.call
      ~BaseBackendApproximator.compile
      ~BaseBackendApproximator.compile_from_config
      ~BaseBackendApproximator.compiled_loss
      ~BaseBackendApproximator.compute_loss
      ~BaseBackendApproximator.compute_mask
      ~BaseBackendApproximator.compute_metrics
      ~BaseBackendApproximator.compute_output_shape
      ~BaseBackendApproximator.compute_output_spec
      ~BaseBackendApproximator.count_params
      ~BaseBackendApproximator.evaluate
      ~BaseBackendApproximator.export
      ~BaseBackendApproximator.fit
      ~BaseBackendApproximator.from_config
      ~BaseBackendApproximator.get_build_config
      ~BaseBackendApproximator.get_compile_config
      ~BaseBackendApproximator.get_config
      ~BaseBackendApproximator.get_layer
      ~BaseBackendApproximator.get_metrics_result
      ~BaseBackendApproximator.get_state_tree
      ~BaseBackendApproximator.get_weights
      ~BaseBackendApproximator.load_own_variables
      ~BaseBackendApproximator.load_weights
      ~BaseBackendApproximator.loss
      ~BaseBackendApproximator.make_predict_function
      ~BaseBackendApproximator.make_test_function
      ~BaseBackendApproximator.make_train_function
      ~BaseBackendApproximator.predict
      ~BaseBackendApproximator.predict_on_batch
      ~BaseBackendApproximator.predict_step
      ~BaseBackendApproximator.quantize
      ~BaseBackendApproximator.quantized_build
      ~BaseBackendApproximator.quantized_call
      ~BaseBackendApproximator.reset_metrics
      ~BaseBackendApproximator.save
      ~BaseBackendApproximator.save_own_variables
      ~BaseBackendApproximator.save_weights
      ~BaseBackendApproximator.set_state_tree
      ~BaseBackendApproximator.set_weights
      ~BaseBackendApproximator.stateless_call
      ~BaseBackendApproximator.stateless_compute_loss
      ~BaseBackendApproximator.summary
      ~BaseBackendApproximator.symbolic_call
      ~BaseBackendApproximator.test_on_batch
      ~BaseBackendApproximator.test_step
      ~BaseBackendApproximator.to_json
      ~BaseBackendApproximator.train_on_batch
      ~BaseBackendApproximator.train_step
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~BaseBackendApproximator.compiled_metrics
      ~BaseBackendApproximator.compute_dtype
      ~BaseBackendApproximator.distribute_reduction_method
      ~BaseBackendApproximator.distribute_strategy
      ~BaseBackendApproximator.dtype
      ~BaseBackendApproximator.dtype_policy
      ~BaseBackendApproximator.input
      ~BaseBackendApproximator.input_dtype
      ~BaseBackendApproximator.input_spec
      ~BaseBackendApproximator.jit_compile
      ~BaseBackendApproximator.layers
      ~BaseBackendApproximator.losses
      ~BaseBackendApproximator.metrics
      ~BaseBackendApproximator.metrics_names
      ~BaseBackendApproximator.metrics_variables
      ~BaseBackendApproximator.non_trainable_variables
      ~BaseBackendApproximator.non_trainable_weights
      ~BaseBackendApproximator.output
      ~BaseBackendApproximator.path
      ~BaseBackendApproximator.quantization_mode
      ~BaseBackendApproximator.run_eagerly
      ~BaseBackendApproximator.supports_masking
      ~BaseBackendApproximator.trainable
      ~BaseBackendApproximator.trainable_variables
      ~BaseBackendApproximator.trainable_weights
      ~BaseBackendApproximator.variable_dtype
      ~BaseBackendApproximator.variables
      ~BaseBackendApproximator.weights
   
   