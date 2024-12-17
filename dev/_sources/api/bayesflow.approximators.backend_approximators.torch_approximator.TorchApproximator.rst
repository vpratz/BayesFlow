bayesflow.approximators.backend\_approximators.torch\_approximator.TorchApproximator
====================================================================================

.. currentmodule:: bayesflow.approximators.backend_approximators.torch_approximator

.. autoclass:: TorchApproximator
   :members:                                    
   :show-inheritance:                           
   :inherited-members:                          

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~TorchApproximator.__init__
      ~TorchApproximator.add_loss
      ~TorchApproximator.add_metric
      ~TorchApproximator.add_variable
      ~TorchApproximator.add_weight
      ~TorchApproximator.build
      ~TorchApproximator.build_from_config
      ~TorchApproximator.call
      ~TorchApproximator.compile
      ~TorchApproximator.compile_from_config
      ~TorchApproximator.compiled_loss
      ~TorchApproximator.compute_loss
      ~TorchApproximator.compute_mask
      ~TorchApproximator.compute_metrics
      ~TorchApproximator.compute_output_shape
      ~TorchApproximator.compute_output_spec
      ~TorchApproximator.count_params
      ~TorchApproximator.evaluate
      ~TorchApproximator.export
      ~TorchApproximator.fit
      ~TorchApproximator.from_config
      ~TorchApproximator.get_build_config
      ~TorchApproximator.get_compile_config
      ~TorchApproximator.get_config
      ~TorchApproximator.get_layer
      ~TorchApproximator.get_metrics_result
      ~TorchApproximator.get_state_tree
      ~TorchApproximator.get_weights
      ~TorchApproximator.load_own_variables
      ~TorchApproximator.load_weights
      ~TorchApproximator.loss
      ~TorchApproximator.make_predict_function
      ~TorchApproximator.make_test_function
      ~TorchApproximator.make_train_function
      ~TorchApproximator.predict
      ~TorchApproximator.predict_on_batch
      ~TorchApproximator.predict_step
      ~TorchApproximator.quantize
      ~TorchApproximator.quantized_build
      ~TorchApproximator.quantized_call
      ~TorchApproximator.reset_metrics
      ~TorchApproximator.save
      ~TorchApproximator.save_own_variables
      ~TorchApproximator.save_weights
      ~TorchApproximator.set_state_tree
      ~TorchApproximator.set_weights
      ~TorchApproximator.stateless_call
      ~TorchApproximator.stateless_compute_loss
      ~TorchApproximator.summary
      ~TorchApproximator.symbolic_call
      ~TorchApproximator.test_on_batch
      ~TorchApproximator.test_step
      ~TorchApproximator.to_json
      ~TorchApproximator.train_on_batch
      ~TorchApproximator.train_step
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~TorchApproximator.compiled_metrics
      ~TorchApproximator.compute_dtype
      ~TorchApproximator.distribute_reduction_method
      ~TorchApproximator.distribute_strategy
      ~TorchApproximator.dtype
      ~TorchApproximator.dtype_policy
      ~TorchApproximator.input
      ~TorchApproximator.input_dtype
      ~TorchApproximator.input_spec
      ~TorchApproximator.jit_compile
      ~TorchApproximator.layers
      ~TorchApproximator.losses
      ~TorchApproximator.metrics
      ~TorchApproximator.metrics_names
      ~TorchApproximator.metrics_variables
      ~TorchApproximator.non_trainable_variables
      ~TorchApproximator.non_trainable_weights
      ~TorchApproximator.output
      ~TorchApproximator.path
      ~TorchApproximator.quantization_mode
      ~TorchApproximator.run_eagerly
      ~TorchApproximator.supports_masking
      ~TorchApproximator.trainable
      ~TorchApproximator.trainable_variables
      ~TorchApproximator.trainable_weights
      ~TorchApproximator.variable_dtype
      ~TorchApproximator.variables
      ~TorchApproximator.weights
   
   