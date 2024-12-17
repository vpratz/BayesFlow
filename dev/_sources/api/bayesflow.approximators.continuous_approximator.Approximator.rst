bayesflow.approximators.continuous\_approximator.Approximator
=============================================================

.. currentmodule:: bayesflow.approximators.continuous_approximator

.. autoclass:: Approximator
   :members:                                    
   :show-inheritance:                           
   :inherited-members:                          

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Approximator.__init__
      ~Approximator.add_loss
      ~Approximator.add_metric
      ~Approximator.add_variable
      ~Approximator.add_weight
      ~Approximator.build
      ~Approximator.build_adapter
      ~Approximator.build_dataset
      ~Approximator.build_from_config
      ~Approximator.build_from_data
      ~Approximator.call
      ~Approximator.compile
      ~Approximator.compile_from_config
      ~Approximator.compiled_loss
      ~Approximator.compute_loss
      ~Approximator.compute_mask
      ~Approximator.compute_metrics
      ~Approximator.compute_output_shape
      ~Approximator.compute_output_spec
      ~Approximator.count_params
      ~Approximator.evaluate
      ~Approximator.export
      ~Approximator.fit
      ~Approximator.from_config
      ~Approximator.get_build_config
      ~Approximator.get_compile_config
      ~Approximator.get_config
      ~Approximator.get_layer
      ~Approximator.get_metrics_result
      ~Approximator.get_state_tree
      ~Approximator.get_weights
      ~Approximator.load_own_variables
      ~Approximator.load_weights
      ~Approximator.loss
      ~Approximator.make_predict_function
      ~Approximator.make_test_function
      ~Approximator.make_train_function
      ~Approximator.predict
      ~Approximator.predict_on_batch
      ~Approximator.predict_step
      ~Approximator.quantize
      ~Approximator.quantized_build
      ~Approximator.quantized_call
      ~Approximator.reset_metrics
      ~Approximator.save
      ~Approximator.save_own_variables
      ~Approximator.save_weights
      ~Approximator.set_state_tree
      ~Approximator.set_weights
      ~Approximator.stateless_call
      ~Approximator.stateless_compute_loss
      ~Approximator.summary
      ~Approximator.symbolic_call
      ~Approximator.test_on_batch
      ~Approximator.test_step
      ~Approximator.to_json
      ~Approximator.train_on_batch
      ~Approximator.train_step
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Approximator.compiled_metrics
      ~Approximator.compute_dtype
      ~Approximator.distribute_reduction_method
      ~Approximator.distribute_strategy
      ~Approximator.dtype
      ~Approximator.dtype_policy
      ~Approximator.input
      ~Approximator.input_dtype
      ~Approximator.input_spec
      ~Approximator.jit_compile
      ~Approximator.layers
      ~Approximator.losses
      ~Approximator.metrics
      ~Approximator.metrics_names
      ~Approximator.metrics_variables
      ~Approximator.non_trainable_variables
      ~Approximator.non_trainable_weights
      ~Approximator.output
      ~Approximator.path
      ~Approximator.quantization_mode
      ~Approximator.run_eagerly
      ~Approximator.supports_masking
      ~Approximator.trainable
      ~Approximator.trainable_variables
      ~Approximator.trainable_weights
      ~Approximator.variable_dtype
      ~Approximator.variables
      ~Approximator.weights
   
   