# import ray
# from ray.tune.logger import pretty_print
# from ray.rllib.algorithms.algorithm import Algorithm

# from ray.rllib.algorithms.ppo import (
#     PPOConfig,
#     PPOTF1Policy,
#     PPOTF2Policy,
#     PPOTorchPolicy,
# )
# #test branche opti
# from time import sleep
# from gymnasium import spaces
# from ray import tune    
# from ray.air import CheckpointConfig
# import subprocess
# import time
# import torch
# import numpy as np
# from ray.rllib.policy.policy import Policy
# import ray.rllib.policy.torch_policy_v2

# path = "/home/ia/Desktop/DRL_platform/DRL_platform_ENSTA_v1/Scenarios/UUV_Mono_Agent_TSP/models/3x3_3_100/PPO_UUVMonoAgentTSPEnv_de260_00000_0_2024-03-13_08-39-47/checkpoint_000026"

# loaded_algo = Algorithm.from_checkpoint(path)

# print("loaded_model : ",loaded_algo)
# #loaded_algo.export_model(export_dir="/home/ia/Desktop/DRL_platform/DRL_platform_ENSTA_v1/Scenarios/UUV_Mono_Agent_TSP/models/mymodel.h5",export_formats='h5')

# loaded_algo.export_policy_model(export_dir="/home/ia/Desktop/DRL_platform/DRL_platform_ENSTA_v1/Scenarios/UUV_Mono_Agent_TSP/models/3x3_3_100/")
##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import onnx
from onnx2keras import onnx_to_keras

import tensorflow as tf
from tensorflow import keras

# Load ONNX model
onnx_model = onnx.load('/home/ia/Desktop/DRL_platform/DRL_platform_ENSTA_v1/Scenarios/UUV_Mono_Agent_TSP/models/3x3_3_100/model.onnx')
# Ci-dessous Un moyen de retrouver le ['input'] dont on a besoin plus bas ?
# https://github.com/onnx/onnx/issues/2657
# https://onnx.ai/onnx/api/reference.html
for node in onnx_model.graph.input :
    print('Input node.name: ', node.name)
# Call the converter (input will be equal to the input_names parameter that you defined during exporting)


k_model = onnx_to_keras(onnx_model, ['state_ins'])
k_model.summary()

# # https://www.tensorflow.org/tutorials/keras/save_and_load?hl=fr
# # https://www.tensorflow.org/guide/keras/serialization_and_saving
# # Save the entire model to a HDF5 file.
# # The '.h5' extension indicates that the model should be saved to HDF5.
# k_model.save('my_model.h5')

# # si k_model.save() ne fonctionne pas, tester :
# # tf.keras.saving.save_model(k_model, 'my_model.h5', overwrite=True, save_format=None, **kwargs)
# # https://deeplizard.com/learn/video/7n1SpeudvAE
# # on peut aussi envisager de sauver que les poids, ce qui a l'air d'être
# # attendu ici : 
# #https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.models.modelv2.ModelV2.import_from_h5.html

# k_model.save_weights('my_model_weights.h5')