
import sys
import matplotlib.pyplot as plt

sys.path.append('.')

import os
import numpy as np
import torch


from learning import transformer_trainer #hiprssm_dyn_trainer
from inference import transformer_inference  # hiprssm_dyn_inference

#from transformer_architecture.model_transformer_TS import  TransformerModel, GPTConfig # the older GPT model only contains decoder
from transformer_architecture.models.Transformer_Longterm import  LongTermModel, TSConfig
from transformer_architecture.ns_models.ns_Transformer import  Model, NS_TSConfig
from mlp_arcitecture.mlp_arch import MLP
from  utils.dataProcess import ts2batch_ctx_tar
from ChiSquareMeasure.ChiSquare import chisquare_test
from PearsonCorrelation_Measure.pearson import pearson_test
from MutualInformationMeasure.Mutual_information import get_mutual_information

import csv
import yaml

nn = torch.nn

def circular_shift_features(data, t):
    '''
    performe a circular shift of t to the most-left dimension of the data
    '''

    shifted_data = np.roll(data, t, axis=0)
    return shifted_data

def convert_to_numpy(data):
    if isinstance(data, np.ndarray):
        # If it's already a NumPy array, no conversion needed
        return data
    elif isinstance(data, torch.Tensor):
        # If it's a PyTorch tensor, convert it to a NumPy array
        return data.numpy()
    else:
        raise ValueError("Input data must be either a NumPy array or a PyTorch tensor")



config = """
learn:
  epochs: 5
  batch_size: 450
  lr: 0.0001
  save_model: True
  model_name: tmp
  loss: 'mse'
  load: False

transformer_arch:
  enc_layer: 2
  dec_layer: 1
  n_head: 4
  d_model: 128
  dropout: 0.1
  factor: 3
  p_hidden_layers: 2
  p_hidden_dims: [64, 64]
  

data_reader:
  context_size: 49
  pred_len: 3
"""

cfg = yaml.safe_load(config)
print(cfg)
cfg["learn"]["save_path"]= os.path.join(os.getcwd(),'saved_models',str(cfg["learn"]["model_name"])+".ckpt")

#cfg={'transformer_arch':{}}


"""Data"""
# Specify the file path
file_path = 'data/sample_data.csv'

# Reading from CSV file
with open(file_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    print("loaded...")
    # Convert the CSV data into a list
    data = np.array([row for row in csv_reader], dtype=np.float32)

plt.plot(data[:100])
plt.show()
#

# train-test split
split=0.7
T = data.shape[0]
l = int (split*T)
train_data = data[:l,:]
test_data  = data[l:,:]

# make batches of data of size [n_batch,ctx_len+target_len,num_features]
train_batched,_ = ts2batch_ctx_tar(train_data,n_batch=1000,len_ctx=cfg["data_reader"]["context_size"] , len_tar=cfg["data_reader"]["pred_len"])
test_batched,_  = ts2batch_ctx_tar(test_data,n_batch=1000,len_ctx=cfg["data_reader"]["context_size"] , len_tar=cfg["data_reader"]["pred_len"])


#### building Model######
transformer_e_layers = cfg["transformer_arch"]["enc_layer"]
transformer_d_layers = cfg["transformer_arch"]["dec_layer"]
transformer_n_head   = cfg["transformer_arch"]["n_head"]
transformer_d_model  = cfg["transformer_arch"]["d_model"]
transformer_dropout = cfg["transformer_arch"]["dropout"]
transformer_seq_len = cfg["data_reader"]["context_size"]  # --> #transformer_seq_len = cfg.transformer_arch.seq_len
transformer_factor  = cfg["transformer_arch"]["factor"]
transformer_p_hidden_layers =cfg["transformer_arch"]["p_hidden_layers"]
transformer_p_hidden_dims = cfg["transformer_arch"]["p_hidden_dims"]



print('None-stationary Transformer')
my_conf = NS_TSConfig(enc_in=train_batched.shape[-1], dec_in = train_batched.shape[-1] , pred_len=cfg["data_reader"]["pred_len"] , c_out=1 , d_model=transformer_d_model , n_heads=transformer_n_head, e_layers=transformer_e_layers , d_layers=transformer_d_layers , dropout= transformer_dropout, seq_len=transformer_seq_len ,label_len=transformer_seq_len//2 , factor=transformer_factor , p_hidden_layers=transformer_p_hidden_layers )  # ctx + target = 2*ctx

print("Transformer_config:",my_conf)

#m = LongTermModel(my_conf) # Vanila transformer
m1 = Model(my_conf) # NS_transformer

hidden_MLP_size = [360,1020, 240]  # 0.64M

m2 = MLP(input_size=cfg["data_reader"]["context_size"], hidden_sizes=hidden_MLP_size, output_size=cfg["data_reader"]["pred_len"])


save_dir = os.path.join(os.getcwd(),"saved_models")
if not(os.path.exists(save_dir)):
    save_path = os.path.join(save_dir,"model.ckpt")


parallel_net = m2

transformer_learn = transformer_trainer.Learn(parallel_net, loss=cfg["learn"]["loss"], config=cfg  )


############ statistical tests before training

input_to_predictability_measures_test_data  = circular_shift_features( convert_to_numpy(test_batched).squeeze().swapaxes(0, 1), t=cfg["data_reader"]["pred_len"])
input_to_predictability_measures_train_data  = circular_shift_features( convert_to_numpy(train_batched).squeeze().swapaxes(0, 1), t=cfg["data_reader"]["pred_len"])

print("=========================  initial Chi-square test & train  ===================================")

results0, pvl0, cnt_dep0 = chisquare_test(input_to_predictability_measures_test_data ,number_output_functions= cfg["data_reader"]["pred_len"],  bonfer=True)
results1, pvl1,cnt_dep1 = chisquare_test(input_to_predictability_measures_train_data,number_output_functions= cfg["data_reader"]["pred_len"],  bonfer=True)

print("=========================  initial MI test & train  ===================================")

############
_, SUM_MI_initial_test, init_MI_pv_test, avg_MI_initial_test_permute, SUM_MI_initial_test_permuted = get_mutual_information(input_to_predictability_measures_test_data, number_output_functions=cfg["data_reader"]["pred_len"], perm_test_flag=True, N=100)
print('SUM_MI_initial_test:', SUM_MI_initial_test)
print("SUM_MI_initial_test_permuted", SUM_MI_initial_test_permuted)
print("initial MI is less than", init_MI_pv_test ,"% of the MI in a random permutations")


_, SUM_MI_initial_train, init_MI_pv_train, avg_MI_initial_train_permute, SUM_MI_initial_train_permuted = get_mutual_information(input_to_predictability_measures_train_data, number_output_functions=cfg["data_reader"]["pred_len"], perm_test_flag=True, N=100)
print('SUM_MI_initial_test:', SUM_MI_initial_test)
print("SUM_MI_initial_test_permuted", SUM_MI_initial_test_permuted)
print("initial MI is less than", init_MI_pv_test ,"% of the MI in a random permutations")
############

print("=========================  initial pearson_test  ===================================")
sum_r_test_init = pearson_test(input_to_predictability_measures_test_data,number_output_functions=cfg["data_reader"]["pred_len"])
sum_r_train_init = pearson_test(input_to_predictability_measures_train_data,number_output_functions=cfg["data_reader"]["pred_len"])

############ statistical tests done
print("#################################### statistical tests done ####################################")



#### training the model
if cfg["learn"]["load"] == False:
    #### Train the Model
    transformer_learn.train(train_batched, cfg["learn"]["epochs"], cfg["learn"]["batch_size"], test_batched)

parallel_net.load_state_dict(torch.load(cfg["learn"]["save_path"]))
print('>>>>>>>>>>Loaded The Best Model From Local Folder<<<<<<<<<<<<<<<<<<<')


#inference


transformer_infer = transformer_inference.Infer(parallel_net, config=cfg)

#k = cfg.data_reader.context_size #=context_size=75


print(" Test started......")
result={}
# the last _ is residual list
pred_mean, _ , gt_multi, observed_part_te, residual_target_te = transformer_infer.predict_mbrl(test_batched, batch_size=cfg["learn"]["batch_size"] )  # returns normalized predicted packets
print("observed_part_test.shape:", observed_part_te.shape)
print("residual_target_test.shape:", residual_target_te.shape)

residual_ctx_and_tar_te = torch.cat([observed_part_te, residual_target_te], dim=1)
print("residual_ctx_and_tar_te:", residual_ctx_and_tar_te.shape)




pred_mean_tr, _, gt_multi_tr, observed_part_tr, residual_target_tr =  transformer_infer.predict_mbrl(train_batched,batch_size=cfg["learn"]["batch_size"])  # returns normalized predicted packets
residual_ctx_and_tar_tr = torch.cat([observed_part_tr, residual_target_tr], dim=1)
print("residual_ctx_and_tar_tr.shape:", residual_ctx_and_tar_tr.shape)
print("check and plot the stuff")


input_to_predictability_measures_test_data  = circular_shift_features( convert_to_numpy(residual_ctx_and_tar_te).squeeze().swapaxes(0, 1), t=cfg["data_reader"]["pred_len"])
input_to_predictability_measures_train_data  = circular_shift_features( convert_to_numpy(residual_ctx_and_tar_tr).squeeze().swapaxes(0, 1), t=cfg["data_reader"]["pred_len"])


print(" ======================chi-square on residual_test:=============================")
results2, pvl2 ,cnt_dep2 = chisquare_test(input_to_predictability_measures_test_data,number_output_functions= cfg["data_reader"]["pred_len"],  bonfer=True)

results3, pvl3, cnt_dep3 = chisquare_test(input_to_predictability_measures_train_data,number_output_functions= cfg["data_reader"]["pred_len"], bonfer=True)


print("=========================MI and permutation on residual_test ============================")


_, sum_res_test ,  res_MI_pv_test,   avg_MI_res_test_permute, sum_res_test_permuted = get_mutual_information(input_to_predictability_measures_test_data, number_output_functions=cfg["data_reader"]["pred_len"], perm_test_flag=True, N=100)
_, sum_res_train , res_MI_pv_train, avg_MI_res_train_permute, sum_res_train_permuted = get_mutual_information(input_to_predictability_measures_train_data, number_output_functions=cfg["data_reader"]["pred_len"], perm_test_flag=True, N=100)

print("==================================== Pearson on test_res ===================================")
#input_to_pearson=circular_shift_features(residual_ctx_and_tar_te.squeeze().swapaxes(0, 1),t=cfg["learn"]["pred_len"])
sum_r_test_res = pearson_test(input_to_predictability_measures_test_data,number_output_functions=cfg["data_reader"]["pred_len"])
sum_r_train_res = pearson_test(input_to_predictability_measures_train_data,number_output_functions=cfg["data_reader"]["pred_len"])




