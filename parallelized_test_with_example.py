
import os
import time

import numpy as np
import torch


from learning import transformer_trainer #hiprssm_dyn_trainer
from inference import transformer_inference  # hiprssm_dyn_inference

#from transformer_architecture.model_transformer_TS import  TransformerModel, GPTConfig # the older GPT model only contains decoder
from transformer_architecture.models.Transformer_Longterm import  LongTermModel, TSConfig
from transformer_architecture.ns_models.ns_Transformer import  Model, NS_TSConfig
from mlp_arcitecture.mlp_arch import MLP
from  utils.dataProcess import ts2batch_ctx_tar

from ChiSquareMeasure.parallelized_chisquare import run_parallel_chisquare_test
from ChiSquareMeasure.ChiSquare import chisquare_test



from PearsonCorrelationMeasure.parallelized_Pearson import parallel_perasonr
from PearsonCorrelationMeasure.pearson import pearson_test



from MutualInformationMeasure.parallelized_MI import get_parallel_MI
from MutualInformationMeasure.Mutual_information import get_mutual_information

from utils.synthetic_data_gen import sin_gen , white_noise





import csv
import yaml
import sys
import random

nn = torch.nn
sys.path.append('.')
import matplotlib.pyplot as plt



torch.manual_seed(2)
random.seed(2)
np.random.seed(2)

def convert_to_numpy(data):
    if isinstance(data, np.ndarray):
        # If it's already a NumPy array, no conversion needed
        return data
    elif isinstance(data, torch.Tensor):
        # If it's a PyTorch tensor, convert it to a NumPy array
        return data.numpy()
    else:
        return data
        #raise ValueError("Input data must be either a NumPy array or a PyTorch tensor")

def circular_shift_features(data, t):
    '''
    performe a circular shift of t to the most-left dimension of the data
    '''

    shifted_data = np.roll(data, t, axis=0)
    return shifted_data



if __name__ == "__main__":

    config = """
    learn:
      epochs: 10
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
      context_size: 50
      pred_len: 1
      n_features: 1
    """

    cfg = yaml.safe_load(config)
    print(cfg)
    cfg["learn"]["save_path"] = os.path.join(os.getcwd(), 'saved_models', str(cfg["learn"]["model_name"]) + ".ckpt")

    # cfg={'transformer_arch':{}}

    """Data"""
    # # Specify the file path
    # file_path = 'data/sample_data.csv'
    #
    # # Reading from CSV file
    # with open(file_path, 'r') as csvfile:
    #     csv_reader = csv.reader(csvfile)
    #     print("loaded...")
    #     # Convert the CSV data into a list
    #     data = np.array([row for row in csv_reader], dtype=np.float32)
    #
    #
    #
    # plt.plot(data[:100])
    # plt.show()
    # #
    #
    # # train-test split
    # split = 0.7
    # T = data.shape[0]
    # l = int(split * T)
    # train_data = data[:l, :]
    # test_data = data[l:, :]
    #
    # # make batches of data of size [n_batch,ctx_len+target_len,num_features]
    # train_batched, _ = ts2batch_ctx_tar(train_data, n_batch=1000, len_ctx=cfg["data_reader"]["context_size"],
    #                                     len_tar=cfg["data_reader"]["pred_len"])
    # test_batched, _ = ts2batch_ctx_tar(test_data, n_batch=1000, len_ctx=cfg["data_reader"]["context_size"],
    #                                    len_tar=cfg["data_reader"]["pred_len"])
    #
    # print("train_batched.shape:", train_batched.shape)

    print("Testing Multi-Process")
    ctx_len = cfg["data_reader"]["pred_len"]
    tar_len = cfg["data_reader"]["context_size"]
    n_features = cfg["data_reader"]["n_features"]
    B = 10000
    N = 10
    num_ijn_cpus =None


    # Parameters for get_parallel_mutual_information function
    number_output_functions = tar_len * n_features
    perm_test_flag = True


    noise    = white_noise(2*B,(ctx_len+tar_len)*n_features)
    clean_signal = sin_gen(2*B,(ctx_len+tar_len)*n_features)
    data = 0.2 * noise +  clean_signal # a timeseries of shape [B,70,1]
    print("raw_data.shape:",data.shape)

    plt.plot(data[:100])
    plt.show()
    #

    # train-test split
    split = 0.5
    T = data.shape[0]
    l = int(split * T)
    train_data = data[:l]
    test_data = data[l:]

    # make batches of data of size [n_batch,ctx_len+target_len,num_features]
    train_batched, _ = ts2batch_ctx_tar(train_data, n_batch=B, len_ctx=cfg["data_reader"]["context_size"],
                                        len_tar=cfg["data_reader"]["pred_len"])
    test_batched, _ = ts2batch_ctx_tar(test_data, n_batch=B, len_ctx=cfg["data_reader"]["context_size"],
                                       len_tar=cfg["data_reader"]["pred_len"])

    print("train_batched.shape:", train_batched.shape)
    print("test_batched.shape:", test_batched.shape)




    #### building Model######
    transformer_e_layers = cfg["transformer_arch"]["enc_layer"]
    transformer_d_layers = cfg["transformer_arch"]["dec_layer"]
    transformer_n_head = cfg["transformer_arch"]["n_head"]
    transformer_d_model = cfg["transformer_arch"]["d_model"]
    transformer_dropout = cfg["transformer_arch"]["dropout"]
    transformer_seq_len = cfg["data_reader"]["context_size"]  # --> #transformer_seq_len = cfg.transformer_arch.seq_len
    transformer_factor = cfg["transformer_arch"]["factor"]
    transformer_p_hidden_layers = cfg["transformer_arch"]["p_hidden_layers"]
    transformer_p_hidden_dims = cfg["transformer_arch"]["p_hidden_dims"]

    print('None-stationary Transformer')
    my_conf = NS_TSConfig(enc_in=train_batched.shape[-1], dec_in=train_batched.shape[-1],
                          pred_len=cfg["data_reader"]["pred_len"], c_out=1, d_model=transformer_d_model,
                          n_heads=transformer_n_head, e_layers=transformer_e_layers, d_layers=transformer_d_layers,
                          dropout=transformer_dropout, seq_len=transformer_seq_len, label_len=transformer_seq_len // 2,
                          factor=transformer_factor,
                          p_hidden_layers=transformer_p_hidden_layers)  # ctx + target = 2*ctx

    print("Transformer_config:", my_conf)

    # m = LongTermModel(my_conf) # Vanila transformer
    m1 = Model(my_conf)  # NS_transformer

    hidden_MLP_size = [360, 1020]  #

    m2 = MLP(input_size=cfg["data_reader"]["context_size"], hidden_sizes=hidden_MLP_size,
             output_size=cfg["data_reader"]["pred_len"])

    save_dir = os.path.join(os.getcwd(), "saved_models")
    if not (os.path.exists(save_dir)):
        save_path = os.path.join(save_dir, "model.ckpt")

    parallel_net = m2

    transformer_learn = transformer_trainer.Learn(parallel_net, loss=cfg["learn"]["loss"], config=cfg)

    ############ statistical tests before training

    input_to_predictability_measures_test_data = circular_shift_features(
        convert_to_numpy(test_batched).squeeze().swapaxes(0, 1), t=cfg["data_reader"]["pred_len"])
    input_to_predictability_measures_train_data = circular_shift_features(
        convert_to_numpy(train_batched).squeeze().swapaxes(0, 1), t=cfg["data_reader"]["pred_len"])

    print("=========================  initial Chi-square test & train  ===================================")

    start_chisq_ser = time.time()
    results0, pvl0, cnt_dep0 , bin5_ser,bin1_ser = chisquare_test(input_to_predictability_measures_test_data,
                                              number_output_functions=cfg["data_reader"]["pred_len"], bonfer=True)
    end_chisq_ser = time.time()
    chisq_ser_elapsed = end_chisq_ser - start_chisq_ser
    # print("ser_bin_less_than_5:",bin5_ser)
    # print("ser_bin_less_than_1:", bin1_ser)
    # results1, pvl1, cnt_dep1 = chisquare_test(input_to_predictability_measures_train_data,
    #                                           number_output_functions=cfg["data_reader"]["pred_len"], bonfer=True)


    start_chisq_par=time.time()
    dep_list_te, pval_te, bin5_te,bin1_te = run_parallel_chisquare_test(input_to_predictability_measures_test_data,number_output_functions= cfg["data_reader"]["pred_len"], bonfer=True)
    end_chisq_par = time.time()
    chisq_par_elapsed = end_chisq_par - start_chisq_par
    # print("dep_pairs_test (i,j):",dep_list_te)
    # print("(pvl_test,i,j):",pval_te)
    # print("par_bin_less_than_5:",bin5_te)
    # print("par_bin_less_than_1:", bin1_te)
    #
    dep_list_tr, pval_tr, bin5_tr, bin1_tr = run_parallel_chisquare_test(input_to_predictability_measures_train_data,number_output_functions=cfg["data_reader"]["pred_len"], bonfer=True)
    # print("dep_pairs_train (i,j):", dep_list_tr)
    # print("(pvl_train,i,j):", pval_tr)
    # print("bin_less_than_5:", bin5_tr)
    # print("bin_less_than_1:", bin1_tr)

    print("=========================  initial MI test & train  ===================================")

    ############
    # _, SUM_MI_initial_test, init_MI_pv_test, avg_MI_initial_test_permute, SUM_MI_initial_test_permuted = get_mutual_information(
    #     input_to_predictability_measures_test_data, number_output_functions=cfg["data_reader"]["pred_len"],
    #     perm_test_flag=True, N=100)
    # print('SUM_MI_initial_test:', SUM_MI_initial_test)
    # print("SUM_MI_initial_test_permuted", SUM_MI_initial_test_permuted)
    # print("initial MI is less than", init_MI_pv_test, "% of the MI in a random permutations")
    #


    # print('SUM_MI_initial_test:', SUM_MI_initial_test)
    # print("SUM_MI_initial_test_permuted", SUM_MI_initial_test_permuted)
    # print("initial MI is less than", init_MI_pv_test, "% of the MI in a random permutations")


    # _, actual_MI_te, pval_MI_te, _, perm_list_te = get_parallel_MI(input_to_predictability_measures_test_data, number_output_functions=cfg["data_reader"]["pred_len"], perm_test_flag=True, N=N ,num_cpus=None)



    start_parallel_mi = time.time()
    _, actual_MI_tr, pval_MI_tr, _, perm_list_tr = get_parallel_MI(input_to_predictability_measures_train_data, number_output_functions=cfg["data_reader"]["pred_len"], perm_test_flag=True, N=N ,num_cpus=None)
    end_parallel_mi = time.time()


    start_serial_mi=time.time()
    _, SUM_MI_initial_train, init_MI_pv_train, avg_MI_initial_train_permute, SUM_MI_initial_train_permuted = get_mutual_information(
        input_to_predictability_measures_train_data, number_output_functions=cfg["data_reader"]["pred_len"],
        perm_test_flag=True, N=N)
    end_serial_mi=time.time()

    ############

    print("=========================  initial pearson_test  ===================================")
    # sum_r_test_init = pearson_test(input_to_predictability_measures_test_data,
    #                                number_output_functions=cfg["data_reader"]["pred_len"])
    # sum_r_train_init = pearson_test(input_to_predictability_measures_train_data,
    #                                 number_output_functions=cfg["data_reader"]["pred_len"])
    sum_r_te = parallel_perasonr(input_to_predictability_measures_test_data,number_output_functions=cfg["data_reader"]["pred_len"])
    sum_r_tr = parallel_perasonr(input_to_predictability_measures_train_data,number_output_functions=cfg["data_reader"]["pred_len"])
    #

    ############ statistical tests done
    print("#################################### statistical tests done ####################################")

    #### training the model
    if cfg["learn"]["load"] == False:
        #### Train the Model
        transformer_learn.train(train_batched, cfg["learn"]["epochs"], cfg["learn"]["batch_size"], test_batched)

    parallel_net.load_state_dict(torch.load(cfg["learn"]["save_path"]))
    print('>>>>>>>>>>Loaded The Best Model From Local Folder<<<<<<<<<<<<<<<<<<<')

    # inference

    transformer_infer = transformer_inference.Infer(parallel_net, config=cfg)

    # k = cfg.data_reader.context_size #=context_size=75

    print(" Test started......")
    result = {}
    # the last _ is residual list
    pred_mean, _, gt_multi, observed_part_te, residual_target_te = transformer_infer.predict_mbrl(test_batched,
                                                                                                  batch_size=
                                                                                                  cfg["learn"][
                                                                                                      "batch_size"])  # returns normalized predicted packets
    # print("observed_part_test.shape:", observed_part_te.shape)
    # print("residual_target_test.shape:", residual_target_te.shape)

    residual_ctx_and_tar_te = torch.cat([observed_part_te, residual_target_te], dim=1)
    # print("residual_ctx_and_tar_te:", residual_ctx_and_tar_te.shape)

    pred_mean_tr, _, gt_multi_tr, observed_part_tr, residual_target_tr = transformer_infer.predict_mbrl(train_batched,
                                                                                                        batch_size=
                                                                                                        cfg["learn"][
                                                                                                            "batch_size"])  # returns normalized predicted packets
    residual_ctx_and_tar_tr = torch.cat([observed_part_tr, residual_target_tr], dim=1)
    # print("residual_ctx_and_tar_tr.shape:", residual_ctx_and_tar_tr.shape)
    # print("check and plot the stuff")

    input_res_to_predictability_measures_test_data = circular_shift_features(
        convert_to_numpy(residual_ctx_and_tar_te).squeeze().swapaxes(0, 1), t=cfg["data_reader"]["pred_len"])
    input_res_to_predictability_measures_train_data = circular_shift_features(
        convert_to_numpy(residual_ctx_and_tar_tr).squeeze().swapaxes(0, 1), t=cfg["data_reader"]["pred_len"])

    print(" ======================chi-square on residual_test:=============================")

    dep_list_res_te, pval_res_te, bin1_res_te,bin5_res_te = run_parallel_chisquare_test(input_res_to_predictability_measures_test_data,number_output_functions= cfg["data_reader"]["pred_len"], bonfer=True)

    # print("dep_pairs_test (i,j):",dep_list_te)
    # print("(pvl_test,i,j):",pval_te)
    # print("bin_less_than_5:",bin5_te)
    # print("bin_less_than_1:", bin1_te)

    dep_list_res_tr, pval_res_tr, bin5_res_tr, bin1_res_tr = run_parallel_chisquare_test(input_res_to_predictability_measures_train_data,number_output_functions=cfg["data_reader"]["pred_len"], bonfer=True)
    # print("dep_pairs_train (i,j):", dep_list_tr)
    # print("(pvl_train,i,j):", pval_tr)
    # print("bin_less_than_5:", bin5_tr)
    # print("bin_less_than_1:", bin1_tr)

    print("=========================MI and permutation on residual_test ============================")

    _, actual_MI_res_te, pval_MI_res_te, _, perm_list_res_te = get_parallel_MI(input_res_to_predictability_measures_test_data,
                                                                   number_output_functions=cfg["data_reader"][
                                                                       "pred_len"], perm_test_flag=True, N=N,
                                                                   num_cpus=None)

    print("res_MI_test:", actual_MI_res_te)
    _, actual_MI_res_tr, pval_MI_res_tr, _, perm_list_res_tr = get_parallel_MI(input_res_to_predictability_measures_train_data,
                                                                   number_output_functions=cfg["data_reader"][
                                                                       "pred_len"], perm_test_flag=True, N=N,
                                                                   num_cpus=None)

    print("res_MI_train:",actual_MI_res_tr)
    print("==================================== Pearson on test_res ===================================")
    # sum_r_test_res = pearson_test(input_to_predictability_measures_test_data,
    #                               number_output_functions=cfg["data_reader"]["pred_len"])
    # sum_r_train_res = pearson_test(input_to_predictability_measures_train_data,
    #                                number_output_functions=cfg["data_reader"]["pred_len"])
    sum_r_res_te = parallel_perasonr(input_res_to_predictability_measures_test_data,number_output_functions=cfg["data_reader"]["pred_len"])
    sum_r_res_tr = parallel_perasonr(input_res_to_predictability_measures_train_data,number_output_functions=cfg["data_reader"]["pred_len"])
    print("pearson_correl_res_test:",sum_r_res_te)


    print("==================================== statistical test on residuals done ===================================")

    serial_mi_time = end_serial_mi - start_serial_mi

    parallel_mi_time = end_parallel_mi - start_parallel_mi

    print("Serial_MI_Took:",serial_mi_time)
    print("Parallel_MI_Took:",parallel_mi_time)
    print("speed_gain_MI:",serial_mi_time/parallel_mi_time)

    # print("Serial_chisq_Took:",chisq_ser_elapsed)
    # print("Parallel_chisq_Took:",chisq_par_elapsed)
    # print("speed_gain_chisq:",chisq_ser_elapsed/chisq_par_elapsed)
    #
    #

