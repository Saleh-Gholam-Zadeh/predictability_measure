from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


optim = torch.optim
nn = torch.nn
class Infer:
    def __init__(self, model, config = None):

        """
        :param model: nn module for HiP-RSSM
        :param use_cuda_if_available:  if to use gpu
        """
        #print("transformer_inference.py line 19,   object is created")
        self._device = torch.device( "cpu")
        self._model = model


        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config
        self._loss = config["learn"]["loss"]
        self.pred_len     = self.c["data_reader"]["pred_len"]
        self.context_size = self.c["data_reader"]["context_size"]

    @torch.no_grad()
    def predict_mbrl(self, inner_dict: dict, batch_size: int = -1) -> Tuple[float, float]:
        """
        Predict using the model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param y_context: the label information for the context sets
        :param batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        :param multiStep: how many multiStep ahead predictions do you need. You can also do this by playing with obs_valid flag.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model = self._model.eval()
        out_mean_list = []
        gt_list = []
        observed_list =[]
        residual_list = []


        test_obs = inner_dict
        dataset = TensorDataset(torch.from_numpy(test_obs))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for batch_idx, (obs_batch) in enumerate(loader):

            obs_batch = (obs_batch[0]).to(self._device)        #[10,150,25]
            #act_batch = act_batch.to(self._device)          #[10,150,25]
            #target_batch = (target_batch).to(self._device)  #[10,150,25]
            with torch.no_grad():

                ctx_obs_batch, tar_obs_batch = obs_batch[:, :self.context_size, :].float(), obs_batch[:, self.context_size:, -1:].float()
                Y = tar_obs_batch.to(self._device)
                X_enc = ctx_obs_batch.to(self._device)

                # Forward Pass
                with torch.no_grad():
                    try:
                        pred_logits, _ = self._model(X_enc)
                    except:
                        pred_logits = self._model(X_enc)


                out_mean_list.append(pred_logits.cpu())
                gt_list.append(Y.cpu())
                residual_list.append(Y.cpu()-pred_logits.cpu())
                observed_list.append(ctx_obs_batch.cpu())

        return torch.cat(out_mean_list), np.nan, torch.cat(gt_list) , torch.cat(observed_list) , torch.cat(residual_list)
