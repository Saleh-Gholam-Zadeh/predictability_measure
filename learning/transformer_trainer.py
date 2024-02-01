import os
import time as t
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.Losses import mse, mae
from utils.PositionEmbedding import PositionEmbedding as pe
optim = torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
nn = torch.nn
import torch
class Learn:

    def __init__(self, model, loss: str, config: dict = None  ):
        """
        :param model: nn module for np_dynamics
        :param loss: type of loss to train on 'nll' or 'mse' or 'mae' (added by saleh) or 'CrossEntropy' (added by saleh)
        :param imp: how much to impute
        :param use_cuda_if_available: if gpu training set to True
        """

        self._device = torch.device("cpu")
        self._loss = loss
        self._model = model
        self._pe = pe(self._device)
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config

        self._learning_rate = self.c["learn"]["lr"]
        self._save_path = self.c["learn"]["save_path"]
        self._save_floder = os.path.dirname(self._save_path)
        if not os.path.exists(self._save_floder ):
            os.makedirs(self._save_floder )

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._scheduler = ReduceLROnPlateau(self._optimizer, mode='min', patience=6, factor=0.5, verbose=True)
        self.cool_down = 0

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self.save_model = self.c["learn"]["save_model"]

        self.pred_len     = self.c["data_reader"]["pred_len"]
        self.context_size = self.c["data_reader"]["context_size"]
        #self.log_stat = self.c.learn.log_stat

    def train_step(self, train_dict: dict, batch_size: int)  -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions
        :param train_targets: training targets
        :param train_task_idx: task ids per episode
        :param batch_size: batch size for each gradient update
        :return: average loss (nll) and  average metric (rmse), execution time
        """
        mse_per_batch = []
        self._model.train()
        avg_loss = avg_metric_nll = avg_metric_mse  = avg_metric_mae = avg_metric_CrossEntropy = avg_metric_combined_kl_rmse = avg_metric_kl = 0
        t0 = t.time()

        train_obs = train_dict
        dataset = TensorDataset(torch.from_numpy(train_obs))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for batch_idx, obs in enumerate(loader):
            # Set Optimizer to Zero
            self._optimizer.zero_grad()
            obs_batch = obs[0].to(self._device)
            ctx_obs_batch , tar_obs_batch = obs_batch[:,:self.context_size,:].float(),obs_batch[:,self.context_size:,:].float()
            Y = tar_obs_batch.to(self._device)
            X_enc = ctx_obs_batch.to(self._device)                                                                               # obs: -----ctx_obs-------/-------tar_obs------
                                                                                                                                 # tar:  -----ctx_tar-------/------tar_tar-------

            try:
                #print("X_enc.shape", X_enc.shape)
                pred_logits,_ = self._model(X_enc.float())
            except:
                #print("X_enc.shape",X_enc.shape)
                pred_logits = self._model(X_enc.float())


            if self._loss == 'mae':
                loss = mae(pred_logits, Y)
            elif self._loss == 'mse':
                mseloss = nn.MSELoss()
                loss = mseloss(pred_logits, Y)

            else:
                raise NotImplementedError

            mse_per_batch.append(loss.item())

            # Backward Pass
            loss.backward()

            self._optimizer.step()

            with torch.no_grad():

                out_mean_pred = pred_logits
                metric_mse = mse( out_mean_pred,Y)
                metric_mae = mae(out_mean_pred, Y)


            avg_loss += loss.detach().cpu().numpy()
            avg_metric_mse += metric_mse.detach().cpu().numpy()
            avg_metric_mae += metric_mae.detach().cpu().numpy()



        assert len(mse_per_batch)==len(loader) , "something went wrong"
        with torch.no_grad():

            if self._loss == 'mse':
                avg_loss = np.sqrt(avg_loss /  len(mse_per_batch))

            elif self._loss == 'mae':
                avg_loss = avg_loss / (  len(mse_per_batch) )

            else:
                raise NotImplementedError


        with torch.no_grad():
            avg_metric_rmse = np.sqrt(avg_metric_mse / len(mse_per_batch))
            avg_metric_mae = avg_metric_mae / len(mse_per_batch)


        return avg_loss, avg_metric_rmse,avg_metric_mae ,t.time() - t0

    @torch.no_grad()
    def eval(self, test_dict: np.ndarray, batch_size: int = -1) -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param targets: targets to evaluate on
        :param task_idx: task index
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        val_mse_per_batch = []

        avg_loss = avg_metric_mse = avg_metric_mae = 0.0


        test_obs = test_dict
        dataset = TensorDataset(torch.from_numpy(test_obs))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


        for batch_idx, (obs_batch) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices
                obs_batch = obs_batch[0]

                ctx_obs_batch, tar_obs_batch = obs_batch[:, :self.context_size, :].float(), obs_batch[:, self.context_size:, :].float()
                Y = tar_obs_batch.to(self._device)
                X_enc = ctx_obs_batch.to(self._device)


                try:
                    pred_logits, _ = self._model(X_enc) # when with transformer we also return attention wights
                except:
                    pred_logits = self._model(X_enc)


                out_mean_pred = pred_logits

                ## Calculate Loss
                if self._loss == 'mse':
                    loss = mse( pred_logits,Y)

                elif self._loss == 'mae':
                    loss = mae(pred_logits,Y)
                else:
                    raise NotImplementedError

                val_mse_per_batch.append(loss)
                metric_mse = mse(Y, out_mean_pred)
                metric_mae = mae(Y, out_mean_pred)

                avg_loss += loss.detach().cpu().numpy()

                avg_metric_mse += metric_mse.detach().cpu().numpy()
                avg_metric_mae += metric_mae.detach().cpu().numpy()

        if self._loss == 'mse':
            avg_loss = np.sqrt(avg_loss /  len(val_mse_per_batch))

        elif self._loss == 'mae':
            avg_loss = avg_loss /  len(val_mse_per_batch)

        else:
            raise NotImplementedError
        self._scheduler.step(avg_loss)

        if self._scheduler.in_cooldown:
            print("Learning rate was reduced!")
            self.cool_down = self.cool_down+1


        avg_metric_rmse = np.sqrt(avg_metric_mse / len(val_mse_per_batch))
        avg_metric_mae =(avg_metric_mae /  len(val_mse_per_batch))


        return avg_loss, avg_metric_rmse, avg_metric_mae

    def train(self, train_obs: torch.Tensor, epochs: int, batch_size: int,val_obs: torch.Tensor = None,val_interval: int = 1,val_batch_size: int = -1) -> None:
        '''
        :param train_obs: training observations for the model (includes context and targets)
        :param train_act: training actions for the model (includes context and targets)
        :param train_targets: training targets for the model (includes context and targets)
        :param train_task_idx: task_index for different training sequence
        :param epochs: number of epochs to train on
        :param batch_size: batch_size for gradient descent
        :param val_obs: validation observations for the model (includes context and targets)
        :param val_act: validation actions for the model (includes context and targets)
        :param val_targets: validation targets for the model (includes context and targets)
        :param val_task_idx: task_index for different testing sequence
        :param val_interval: how often to perform validation
        :param val_batch_size: batch_size while performing inference
        :return:
        '''


        """ Train Loop"""
        torch.cuda.empty_cache() #### Empty Cache
        if val_batch_size == -1:
            val_batch_size = 4 * batch_size
        best_loss = np.inf


        init_lr = self._optimizer.param_groups[0]['lr']
        print("initial_learning_rate",init_lr)
        print("===========================...epoching...=================================")
        for i in range(epochs):
            print("===================================================================================")
            #print("epochs = :",i , "/",epochs)
            print(f"Epoch {i+1}: Learning Rate: {self._optimizer.param_groups[0]['lr']}")
            old_lr = self._optimizer.param_groups[0]['lr']

            #print("scheduler_cooldown_counter:", self.cool_down )
            if (self._optimizer.param_groups[0]['lr']< init_lr*1e-3):
                print("scheduler terminates the training ")
                return

            train_loss, train_metric_rmse,train_metric_mae , time_tr = self.train_step(train_obs,batch_size)
            print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.4f}, {}, {:.5f}, Took:{:.3f} seconds".format(i + 1, self._loss, train_loss, 'target_rmse:', train_metric_rmse, 'target_mae:', train_metric_mae,time_tr))


            if val_obs is not None  and np.mod(i,val_interval)  == 0:
                val_loss, val_metric_rmse ,val_metric_mae = self.eval(val_obs, batch_size=val_batch_size)

                new_lr = self._optimizer.param_groups[0]['lr']
                if new_lr!=old_lr :
                    print("Learning rate was reduced ")
                    self.cool_down=self.cool_down+1

                if val_loss < best_loss:
                    if self.save_model:
                        print('>>>>>>>Saving Best Model<<<<<<<<<<',"epoch:",i+1)
                        torch.save(self._model.state_dict(), self._save_path)

