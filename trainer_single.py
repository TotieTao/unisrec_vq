import os
from time import time
from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.trainer import Trainer
from recbole.utils import set_color, get_gpu_usage
from model.module.Kmeans import KMeans_Pytorch,KMeans,run_kmeans_pcl
import numpy as np
class PretrainTrainer(Trainer):
    def __init__(self, config, model):
        super(PretrainTrainer, self).__init__(config, model)
        self.pretrain_epochs = self.config['pretrain_epochs']
        self.save_step = self.config['save_step']

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)

    def _trans_dataload(self, interaction):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        #using pytorch dataload to re-wrap dataset
        def sub_trans(dataset):
            dis_loader = DataLoader(dataset=dataset,
                                    batch_size=dataset.shape[0],
                                    sampler=DistributedSampler(dataset, shuffle=False))
            for data in dis_loader:
                batch_data = data

            return batch_data
        #change `interaction` datatype to a python `dict` object.  
        #for some methods, you may need transfer more data unit like the following way.  

        data_dict = {}
        for k, v in interaction.interaction.items():
            data_dict[k] = sub_trans(v)
        return data_dict

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            # interaction = self._trans_dataload(interaction)
            self.optimizer.zero_grad()
            losses = loss_func(interaction, self.centroids, epoch_idx, self.cur_step)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def pretrain(self, train_data, prekmeans_data, verbose=True, show_progress=False):

        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            if epoch_idx >= 0 and epoch_idx < self.config['kmeans_epochs']:
                    self.train_kmeans(prekmeans_data, epoch_idx, show_progress=show_progress)
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}.pth".format(
                        self.config["model"], self.config["dataset"], str(epoch_idx + 1)
                    ),
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = (
                        set_color("Saving current", "blue") + ": %s" % saved_model_file
                )
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result

    def train_kmeans(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        loss_func = loss_func or self.model.calculate_loss
        print("Preparing Clustering:")
        self.model.eval()
        kmeans_training_data = []
        short_training_data = []
        long_training_data = []

        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            n = len(kmeans_training_data) * self.config['train_batch_size']
            print("kmeans length")
            print(n)
            if n > 20000:
                break
            with torch.no_grad():
                sequence_output1 = loss_func(interaction, [], epoch_idx, self.cur_step, True)


            # average sum
            sequence_output1 = sequence_output1.view(sequence_output1.shape[0], -1)
            # sequence_output2 = sequence_output2.view(sequence_output2.shape[0], -1)
            # sequence_output3 = sequence_output3.view(sequence_output3.shape[0], -1)

            sequence_output1 = sequence_output1.detach().cpu().numpy()
            # sequence_output2 = sequence_output2.detach().cpu().numpy()
            # sequence_output3 = sequence_output3.detach().cpu().numpy()
            kmeans_training_data.append(sequence_output1)
            # short_training_data.append(sequence_output2)
            # long_training_data.append(sequence_output3)


        kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
        # short_training_data = np.concatenate(short_training_data, axis=0)
        # long_training_data = np.concatenate(long_training_data, axis=0)
        ## kmeans_training_data = torch.Tensor(kmeans_training_data).to(self._device)
        print(kmeans_training_data.shape)

        # train multiple clusters

        print("Training Clusters:")

        cluster_result = {'im2cluster':[],'centroids': [], 'density': []}
        # for num_cluster in self._args.num_cluster:
        # num_cluster = self._args.num_cluster.split(',')
        cluster_result['im2cluster'].append(torch.zeros(len(kmeans_training_data), dtype=torch.long).cuda())
        cluster_result['centroids'].append(torch.zeros(int(self.config['long_num_cluster']), self.config['adaptor_output_size']).cuda())
        cluster_result['density'].append(torch.zeros(int(self.config['long_num_cluster'])).cuda())

        # cluster_result['im2cluster'].append(torch.zeros(len(short_training_data), dtype=torch.long).cuda())
        # cluster_result['centroids'].append(torch.zeros(int(self._args.short_num_cluster), self._args.hidden_dim).cuda())
        # cluster_result['density'].append(torch.zeros(int(self._args.short_num_cluster)).cuda())

# `       if self._args.dist_rank == 0:
        # cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
        cluster_result = run_kmeans_pcl(kmeans_training_data,[int(self.config['long_num_cluster'])], self.config)  # run kmeans clustering on master node
        # short_cluster_result = run_kmeans_pcl(short_training_data,[int(self._args.short_num_cluster)], args)  # run kmeans clustering on master node

        # cluster_result['im2cluster'].append(short_cluster_result['im2cluster'][0])
        # cluster_result['centroids'].append(short_cluster_result['centroids'][0])
        # cluster_result['density'].append(short_cluster_result['density'][0])


        # print("Cluster id:{}".format(cluster_result['im2cluster'][:30]))
        # print("Cluster relust:{},\n{}".format(cluster_result['centroids'],cluster_result['density']))


        # torch.distributed.barrier()
        # broadcast clustering result
        # for k, data_list in cluster_result.items():
        #     for data_tensor in data_list:
        #         torch.distributed.broadcast(data_tensor, 0, async_op=False)
                # clean memory

        self.centroids = cluster_result['centroids']
        print(len(self.centroids))
        print("Cluster relust:{},\n{}".format(cluster_result['centroids'], cluster_result['density']))

        del kmeans_training_data
        del long_training_data
        del short_training_data, cluster_result
        import gc
        gc.collect()
        print("kmeans finish")