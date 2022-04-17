import torch
from torch.utils.data import DataLoader
import utils
import numpy as np
import utils
from sklearn.metrics import top_k_accuracy_score, average_precision_score
from tqdm import tqdm


class TrainerHOI:
  def __init__(self, cfg):
      self.cfg = cfg
      self.device = torch.device('cuda:{}'.format(cfg.gpu) if torch.cuda.is_available() else 'cpu')
      print(self.device)
      # if not cfg.test:
      #   self.logger = utils.Logger(cfg.save_dir, cfg)

  def fit(self, model, dataset_train, dataset_val):
      dataloader_train = DataLoader(dataset_train, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=utils.collate_fn)
      dataloader_val = DataLoader(dataset_val, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=utils.collate_fn)

      optimizer = model.get_optimizer()
      scheduler = model.get_scheduler(optimizer)

      model = model.to(self.device)

      for epoch in range(self.cfg.num_epochs):
          print("epoch: ", epoch)
          # train
          model.train()
          train_loss = 0
          model = model.to(self.device)
          for i, batch in enumerate(tqdm(dataloader_train)):
              batch = batch.to(self.device)
              loss, _ = model(batch)
              train_loss += loss.item()
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              # self.logger.update(stats_step, 'train')
          print("train loss: ", train_loss/len(dataloader_train))
          # lr decay
          if scheduler is not None:
              scheduler.step()

          # val
          model.eval()
          val_loss = 0
          with torch.no_grad():
              for i, batch in enumerate(tqdm(dataloader_val)):
                  batch = batch.to(self.device)
                  loss, _ = model(batch)
                  val_loss += loss
                  # self.logger.update(stats_step, 'val')
          print("val loss: ", val_loss/len(dataloader_val))

          torch.save(model.state_dict(), self.cfg.model_path)
          print("model saved")

          stats_epoch = {'lr': optimizer.param_groups[0]['lr']}
          # self.logger.summarize(epoch, stats=stats_epoch)

  def test(self, model, dataset_test, k=5):
      dataloader_test = DataLoader(dataset_test, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=utils.collate_fn)
      total_rel = 0
      total_ta = 0
      total_att = 0
      total_ia = 0

      pred_rel_5 = 0
      pred_ta_5 = 0
      pred_att_5 = 0
      pred_ia_5 = 0

      pred_rel_10 = 0
      pred_ta_10 = 0
      pred_att_10 = 0
      pred_ia_10 = 0

      model = model.to(self.device)
      with torch.no_grad():
          for i, batch in enumerate(tqdm(dataloader_test)):
              batch = batch.to(self.device)
              loss, preds = model(batch)

              total_att += (batch.node_att == 1).sum().item()
              total_ia += (batch.node_ia == 1).sum().item()
              total_rel += (batch.edge_rel == 1).sum().item()
              total_ta += (batch.edge_ta == 1).sum().item()

              k = 50

              v, i = torch.topk(preds['att'].flatten(), k)
              indices = np.array(np.unravel_index(i.cpu().numpy(), preds['att'].shape)).T

              for ind in indices:
                  if batch.node_att[ind[0]][ind[1]] == 1:
                      pred_att_5 += 1

              v, i = torch.topk(preds['ia'].flatten(), k)
              indices = np.array(np.unravel_index(i.cpu().numpy(), preds['ia'].shape)).T

              for ind in indices:
                  if batch.node_ia[ind[0]][ind[1]] == 1:
                      pred_ia_5 += 1

              v, i = torch.topk(preds['rel'].flatten(), k)
              indices = np.array(np.unravel_index(i.cpu().numpy(), preds['rel'].shape)).T

              for ind in indices:
                  if batch.edge_rel[ind[0]][ind[1]] == 1:
                      pred_rel_5 += 1

              v, i = torch.topk(preds['ta'].flatten(), k)
              indices = np.array(np.unravel_index(i.cpu().numpy(), preds['ta'].shape)).T

              for ind in indices:
                  if batch.edge_ta[ind[0]][ind[1]] == 1:
                      pred_ta_5 += 1

              k = 100

              v, i = torch.topk(preds['att'].flatten(), k)
              indices = np.array(np.unravel_index(i.cpu().numpy(), preds['att'].shape)).T

              for ind in indices:
                  if batch.node_att[ind[0]][ind[1]] == 1:
                      pred_att_10 += 1

              v, i = torch.topk(preds['ia'].flatten(), k)
              indices = np.array(np.unravel_index(i.cpu().numpy(), preds['ia'].shape)).T

              for ind in indices:
                  if batch.node_ia[ind[0]][ind[1]] == 1:
                      pred_ia_10 += 1

              v, i = torch.topk(preds['rel'].flatten(), k)
              indices = np.array(np.unravel_index(i.cpu().numpy(), preds['rel'].shape)).T

              for ind in indices:
                  if batch.edge_rel[ind[0]][ind[1]] == 1:
                      pred_rel_10 += 1

              v, i = torch.topk(preds['ta'].flatten(), k)
              indices = np.array(np.unravel_index(i.cpu().numpy(), preds['ta'].shape)).T

              for ind in indices:
                  if batch.edge_ta[ind[0]][ind[1]] == 1:
                      pred_ta_10 += 1

      print("r@5@att", pred_att_5 / total_att)
      print("r@5@ia", pred_ia_5 / total_ia)
      print("r@5@rel", pred_rel_5/total_rel)
      print("r@5@ta", pred_ta_5/total_ta)

      print('r@50', (pred_att_5+pred_ia_5+pred_rel_5+pred_ta_5)/(total_att+total_ia+total_rel+total_ta))

      print("r@10@att", pred_att_10 / total_att)
      print("r@10@ia", pred_ia_10 / total_ia)
      print("r@10@rel", pred_rel_10 / total_rel)
      print("r@10@ta", pred_ta_10 / total_ta)

      print('r@100', (pred_att_10 + pred_ia_10 + pred_rel_10 + pred_ta_10) / (total_att + total_ia + total_rel + total_ta))









