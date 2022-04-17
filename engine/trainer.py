import torch
from torch.utils.data import DataLoader
import utils
# import wandb
import numpy as np
import utils
from sklearn.metrics import top_k_accuracy_score, average_precision_score, accuracy_score
# wandb.login()
import pickle

class Trainer:
  def __init__(self, cfg):
      self.cfg = cfg
      self.device = torch.device('cuda:{}'.format(cfg.gpu) if torch.cuda.is_available() else 'cpu')
      print(self.device)
      # if not cfg.test:
      self.logger = utils.Logger(cfg.save_dir, cfg)

  def fit(self, model, dataset_train, dataset_val):
      dataloader_train = DataLoader(dataset_train, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=utils.collate_fn)
      dataloader_val = DataLoader(dataset_val, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=utils.collate_fn)

      optimizer = model.get_optimizer()
      scheduler = model.get_scheduler(optimizer)

      model = model.to(self.device)

      for epoch in range(self.cfg.num_epochs):
          # train
          model.train()
          for i, batch in enumerate(dataloader_train):
              batch = batch.to(self.device)
              loss, stats_step, _ = model(batch)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              self.logger.update(stats_step, 'train')


          # lr decay
          if scheduler is not None:
              scheduler.step()

          # val
          model.eval()
          # all_preds = []
          # all_target = []
          with torch.no_grad():
              for i, batch in enumerate(dataloader_val):
                  batch = batch.to(self.device)
                  loss, stats_step, preds = model(batch)
                  self.logger.update(stats_step, 'val')
          if epoch % 10 == 0:
              if self.cfg.sub_activity:
                  torch.save(model.state_dict(), "sact_epoch" +str(epoch)+".pkl")
              else:
                  torch.save(model.state_dict(), "act_epoch" + str(epoch) + ".pkl")
          print("model saved")

          stats_epoch = {'lr': optimizer.param_groups[0]['lr']}
          self.logger.summarize(epoch, stats=stats_epoch)

  def test(self, model, dataset_test):
      dataloader_test = DataLoader(dataset_test, batch_size=self.cfg.batch_size, shuffle=False,
                                   collate_fn=utils.collate_fn)
      all_preds = []
      all_target = []
      model = model.to(self.device)
      softmax = {}
      model.eval()
      with torch.no_grad():
          for i, batch in enumerate(dataloader_test):
              batch = batch.to(self.device)
              loss, stats_step, preds = model(batch)
              preds = torch.nn.functional.softmax(preds, dim=1)
              all_preds.extend(preds.cpu())
              if self.cfg.sub_activity:
                  all_target.extend(batch.sact_cids.cpu())
              else:
                  all_target.extend(batch.act_cid.cpu())


              # for i in range(len(batch.sact_id)):
              #     sact_id = batch.sact_id[i]
              #
              #     softmax[sact_id.item()] = preds[i]
              for i in range(len(batch.act_id)):
                  act_id = batch.act_id[i]
                  softmax[act_id] = preds[i]
                  print(act_id)

      # torch.save(softmax, "sact_oracle.pkl")

      all_preds = np.stack(all_preds)
      all_target = np.array(all_target)
      # all_target[-1] = 46
      # all_target[-2] = 55
      # all_target[-3] = 84

      # labels = np.unique(all_target)
      # all_preds = np.zeros((len(all_preds), 88))
      # w = 0
      # for i in range(len(all_preds)):
      #     if i not in all_preds:
      #         labels[w,:] = all_preds[i,:]

      acc_1 = top_k_accuracy_score(all_target, all_preds, k=1)
      acc_5 = top_k_accuracy_score(all_target, all_preds, k=5)

      print("acc@1:", acc_1)
      print("acc@5:", acc_5)


  #
  # def test(self, model, dataset_test):
  #     dataloader_test = DataLoader(dataset_test, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=utils.collate_fn)
  #     all_preds = []
  #     all_target = []
  #     model = model.to(self.device)
  #     with torch.no_grad():
  #         for i, batch in enumerate(dataloader_test):
  #             batch = batch.to(self.device)
  #             loss, stats_step, preds = model(batch)
  #             self.logger.update(stats_step, 'val')
  #
  #    stats_epoch = {'lr': 0.1}
  #    self.logger.summarize(1, stats=stats_epoch)
  #
  #         # for i, batch in enumerate(dataloader_test):
  #         #     batch = batch.to(self.device)
  #         #     _, _, preds = model(batch)
  #         #
  #         #     all_preds.extend(preds.cpu())
  #         #     if self.cfg.sub_activity:
  #         #         all_target.extend(batch.sact_cids.cpu())
  #         #     else:
  #         #         all_target.extend(batch.act_cid.cpu())
  #     all_preds = np.stack(all_preds)
  #     all_target = np.array(all_target)
  #
  #
  #     acc_1 = top_k_accuracy_score(all_target, all_preds, k=1)
  #     acc_5 = top_k_accuracy_score(all_target, all_preds, k=5)
  #
  #     # # acc = accuracy_score(all_target, all_preds)
  #     # # mAP = average_precision_score(all_preds, all_target)
  #
  #     # with torch.no_grad():
  #     #     for i, batch in enumerate(dataloader_test):
  #     #         batch = batch.to(self.device)
  #     #         loss, stats_step, preds = model(batch)
  #     #         self.logger.update(stats_step, 'val')
  #
  #     print("acc@1:", acc_1)
  #     print("acc@5:", acc_5)
  #     # print("acc:", acc)
  #     # print("mAP:", mAP)
  #
  #
  #
  #
  #
  #
  #
  #
  #
