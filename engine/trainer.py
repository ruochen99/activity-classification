import torch
from torch.utils.data import DataLoader
import utils
# import wandb
import numpy as np
import utils
# wandb.login()

class Trainer:
  def __init__(self, cfg):
      self.cfg = cfg
      self.device = torch.device('cuda:{}'.format(cfg.gpu) if torch.cuda.is_available() else 'cpu')
      print(self.device)
      if not cfg.test:
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


          torch.save(model.state_dict(), self.cfg.model_path)
          print("model saved")

          stats_epoch = {'lr': optimizer.param_groups[0]['lr']}
          self.logger.summarize(epoch, stats=stats_epoch)

  def test(self, model, dataset_test):
      dataloader_test = DataLoader(dataset_test, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=utils.collate_fn)
      all_preds = []
      all_target = []
      model = model.to(self.device)
      with torch.no_grad():
          for i, batch in enumerate(dataloader_test):
              batch = batch.to(self.device)
              _, _, preds = model(batch)
              all_preds.append(preds.item())
              all_target.extend(batch.sact_cids.item())

      acc = utils.get_acc(np.array(all_preds), np.array(all_target))
      mAP = utils.get_mAP(np.array(all_preds), np.array(all_target))

      print("acc:", acc)
      print("mAP:", mAP)









