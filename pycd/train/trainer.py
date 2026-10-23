# PYCD/pycd/train/trainer.py
import torch
from tqdm import tqdm

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, patience=5, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metric):
        """
        Returns True if training should stop.
        """
        if self.best is None:
            self.best = metric
            return False

        improve = False
        if self.mode == 'max' and metric > self.best + self.min_delta:
            improve = True
        elif self.mode == 'min' and metric < self.best - self.min_delta:
            improve = True

        if improve:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience

class Trainer:
    """
    通用训练器：负责训练、验证/测试循环；支持早停、学习率调度与模型 checkpoint。
    """
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device='cpu',
        early_stop: EarlyStopping = None,
        ckpt_path: str = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stop = early_stop
        self.ckpt_path = ckpt_path

    def train_epoch(self, dataloader, extra_inputs=None):
        if hasattr(self.model, 'on_epoch_start'): #orcdf
            self.model.on_epoch_start()

        self.model.train()
        losses = []
        for batch in tqdm(dataloader, desc='Train'):
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            if extra_inputs:
                preds = self.model(*batch[:-1], *extra_inputs)
            else:
                preds = self.model(*batch[:-1])
            loss = self.model.loss(preds, batch[-1])
            loss.backward()
            self.optimizer.step()
            if hasattr(self.model, 'apply_clipper'): #rcd & orcdf
                self.model.apply_clipper()
                
            losses.append(loss.item())
        if self.scheduler:
            try:
                self.scheduler.step(sum(losses) / len(losses))
            except TypeError:
                self.scheduler.step()
        return sum(losses) / len(losses)

    def eval_epoch(self, dataloader, metrics_fn, extra_inputs=None,extra_params=None):
        """
        在验证/测试集上评估，返回 metrics_fn 计算的结果。
        metrics_fn(trues, preds) 应返回单个指标或字典。
        """
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Eval'):
                batch = [x.to(self.device) for x in batch]
                if extra_inputs:
                    pred = self.model(*batch[:-1], *extra_inputs)
                else:
                    pred = self.model(*batch[:-1])
                if isinstance(pred, tuple):  #orcdf forward返回logits和extra_loss，取前者
                    pred = pred[0]                
                preds.extend(pred.cpu().tolist())
                trues.extend(batch[-1].cpu().tolist())
        return metrics_fn(self.model,trues, preds,extra_params)

    def fit(
        self,
        train_loader,
        val_loader,
        metrics_fn,
        epochs: int = 10,
        extra_inputs=None,
        extra_params=None,
    ):
        """
        完整的训练-验证流程，支持早停、学习率调度与模型 checkpoint。
        """
        best_metric = None
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, extra_inputs)
            val_metric = self.eval_epoch(val_loader, metrics_fn, extra_inputs,extra_params)
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Metric: {val_metric}")

            # 保存最优模型
            if self.ckpt_path is not None:
                if best_metric is None or val_metric > best_metric:
                    best_metric = val_metric
                    torch.save(self.model.state_dict(), self.ckpt_path)
                    print(f"  → Saved best model to {self.ckpt_path}")

            # 早停判断
            if self.early_stop is not None:
                if self.early_stop.step(val_metric):
                    print(f"  → Early stopping at epoch {epoch}")
                    break

        # 训练结束后，加载最优参数
        if self.ckpt_path is not None:
            self.model.load_state_dict(torch.load(self.ckpt_path))


class Trainer4DisenGCD(Trainer):
    def __init__(self, model, optimizer,scheduler=None, device='cpu', early_stop: EarlyStopping = None, ckpt_path: str = None):
        super().__init__(model, optimizer, scheduler, device, early_stop, ckpt_path)
        self.model = model
        self.optimizer = optimizer[0]
        self.optimizer2 = optimizer[1]
        self.scheduler = scheduler
        self.device = device
        self.early_stop = early_stop
        self.ckpt_path = ckpt_path
    
    def train_epoch(self, dataloader, extra_inputs=None):
        self.model.train()
        losses = []
        progress_bar = tqdm(dataloader, desc='Train')
        for batch in progress_bar:
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            if extra_inputs:
                output_1 = self.model(*batch[:-1], *extra_inputs)
            else:
                output_1 = self.model(*batch[:-1])
            output_0 = torch.ones(output_1.size()).to(self.device) - output_1
            output = torch.cat((output_0, output_1), 1)
            loss_w = self.model.loss(torch.log(output + 1e-10), batch[-1])
            loss_w.backward()
            self.optimizer.step()
                
            losses.append(loss_w.item())
            # 更新进度条描述
            progress_bar.set_description(f"Train (loss={loss_w.item():.4f})")
        
        return sum(losses) / len(losses) if losses else 0.0

    def eval_epoch(self, dataloader, metrics_fn, extra_inputs=None,extra_params=None):
        """
        在验证/测试集上评估，返回 metrics_fn 计算的结果。
        metrics_fn(trues, preds) 应返回单个指标或字典。
        """
        self.model.eval()
        preds, trues = [], []
        
        with torch.no_grad():  # 不计算梯度
            for batch in tqdm(dataloader, desc='Eval'):
                batch = [x.to(self.device) for x in batch]
                if extra_inputs:
                    output_1 = self.model(*batch[:-1], *extra_inputs)
                else:
                    output_1 = self.model(*batch[:-1])
                output_0 = torch.ones(output_1.size()).to(self.device) - output_1
                output = torch.cat((output_0, output_1), 1)
                
                # loss_a1 = self.model.loss(torch.log(output + 1e-10),batch[-1])
                # loss_a1.backward()
                # self.optimizer2.step()
                output = output_1.view(-1)
                preds.extend(output.cpu().tolist())
                trues.extend(batch[-1].cpu().tolist())
        return metrics_fn(self.model,trues, preds,extra_params)

