import torch
import time
from icdl.utils import comm
from icdl.utils.logger import Logger
from .trainer import DefaultTrainer
from ..registry import ENFINE_REGISTRY

try:
    import apex
except:
    print("没有 apex库")


@ENFINE_REGISTRY.register()
class CLSTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(CLSTrainer, self).__init__(cfg)

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        s = time.time()
        for i, (img, label) in enumerate(self.train_loader):
            img = img.cuda()
            label = label.cuda()

            self.optimer.zero_grad()
            pred = self.model(img)

            if isinstance(pred, tuple):
                loss = None
                for p in pred:
                    if loss:
                        loss += self.criterion(p, label)
                    else:
                        loss = self.criterion(p, label)
            else:
                loss = self.criterion(pred, label)

            if self.cfg.TRAIN.APEX:
                with apex.amp.scale_loss(loss, self.optimer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            total_loss += loss.item()
            avg_loss = total_loss / (i+1)

            if comm.is_main_process() and i % 10 == 0:
                Logger.log(
                    "\r [Train Epoch %d/%d] [lr %.6f] [speed %f pic/s] [avg_loss %.6f]"
                    % (
                        epoch,
                        self.cfg.SOLVER.MAX_ITER,
                        self.optimer.param_groups[0]["lr"],
                        label.shape[0] / (time.time() - s) * comm.get_world_size(),
                        avg_loss
                    ),
                    mode="console"
                )

            s = time.time()

            self.optimer.step()
            self.lr_scheduler.step()

        Logger.log("[Train Epoch {}/{}] [avg_loss {:.6f}]".format(
                epoch,
                self.cfg.SOLVER.MAX_ITER,
                avg_loss
        ))

        return 1 / avg_loss

    def val_one_epoch(self, epoch):
        if not self.val_loader:
            return

        self.model.eval()

        total_corrects = 0.0
        for img, label in self.val_loader:
            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                pred = self.model(img)

            _, preds = torch.max(pred, 1)
            total_corrects += torch.sum(preds == label.data)

        if comm.get_world_size() > 1:
            torch.distributed.reduce(total_corrects, 0, op=torch.distributed.ReduceOp.SUM)

        epoch_acc = total_corrects.cpu().item() / len(self.val_loader.dataset)

        Logger.log("[Val Epoch {}/{}] [acc {:.6f}]".format(
            epoch,
            self.cfg.SOLVER .MAX_ITER,
            epoch_acc
        ), mode="console"
        )

        return epoch_acc
