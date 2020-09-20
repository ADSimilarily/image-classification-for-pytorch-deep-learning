import torch
import time
from tqdm import tqdm
from .trainer import DefaultTrainer
from ..registry import ENFINE_REGISTRY
from icdl.datasets.samplers.hard_sampler import HardSampler
from icdl.utils import comm
from icdl.utils.logger import Logger

try:
    import apex
except:
    print("没有 apex库")


@ENFINE_REGISTRY.register()
class METRICTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(METRICTrainer, self).__init__(cfg)
        self.batch_sampler = self.train_loader.batch_sampler

    def train_one_epoch(self, epoch):
        self.model.train()

        running_loss = 0.0
        running_low_acc = 0.0
        running_medium_acc = 0.0
        running_high_acc = 0.0
        count_total = 0.0
        epoch_loss = 0.0
        epoch_low_acc = 0.0
        epoch_medium_acc = 0.0
        epoch_high_acc = 0.0
        # Iterate over data.

        s = time.time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            self.optimer.zero_grad()

            count_total += labels.shape[0]
            # forward
            outputs = self.model(inputs)

            loss, low_acc, medium_acc, high_acc = self.criterion(outputs, labels)

            if self.cfg.TRAIN.APEX:
                with apex.amp.scale_loss(loss, self.optimer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimer.step()

            running_loss += loss.item() * labels.shape[0]
            running_low_acc += low_acc * labels.shape[0]
            running_medium_acc += medium_acc * labels.shape[0]
            running_high_acc += high_acc * labels.shape[0]

            if comm.get_world_size() > 1:
                running_loss = comm.reduce_cpu_real_number(running_loss)
                running_low_acc = comm.reduce_cpu_real_number(running_low_acc)
                running_medium_acc = comm.reduce_cpu_real_number(running_medium_acc)
                running_high_acc = comm.reduce_cpu_real_number(running_high_acc)
                count_total = comm.reduce_cpu_real_number(count_total)


            if comm.is_main_process():
                epoch_loss = running_loss / count_total
                epoch_low_acc = running_low_acc / count_total
                epoch_medium_acc = running_medium_acc / count_total
                epoch_high_acc = running_high_acc / count_total

                Logger.log(
                    "\r [Train Epoch %d/%d] [speed %f pic/s] [lr %f] [avg_loss %.6f] [low_acc %.5f] [medium_acc %.5f] [high_acc %.5f]"
                    % (
                        epoch,
                        self.cfg.SOLVER.MAX_ITER,

                        labels.shape[0]/(time.time() - s) * comm.get_world_size(),

                        self.optimer.param_groups[0]["lr"],
                        epoch_loss,
                        epoch_low_acc,
                        epoch_medium_acc,
                        epoch_high_acc
                    ),
                    mode="console"
                )

            s = time.time()

            if self.lr_scheduler:
                self.lr_scheduler.step()

        Logger.log("[Train Epoch {}/{}] [lr {:.6f}] [loss {:.5f}] [low_acc {:.5f}] [medium_acc {:.5f}] [high_acc {:.5f}]".format(
                epoch,
                self.cfg.SOLVER.MAX_ITER,
                self.optimer.param_groups[0]["lr"],
                epoch_loss,
                epoch_low_acc,
                epoch_medium_acc,
                epoch_high_acc
        ))

        return epoch_high_acc

    def hard_sampler(self, epoch):
        HardSampler.clearSamplerDict()

        template = self.criterion.getApproximateTemplate()
        template = sorted(template.items(), key = lambda d:d[0])

        label_list = []
        feature_list = []
        for key, value in template:
            label_list.append(self.train_loader.dataset.classes[key])
            feature_list.append(value)

        feature_value = torch.cat(feature_list, dim=0)
        sec_sim = 0.0
        # 若启用类间困难样本，则更新hard_list
        if self.cfg.TRAIN.METRIC.OUTER_HARDSAMPLER:
            sim_mat = torch.matmul(feature_value, torch.t(feature_value))

            for i in range(sim_mat.shape[0]):
                sim = sim_mat[i]
                sim, index = torch.sort(sim, dim=0, descending=True)
                hard_index = index[0:self.cfg.TRAIN.METRIC.INSTANCES_NUM]
                HardSampler.insertSampler(i, hard_index.cpu().numpy())
                sec_sim += sim[1].item()

            sec_sim = sec_sim / sim_mat.shape[0]

        Logger.log(
            "[Val Epoch {}/{}] [sec_sim {:.6f}]".format(
                epoch,
                self.cfg.SOLVER.MAX_ITER,
                sec_sim,
            ))

        return label_list, feature_value

    def val_one_epoch(self, epoch):
        if epoch < self.cfg.TRAIN.METRIC.HARDSAMPLER_START_ITER:
            return

        self.model.eval()
        feature_label, feature_template = self.hard_sampler(epoch)

        if not self.val_loader:
            return

        test_labels = None
        test_features = None

        for image, target in tqdm(self.val_loader):
            target = target.cuda()
            with torch.no_grad():
                features = self.model(image.cuda())

            if comm.get_world_size() > 1:
                all_embedding = comm.concat_all_gather(features)
                all_targets = comm.concat_all_gather(target)
            else:
                all_embedding = features
                all_targets = target

            if test_labels is None:
                test_labels = all_targets
                test_features = all_embedding
            else:
                test_labels = torch.cat([test_labels, all_targets], dim=0)
                test_features = torch.cat([test_features, all_embedding], dim=0)

        if comm.is_main_process():
            features_map = {}
            labels_set = set(test_labels.cpu().numpy())
            for label in labels_set:
                pos = test_labels == label
                target_embeddings = test_features[pos]
                features_map[self.val_loader.dataset.classes[label]] = target_embeddings

            ok = 0.0
            hard_ok = 0.0
            total = 0
            class_not_in_template = ''
            for key, value in features_map.items():
                if key not in feature_label:
                    class_not_in_template += key + ' '
                    # print("类别{}不在模板中".format(key))
                    continue

                simi = torch.matmul(value, torch.t(feature_template))
                # 计算最大相似度对应的类别是不是目标类别，是则对，不是则错
                _, hard_index = simi.max(1)
                label_list = [feature_label[i] for i in hard_index.cpu().numpy()]

                for label in label_list:
                    total += 1
                    if label == key:
                        hard_ok += 1

                    if label.split('_')[0].split('-')[0] == key.split('_')[0].split('-')[0]:
                        ok += 1

            Logger.log(
                "[Val Epoch {}/{}] [acc {:.6f} ({}/{})] [hard acc {:.6f} ({}/{})] [类别{}不在模板中]".format(
                    epoch,
                    self.cfg.SOLVER.MAX_ITER,
                    ok / total,
                    int(ok), total,
                    hard_ok / total,
                    int(hard_ok), total,
                    class_not_in_template
                ))