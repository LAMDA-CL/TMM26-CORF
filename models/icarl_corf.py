import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from models.base import BaseLearner
from utils.inc_net import *
from utils.toolkit import *
from utils.Augmentation import Augmentation
from torch.utils.tensorboard import SummaryWriter
from utils.losses import *


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.learning_rate = args["learning_rate"]
        self.init_weight_decay = args["init_weight_decay"]
        self.weight_decay = args["weight_decay"]
        self.T = args["T"]
        self.num_workers = args["num_workers"]
        self.batch_size = args["batch_size"]
        self.init_epoch = args["init_epoch"]
        self.epochs = args["epochs"]

        self.inter_ref = args["inter_refinement"]
        self.intra_ref = args["intra_refinement"]
        self.inter_ref_high = args["inter_ref_high"]
        self.intra_ref_high = args["intra_ref_high"]
        self.lamda_ref = args["lamda_ref"]
        self.data_aug = Augmentation(device=self._device, aug_times=1)
        self.cam_extractor = None
        self.EPSILON = args.get("EPSILON", 1e-8)

        self.kernel = args["kernel"]
        self.k_cosine = args["k_cosine"]
        self.k_tstd = args["k_tstd"]
        self.kernel_criterion = KernelBasedDistillation(device=self._device, cosine_kernel=True, tstd_kernel=True)
        self.lamda_kernel = args["lamda_kernel"]


    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self.current_task_classes = list(range(self._known_classes, self._total_classes))
        self.task_size = data_manager.get_task_size(self._cur_task)
        self._network.update_morefc(self._known_classes, self._total_classes, self.task_size)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
            combined=True
        )
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                                       num_workers=self.num_workers, pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = {domain: DataLoader(
            test_dataset[domain], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        ) for domain in test_dataset.keys()}

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self.cam_extractor = NetGradCAM(self._network, device=self._device)
        if self._old_network is not None:
            self._old_network.to(self._device, non_blocking=True)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.learning_rate,
                weight_decay=self.init_weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.init_epoch
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.learning_rate,
                weight_decay=self.init_weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.epochs
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, losses_clf = 0.0, 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets, domain_labels) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device,
                                                                                         non_blocking=True)
                domain_labels = np.array(domain_labels)

                all_inputs_list = [inputs]
                all_targets_list = [targets]
                all_domains_list = [domain_labels]

                if self.inter_ref:
                    refined_inputs, refined_targets, refined_domain_labels = self.inter_refinement(self.data_aug, inputs,
                                                                                                   targets,
                                                                                                   domain_labels,
                                                                                                   self.cam_extractor,
                                                                                                   range=self.current_task_classes,
                                                                                                   high=self.inter_ref_high,
                                                                                                   ratio=self.lamda_ref * (
                                                                                                           2 - epoch / self.epochs))
                    if refined_inputs is not None and len(refined_inputs) > 0:
                        all_inputs_list.append(refined_inputs)
                        all_targets_list.append(refined_targets)
                        all_domains_list.append(refined_domain_labels)

                if self.intra_ref:
                    refined_inputs, refined_targets, refined_domain_labels = self.intra_refinement(self.data_aug, inputs,
                                                                                                   targets,
                                                                                                   domain_labels,
                                                                                                   self.cam_extractor,
                                                                                                   range=self.current_task_classes,
                                                                                                   high=self.intra_ref_high,
                                                                                                   ratio=self.lamda_ref * (
                                                                                                           2 - epoch / self.epochs))
                    if refined_inputs is not None and len(refined_inputs) > 0:
                        all_inputs_list.append(refined_inputs)
                        all_targets_list.append(refined_targets)
                        all_domains_list.append(refined_domain_labels)

                combined_inputs = torch.cat(all_inputs_list)
                combined_targets = torch.cat(all_targets_list)
                combined_domains = np.concatenate(all_domains_list)

                shuffle_indices = torch.randperm(combined_inputs.size(0))

                shuffled_inputs = combined_inputs[shuffle_indices]
                shuffled_targets = combined_targets[shuffle_indices]
                shuffled_domains = combined_domains[shuffle_indices.cpu().numpy()]

                outputs = self._network(shuffled_inputs)
                logits = outputs["logits"]
                features = outputs["features"]
                loss_clf = F.cross_entropy(logits, shuffled_targets)

                loss = loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()

                max_logits, preds = torch.max(logits, dim=1)
                correct += preds.eq(shuffled_targets.expand_as(preds)).cpu().sum()
                total += len(shuffled_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 20 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Train_accy {:.2f}, Test_accy {}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epoch,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epoch,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, losses_clf, losses_kd, losses_kernel = 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets, domain_labels) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device,
                                                                                         non_blocking=True)
                domain_labels = np.array(domain_labels)

                all_inputs_list = [inputs]
                all_targets_list = [targets]
                all_domains_list = [domain_labels]

                if self.inter_ref:
                    refined_inputs, refined_targets, refined_domain_labels = self.inter_refinement(self.data_aug, inputs,
                                                                                                   targets,
                                                                                                   domain_labels,
                                                                                                   self.cam_extractor,
                                                                                                   range=self.current_task_classes,
                                                                                                   high=self.inter_ref_high,
                                                                                                   ratio=self.lamda_ref * (
                                                                                                           2 - epoch / self.epochs))
                    if refined_inputs is not None and len(refined_inputs) > 0:
                        all_inputs_list.append(refined_inputs)
                        all_targets_list.append(refined_targets)
                        all_domains_list.append(refined_domain_labels)

                if self.intra_ref:
                    refined_inputs, refined_targets, refined_domain_labels = self.intra_refinement(self.data_aug, inputs,
                                                                                                   targets,
                                                                                                   domain_labels,
                                                                                                   self.cam_extractor,
                                                                                                   range=self.current_task_classes,
                                                                                                   high=self.intra_ref_high,
                                                                                                   ratio=self.lamda_ref * (
                                                                                                           2 - epoch / self.epochs))
                    if refined_inputs is not None and len(refined_inputs) > 0:
                        all_inputs_list.append(refined_inputs)
                        all_targets_list.append(refined_targets)
                        all_domains_list.append(refined_domain_labels)


                combined_inputs = torch.cat(all_inputs_list)
                combined_targets = torch.cat(all_targets_list)
                combined_domains = np.concatenate(all_domains_list)

                shuffle_indices = torch.randperm(combined_inputs.size(0))

                shuffled_inputs = combined_inputs[shuffle_indices]
                shuffled_targets = combined_targets[shuffle_indices]
                shuffled_domains = combined_domains[shuffle_indices.cpu().numpy()]

                outputs = self._network(shuffled_inputs)
                logits = outputs["logits"]
                features = outputs["features"]
                maps = outputs["fmaps"]

                loss_clf = F.cross_entropy(logits, shuffled_targets)

                loss_kd = self._KD_loss(
                    logits[:, :self._known_classes],
                    self._old_network(shuffled_inputs)["logits"][:, :self._known_classes],
                    self.T,
                )

                old_outputs = self._old_network(shuffled_inputs)
                old_fmaps = old_outputs["fmaps"]

                if self.kernel:
                    loss_kernel = self.kernel_criterion(old_fmaps, maps) * self.lamda_kernel
                else:
                    loss_kernel = torch.tensor(0.0).to(self._device)

                loss = loss_clf + loss_kd + loss_kernel

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_kd += loss_kd.item()
                losses_kernel += loss_kernel.item()

                max_logits, preds = torch.max(logits, dim=1)
                correct += preds.eq(shuffled_targets.expand_as(preds)).cpu().sum()
                total += len(shuffled_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 20 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Loss_kernel {:.3f}, Train_accy {:.2f}, Test_accy {}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    losses_kernel / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Loss_kernel {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    losses_kernel / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
