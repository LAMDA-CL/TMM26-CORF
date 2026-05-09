import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self.task_size = 10
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory, self._domains_memory = np.array([]), np.array([]), np.array([])
        self.topk = args["topk"]

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "total_tasks": self._total_classes,
            "known_tasks": self._known_classes,
            "cur_task": self._cur_task,
            "task_size": self.task_size,
            "model_state_dict": self._network.state_dict(),
            "old_model_state_dict": self._old_network.state_dict() if self._old_network is not None else None,
            "data_memory": self._data_memory,
            "targets_memory": self._targets_memory,
            "domains_memory": self._domains_memory
        }
        torch.save(save_dict, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self._known_classes = checkpoint['known_tasks']
        self._total_classes = checkpoint['total_tasks']
        self._cur_task = checkpoint["cur_task"]
        self.task_size = checkpoint["task_size"]
        self._network.update_morefc(self._known_classes, self._total_classes, self.task_size)
        self._network.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['old_model_state_dict']:
            self._old_network = copy.deepcopy(self._network)
            self._network.update_morefc(self._known_classes, self._total_classes, self.task_size)
            self._old_network.load_state_dict(checkpoint['old_model_state_dict'])
        self._data_memory = checkpoint['data_memory']
        self._targets_memory = checkpoint['targets_memory']
        self._domains_memory = checkpoint['domains_memory']
        self._network.eval()
        print(f"Model checkpoint loaded from {filename}, Task {self._cur_task}")

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, increment=self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = float(np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        ))
        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true, cnn_accy = {}, {}, {}
        for domain in self.test_loader.keys():
            y_pred[domain], y_true[domain] = self._eval_cnn(self.test_loader[domain])
            cnn_accy[domain] = self._evaluate(y_pred[domain], y_true[domain])

        if hasattr(self, "_class_means"):
            y_pred, y_true, nme_accy = {}, {}, {}
            for domain in self.test_loader.keys():
                y_pred[domain], y_true[domain] = self._eval_nme(self.test_loader[domain], self._class_means)
                nme_accy[domain] = self._evaluate(y_pred[domain], y_true[domain])
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory, self._domains_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = {}, {}
        for domain in loader.keys():
            correct[domain], total[domain] = 0, 0
            for i, (_, inputs, targets) in enumerate(loader[domain]):
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    outputs = model(inputs)["logits"]
                predicts = torch.max(outputs, dim=1)[1]
                correct[domain] += (predicts.cpu() == targets).sum()
                total[domain] += len(targets)

        acc = {domain: float(np.around(tensor2numpy(correct[domain]) * 100 / total[domain], decimals=2)) for domain in
               loader.keys()}
        return acc

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        datas, vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        topk_indices = np.argsort(scores, axis=1)[:, :self.topk]
        return topk_indices, y_true

    def _extract_vectors(self, loader):
        self._network.eval()
        all_datas, all_vectors, all_targets, all_domains = [], [], [], []
        has_domains = False
        with torch.no_grad():
            for i, data_batch in enumerate(loader):
                if len(data_batch) == 3:
                    _, inputs, targets = data_batch
                elif len(data_batch) == 2:  # (inputs, targets)
                    inputs, targets = data_batch
                elif len(data_batch) == 4:  # (_, inputs, targets, domains)
                    _, inputs, targets, domains = data_batch
                    has_domains = True
                else:
                    raise ValueError(f"Unexpected batch format with {len(data_batch)} elements")
                if has_domains:
                    _domains = np.array(domains)
                    all_domains.append(_domains)

                _inputs = inputs.to(self._device, non_blocking=True)
                _targets = targets.numpy()

                if isinstance(self._network, nn.DataParallel):
                    _vectors = self._network.module.extract_vector(_inputs)
                else:
                    _vectors = self._network.extract_vector(_inputs)
                all_datas.append(tensor2numpy(_inputs.detach()))
                all_targets.append(_targets)
                all_vectors.append(tensor2numpy(_vectors.detach()))

        if has_domains:
            return np.concatenate(all_datas), np.concatenate(all_vectors), np.concatenate(all_targets), np.concatenate(
                all_domains)
        else:
            return np.concatenate(all_datas), np.concatenate(all_vectors), np.concatenate(all_targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets, dummy_domains = copy.deepcopy(self._data_memory), \
            copy.deepcopy(self._targets_memory), copy.deepcopy(self._domains_memory)

        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory, self._domains_memory = np.array([]), np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt, dm = dummy_data[mask][:m], dummy_targets[mask][:m], dummy_domains[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )
            self._domains_memory = (
                np.concatenate((self._domains_memory, dm))
                if len(self._domains_memory) != 0
                else dm
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt, dm), combined=True
            )

            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            datas, vectors, targets, domains = self._extract_vectors(idx_loader)

            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            datas, targets, domains, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
                combined=True
            )

            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            datas_back, vectors, targets_back, domains_back = self._extract_vectors(idx_loader)
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            selected_domains = []
            for k in range(1, m + 1):
                if vectors.size == 0:
                    S = np.sum(np.stack(exemplar_vectors), axis=0) if exemplar_vectors else np.zeros_like(
                        exemplar_vectors[0])
                    mu_p = (np.stack(exemplar_vectors) + S) / k
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                    selected_exemplars.append(selected_exemplars[i])
                    exemplar_vectors.append(exemplar_vectors[i])
                    selected_domains.append(selected_domains[i])
                    continue

                S = np.sum(np.stack(exemplar_vectors), axis=0) if exemplar_vectors else np.zeros_like(vectors[0])
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(datas[i])  # New object to avoid passing by inference
                exemplar_vectors.append(vectors[i])  # New object to avoid passing by inference
                selected_domains.append(domains[i])

                vectors = np.delete(vectors, i, axis=0)
                datas = np.delete(datas, i, axis=0)
                domains = np.delete(domains, i, axis=0)

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            self._domains_memory = (
                np.concatenate((self._domains_memory, selected_domains))
                if len(self._domains_memory) != 0
                else selected_domains
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets, selected_domains),
                combined=True
            )

            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            datas_back, vectors, targets_back, domains_back = self._extract_vectors(idx_loader)

            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets, class_domains = (
                self._data_memory[mask],
                self._targets_memory[mask],
                self._domains_memory[mask]
            )

            datas, targets, domains, class_dset = data_manager.get_dataset(
                [], source="train", mode="test",
                appendent=(class_data, class_targets, class_domains),
                ret_data=True,
                combined=True
            )
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            datas, vectors, targets, domains = self._extract_vectors(class_loader)  # Vectors are on GPU
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)  # Normalize on GPU
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean  # Store in GPU tensor

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            datas, targets, domains, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
                combined=True
            )
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            datas_back, vectors, targets_back, domains_back = self._extract_vectors(idx_loader)
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            selected_domains = []
            for k in range(1, m + 1):
                if vectors.size == 0:
                    S = np.sum(np.stack(exemplar_vectors), axis=0) if exemplar_vectors else np.zeros_like(
                        exemplar_vectors[0])
                    mu_p = (np.stack(exemplar_vectors) + S) / k
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                    selected_exemplars.append(selected_exemplars[i])
                    exemplar_vectors.append(exemplar_vectors[i])
                    selected_domains.append(selected_domains[i])
                    continue

                S = np.sum(np.stack(exemplar_vectors), axis=0) if exemplar_vectors else np.zeros_like(vectors[0])
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(datas[i])  # New object to avoid passing by inference
                exemplar_vectors.append(vectors[i])  # New object to avoid passing by inference
                selected_domains.append(domains[i])

                vectors = np.delete(vectors, i, axis=0)
                datas = np.delete(datas, i, axis=0)
                domains = np.delete(domains, i, axis=0)

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            self._domains_memory = (
                np.concatenate((self._domains_memory, selected_domains))
                if len(self._domains_memory) != 0
                else selected_domains
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets, selected_domains),
                combined=True
            )

            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            datas_back, vectors, targets_back, domains_back = self._extract_vectors(idx_loader)

            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _augmentation(self, augmentor, inputs, targets, domain_labels, ratio=0.2, original_weight=0.5):
        augmented_inputs = None
        selected_targets = None
        selected_domain_labels = None

        with torch.no_grad():
            initial_logits = self._network(inputs)["logits"]
            max_logits, _ = torch.max(initial_logits, dim=1)

            k = int(inputs.shape[0] * ratio)

            if k > 0:
                _, top_indices = torch.topk(max_logits, k=k)

                selected_inputs = inputs[top_indices].clone().detach()
                selected_targets = targets[top_indices].clone().detach()
                selected_domain_labels = domain_labels[top_indices.cpu().numpy()].copy()

                augmented_inputs = augmentor(selected_inputs, original_weight)

                if augmented_inputs is not None:
                    augmented_inputs = augmented_inputs.clone().detach()

        if augmented_inputs is not None and len(augmented_inputs) > 0:
            return augmented_inputs, selected_targets, selected_domain_labels
        else:
            return (torch.tensor([], device=inputs.device),
                    torch.tensor([], device=targets.device),
                    np.array([]))

    def _refinement(self, inputs, targets, domain_labels, cam, blend_ratio=0.1, change_ratio=0.1):
        intra_inputs, intra_targets, intra_domains = torch.tensor([], device=self._device), torch.tensor([],
                                                                                                         device=self._device), np.array(
            [])
        inter_inputs, inter_targets, inter_domains = torch.tensor([], device=self._device), torch.tensor([],
                                                                                                         device=self._device), np.array(
            [])

        with torch.no_grad():
            logits = self._network(inputs)["logits"]
            max_logits, _ = torch.max(logits, dim=1)
            k_intra = int(inputs.size(0) * blend_ratio)

            if k_intra > 0:
                _, top_indices = torch.topk(max_logits, k=k_intra)
                sel_inputs = inputs[top_indices].clone()
                sel_targets = targets[top_indices].clone()
                sel_domains = domain_labels[top_indices.cpu().numpy()].copy()

                shuffled_inputs = sel_inputs.clone()
                shuffled_domains = sel_domains.copy()

                for t in torch.unique(sel_targets):
                    t_idx = (sel_targets == t).nonzero(as_tuple=True)[0].cpu()
                    if len(t_idx) > 1:
                        t_idx_np = t_idx.numpy()
                        perm = np.random.permutation(len(t_idx_np))
                        shuffled_inputs[t_idx] = sel_inputs[t_idx_np[perm]].clone()
                        shuffled_domains[t_idx] = sel_domains[t_idx_np[perm]].copy()

                changed = (shuffled_domains != sel_domains)
                changed_idx = torch.tensor(np.where(changed)[0], device=self._device)

                if len(changed_idx) > 0:
                    act_map = cam(sel_inputs).clone().detach()
                    flattened = act_map.view(sel_inputs.shape[0], -1)
                    k_indices = torch.floor(torch.tensor(flattened.shape[1] * change_ratio, device=self._device)).long()
                    k_indices = k_indices.unsqueeze(0).expand(flattened.shape[0], 1)
                    sorted_values, _ = torch.sort(flattened, dim=1)
                    k_indices = torch.clamp(k_indices, 0, flattened.shape[1] - 1)
                    thresholds = sorted_values.gather(1, k_indices)

                    src_inputs = sel_inputs[changed_idx].clone()
                    dst_inputs = shuffled_inputs[changed_idx].clone()
                    masks = (act_map <= thresholds.view(sel_inputs.shape[0], 1, 1))

                    expanded_masks = masks[changed_idx].unsqueeze(1).clone()
                    expanded_masks = expanded_masks.expand(-1, sel_inputs.shape[1], -1, -1).clone()

                    purturbed_inputs = src_inputs.clone()
                    purturbed_inputs = src_inputs * (~expanded_masks) + (src_inputs + dst_inputs) / 2 * expanded_masks

                    purturbed_targets = sel_targets[changed_idx].clone()
                    purturbed_domains = sel_domains[changed_idx.cpu().numpy()].copy()

                    purturbed_inputs = purturbed_inputs.contiguous()
                    purturbed_targets = purturbed_targets.contiguous()

        with torch.no_grad():
            logits = self._network(inputs)["logits"]
            max_logits, _ = torch.max(logits, dim=1)
            k_inter = int(inputs.size(0) * blend_ratio)

            if k_inter > 0:
                _, top_indices = torch.topk(-max_logits, k=k_inter)
                sel_inputs = inputs[top_indices].clone()
                sel_targets = targets[top_indices].clone()
                sel_domains = domain_labels[top_indices.cpu().numpy()].copy()

                shuffled_inputs = sel_inputs.clone()
                shuffled_targets = sel_targets.clone()
                shuffled_domains = sel_domains.copy()

                for d in np.unique(sel_domains):
                    d_idx = np.where(sel_domains == d)[0]
                    if len(d_idx) > 1:
                        perm = np.random.permutation(len(d_idx))
                        shuffled_inputs[d_idx] = sel_inputs[d_idx[perm]].clone()
                        shuffled_domains[d_idx] = sel_domains[d_idx[perm]].copy()

                changed = (sel_targets != shuffled_targets)
                changed_idx = torch.tensor(torch.where(changed)[0], device=self._device)

                if len(changed_idx) > 0:
                    act_map = cam(sel_inputs).clone().detach()
                    flattened = act_map.view(sel_inputs.shape[0], -1)
                    k_indices = torch.floor(torch.tensor(flattened.shape[1] * change_ratio, device=self._device)).long()
                    k_indices = k_indices.unsqueeze(0).expand(flattened.shape[0], 1)
                    k_indices = torch.clamp(k_indices, 0, flattened.shape[1] - 1)
                    thresholds = torch.sort(flattened, dim=1)[0].gather(1, k_indices)

                    src_inputs = sel_inputs[changed_idx].clone()
                    dst_inputs = shuffled_inputs[changed_idx].clone()
                    src_targets = sel_targets[changed_idx].clone()
                    dst_targets = shuffled_targets[changed_idx].clone()

                    masks = (act_map >= thresholds.view(sel_inputs.shape[0], 1, 1))
                    expanded_masks = masks[changed_idx].unsqueeze(1).clone()
                    expanded_masks = expanded_masks.expand(-1, sel_inputs.shape[1], -1, -1).clone()

                    inter_inputs = src_inputs * (~expanded_masks) + (src_inputs + dst_inputs) / 2 * expanded_masks

                    inter_targets = self._map_targets(src_targets, dst_targets)
                    inter_domains = sel_domains[changed_idx.cpu().numpy()]

                    inter_inputs = inter_inputs.contiguous()
                    inter_targets = inter_targets.contiguous()

        final_inputs = torch.cat([intra_inputs, inter_inputs], dim=0)
        final_targets = torch.cat([intra_targets, inter_targets], dim=0)
        final_domains = np.concatenate([intra_domains, inter_domains], axis=0)

        return final_inputs, final_targets, final_domains

    def inter_refinement(self, augmentor, inputs, targets, domain_labels, cam, range, high=True, ratio=0.1):
        inter_inputs, inter_targets, inter_domains = torch.tensor([], device=self._device), torch.tensor([],
                                                                                                         device=self._device), np.array(
            [])
        if range is not None:
            mask = torch.zeros_like(targets, dtype=torch.bool, device=self._device)
            for cls in range:
                mask |= (targets == cls)
            selected_indices = mask.nonzero(as_tuple=True)[0]
            if len(selected_indices) == 0:
                return inter_inputs, inter_targets, inter_domains
            inputs = inputs[selected_indices]
            targets = targets[selected_indices]
            domain_labels = domain_labels[selected_indices.cpu().numpy()]

        with torch.no_grad():
            logits = self._network(inputs)["logits"]
            max_logits, _ = torch.max(logits, dim=1)
            k_inter = int(inputs.size(0) * ratio)

            if k_inter > 0:
                if high:
                    _, top_indices = torch.topk(max_logits, k=k_inter)
                else:
                    _, top_indices = torch.topk(-max_logits, k=k_inter)
                sel_inputs = inputs[top_indices].clone()
                sel_targets = targets[top_indices].clone()
                sel_domains = domain_labels[top_indices.cpu().numpy()].copy()

                shuffled_inputs = sel_inputs.clone()
                shuffled_targets = sel_targets.clone()
                shuffled_domains = sel_domains.copy()

                for d in np.unique(sel_domains):
                    d_idx = np.where(sel_domains == d)[0]
                    if len(d_idx) > 1:
                        perm = np.random.permutation(len(d_idx))
                        shuffled_inputs[d_idx] = sel_inputs[d_idx[perm]].clone()
                        shuffled_targets[d_idx] = sel_targets[d_idx[perm]].clone()

                changed = (sel_targets != shuffled_targets)
                changed_idx = torch.tensor(torch.where(changed)[0], device=self._device)

                if len(changed_idx) > 0:
                    act_map = cam(sel_inputs).clone().detach()
                    flattened = act_map.view(sel_inputs.shape[0], -1)
                    k_indices = torch.floor(torch.tensor(flattened.shape[1] * ratio, device=self._device)).long()
                    k_indices = k_indices.unsqueeze(0).expand(flattened.shape[0], 1)
                    k_indices = torch.clamp(k_indices, 0, flattened.shape[1] - 1)
                    thresholds = torch.sort(flattened, dim=1)[0].gather(1, k_indices)

                    src_inputs = sel_inputs[changed_idx].clone()
                    dst_inputs = shuffled_inputs[changed_idx].clone()
                    src_targets = sel_targets[changed_idx].clone()
                    dst_targets = shuffled_targets[changed_idx].clone()

                    masks = (act_map >= thresholds.view(sel_inputs.shape[0], 1, 1))
                    expanded_masks = masks[changed_idx].unsqueeze(1).clone()
                    expanded_masks = expanded_masks.expand(-1, sel_inputs.shape[1], -1, -1).clone()

                    inter_inputs = src_inputs * (~expanded_masks) + (src_inputs + dst_inputs) / 2 * expanded_masks

                    inter_targets = self._map_targets(src_targets, dst_targets)
                    inter_domains = sel_domains[changed_idx.cpu().numpy()]

                    inter_inputs = inter_inputs.contiguous()
                    inter_targets = inter_targets.contiguous()

                    inter_inputs = augmentor(inter_inputs)

        return inter_inputs, inter_targets, inter_domains


    def intra_refinement(self, augmentor, inputs, targets, domain_labels, cam, range, high=True, ratio=0.1):
        intra_inputs, intra_targets, intra_domains = torch.tensor([], device=self._device), torch.tensor([],
                                                                                                         device=self._device), np.array(
            [])

        if range is not None:
            mask = torch.zeros_like(targets, dtype=torch.bool, device=self._device)
            for cls in range:
                mask |= (targets == cls)
            selected_indices = mask.nonzero(as_tuple=True)[0]
            if len(selected_indices) == 0:
                return intra_inputs, intra_targets, intra_domains
            inputs = inputs[selected_indices]
            targets = targets[selected_indices]
            domain_labels = domain_labels[selected_indices.cpu().numpy()]

        with torch.no_grad():
            logits = self._network(inputs)["logits"]
            max_logits, _ = torch.max(logits, dim=1)
            k_intra = int(inputs.size(0) * ratio)

            if k_intra > 0:
                if high:
                    _, top_indices = torch.topk(max_logits, k=k_intra)
                else:
                    _, top_indices = torch.topk(-max_logits, k=k_intra)
                sel_inputs = inputs[top_indices].clone()
                sel_targets = targets[top_indices].clone()
                sel_domains = domain_labels[top_indices.cpu().numpy()].copy()

                shuffled_inputs = sel_inputs.clone()
                shuffled_domains = sel_domains.copy()

                for t in torch.unique(sel_targets):
                    t_idx = (sel_targets == t).nonzero(as_tuple=True)[0]
                    if len(t_idx) > 1:
                        perm = np.random.permutation(len(t_idx))
                        shuffled_inputs[t_idx] = sel_inputs[t_idx[perm]].clone()

                changed = (shuffled_domains != sel_domains)
                changed_idx = torch.tensor(np.where(changed)[0], device=self._device)

                if len(changed_idx) > 0:
                    act_map = cam(sel_inputs).clone().detach()
                    flattened = act_map.view(sel_inputs.shape[0], -1)
                    k_indices = torch.floor(torch.tensor(flattened.shape[1] * ratio, device=self._device)).long()
                    k_indices = k_indices.unsqueeze(0).expand(flattened.shape[0], 1)
                    sorted_values, _ = torch.sort(flattened, dim=1)
                    k_indices = torch.clamp(k_indices, 0, flattened.shape[1] - 1)
                    thresholds = sorted_values.gather(1, k_indices)

                    src_inputs = sel_inputs[changed_idx].clone()
                    dst_inputs = shuffled_inputs[changed_idx].clone()
                    masks = (act_map <= thresholds.view(sel_inputs.shape[0], 1, 1))

                    expanded_masks = masks[changed_idx].unsqueeze(1).clone()
                    expanded_masks = expanded_masks.expand(-1, sel_inputs.shape[1], -1, -1).clone()

                    intra_inputs = src_inputs.clone()
                    intra_inputs = src_inputs * (~expanded_masks) + (src_inputs + dst_inputs) / 2 * expanded_masks

                    intra_targets = sel_targets[changed_idx].clone()
                    intra_domains = sel_domains[changed_idx.cpu().numpy()].copy()

                    intra_inputs = intra_inputs.contiguous()
                    intra_targets = intra_targets.contiguous()

                    intra_inputs = augmentor(intra_inputs)

        return intra_inputs, intra_targets, intra_domains

    def dual_refinement(self, augmentor, inputs, targets, domain_labels, cam, range, inter_high=True, intra_high=True,
                        ratio=0.1, original_weight=0.5):
        inter_inputs, inter_targets, inter_domains = \
            torch.tensor([], device=self._device), torch.tensor([], device=self._device), np.array([])
        intra_inputs, intra_targets, intra_domains = \
            torch.tensor([], device=self._device), torch.tensor([], device=self._device), np.array([])

        if range is not None:
            mask = torch.zeros_like(targets, dtype=torch.bool, device=self._device)
            for cls in range:
                mask |= (targets == cls)
            selected_indices = mask.nonzero(as_tuple=True)[0]
            if len(selected_indices) == 0:
                return inter_inputs, inter_targets, inter_domains
            inputs = inputs[selected_indices]
            targets = targets[selected_indices]
            domain_labels = domain_labels[selected_indices.cpu().numpy()]

        with torch.no_grad():
            logits = self._network(inputs)["logits"]
            max_logits, _ = torch.max(logits, dim=1)
            k_m = int(inputs.size(0) * ratio)

            if k_m > 0:
                if inter_high:
                    _, inter_top_indices = torch.topk(max_logits, k=k_m)
                else:
                    _, inter_top_indices = torch.topk(-max_logits, k=k_m)
                if intra_high:
                    _, intra_top_indices = torch.topk(max_logits, k=k_m)
                else:
                    _, intra_top_indices = torch.topk(-max_logits, k=k_m)

                inter_sel_inputs = inputs[inter_top_indices].clone()
                inter_sel_targets = targets[inter_top_indices].clone()
                inter_sel_domains = domain_labels[inter_top_indices.cpu().numpy()].copy()

                intra_sel_inputs = inputs[intra_top_indices].clone()
                intra_sel_targets = targets[intra_top_indices].clone()
                intra_sel_domains = domain_labels[intra_top_indices.cpu().numpy()].copy()

                inter_shuffled_inputs = inter_sel_inputs.clone()
                inter_shuffled_targets = inter_sel_targets.clone()
                inter_shuffled_domains = inter_sel_domains.copy()

                intra_shuffled_inputs = intra_sel_inputs.clone()
                intra_shuffled_targets = intra_sel_targets.clone()
                intra_shuffled_domains = intra_sel_domains.copy()

                for d in np.unique(inter_sel_domains):
                    d_idx = np.where(inter_sel_domains == d)[0]
                    if len(d_idx) > 1:
                        perm = np.random.permutation(len(d_idx))
                        inter_shuffled_inputs[d_idx] = inter_sel_inputs[d_idx[perm]].clone()
                        inter_shuffled_targets[d_idx] = inter_sel_targets[d_idx[perm]].clone()

                inter_changed = (inter_sel_targets != inter_shuffled_targets)
                inter_changed_idx = torch.tensor(torch.where(inter_changed)[0], device=self._device)

                for t in torch.unique(intra_sel_targets):
                    t_idx = (intra_sel_targets == t).nonzero(as_tuple=True)[0]
                    if len(t_idx) > 1:
                        perm = np.random.permutation(len(t_idx))
                        intra_shuffled_inputs[t_idx] = intra_sel_inputs[t_idx[perm]].clone()

                intra_changed = (intra_shuffled_domains != intra_sel_domains)
                intra_changed_idx = torch.tensor(np.where(intra_changed)[0], device=self._device)

                if len(inter_changed_idx) > 0:
                    inter_act_map = cam(inter_sel_inputs).clone().detach()
                    inter_flattened = inter_act_map.view(inter_sel_inputs.shape[0], -1)
                    k_indices = torch.floor(torch.tensor(inter_flattened.shape[1] * ratio, device=self._device)).long()
                    k_indices = k_indices.unsqueeze(0).expand(inter_flattened.shape[0], 1)
                    k_indices = torch.clamp(k_indices, 0, inter_flattened.shape[1] - 1)
                    thresholds = torch.sort(inter_flattened, dim=1)[0].gather(1, k_indices)

                    src_inputs = inter_sel_inputs[inter_changed_idx].clone()
                    dst_inputs = inter_shuffled_inputs[inter_changed_idx].clone()
                    src_targets = inter_sel_targets[inter_changed_idx].clone()
                    dst_targets = inter_shuffled_targets[inter_changed_idx].clone()

                    masks = (inter_act_map >= thresholds.view(inter_sel_inputs.shape[0], 1, 1))
                    expanded_masks = masks[inter_changed_idx].unsqueeze(1).clone()
                    expanded_masks = expanded_masks.expand(-1, inter_sel_inputs.shape[1], -1, -1).clone()

                    inter_inputs = src_inputs * (~expanded_masks) + (src_inputs + dst_inputs) / 2 * expanded_masks

                    inter_targets = self._map_targets(src_targets, dst_targets)
                    inter_domains = inter_sel_domains[inter_changed_idx.cpu().numpy()]

                    inter_inputs = inter_inputs.contiguous()
                    inter_targets = inter_targets.contiguous()

                    inter_inputs = augmentor(inter_inputs, original_weight)

                if len(intra_changed_idx) > 0:
                    intra_act_map = cam(intra_sel_inputs).clone().detach()
                    flattened = intra_act_map.view(intra_sel_inputs.shape[0], -1)
                    k_indices = torch.floor(torch.tensor(flattened.shape[1] * ratio, device=self._device)).long()
                    k_indices = k_indices.unsqueeze(0).expand(flattened.shape[0], 1)
                    sorted_values, _ = torch.sort(flattened, dim=1)
                    k_indices = torch.clamp(k_indices, 0, flattened.shape[1] - 1)
                    thresholds = sorted_values.gather(1, k_indices)

                    src_inputs = intra_sel_inputs[intra_changed_idx].clone()
                    dst_inputs = intra_shuffled_inputs[intra_changed_idx].clone()
                    masks = (intra_act_map <= thresholds.view(intra_sel_inputs.shape[0], 1, 1))

                    expanded_masks = masks[intra_changed_idx].unsqueeze(1).clone()
                    expanded_masks = expanded_masks.expand(-1, intra_sel_inputs.shape[1], -1, -1).clone()

                    intra_inputs = src_inputs.clone()
                    intra_inputs = src_inputs * (~expanded_masks) + (src_inputs + dst_inputs) / 2 * expanded_masks

                    intra_targets = intra_sel_targets[intra_changed_idx].clone()
                    intra_domains = intra_sel_domains[intra_changed_idx.cpu().numpy()].copy()

                    intra_inputs = intra_inputs.contiguous()
                    intra_targets = intra_targets.contiguous()

                    intra_inputs = augmentor(intra_inputs, original_weight)

        if intra_inputs.size(0) == 0 and inter_inputs.size(0) == 0:
            final_inputs = torch.tensor([], device=self._device)
            final_targets = torch.tensor([], device=self._device)
            final_domains = np.array([])
        elif intra_inputs.size(0) == 0:
            final_inputs = inter_inputs
            final_targets = inter_targets
            final_domains = inter_domains
        elif inter_inputs.size(0) == 0:
            final_inputs = intra_inputs
            final_targets = intra_targets
            final_domains = intra_domains
        else:
            final_inputs = torch.cat([intra_inputs, inter_inputs], dim=0)
            final_targets = torch.cat([intra_targets, inter_targets], dim=0)
            final_domains = np.concatenate([intra_domains, inter_domains], axis=0)

        return final_inputs, final_targets, final_domains

    def _map_targets(self, select_targets, perm_targets):
        assert (select_targets != perm_targets).all()
        large_targets = torch.max(select_targets, perm_targets) - self._known_classes
        small_targets = torch.min(select_targets, perm_targets) - self._known_classes

        mixup_targets = (large_targets * (large_targets - 1) / 2 + small_targets + self._total_classes).long()
        return mixup_targets

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
