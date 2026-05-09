import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import *

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, source_domain,target_domain):
        self.dataset_name = dataset_name
        self.source_domain =source_domain
        self.target_domain=target_domain
        
        self._setup_data(dataset_name, shuffle, seed)

        self.cached_indices = set()
        self.cached_data = {}
        self.cached_targets = {}
        
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None, combined = False
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        if source == "train":
            data, targets = {}, {}
            for domain in x.keys():
                data[domain] = []
                targets[domain] = []
                for idx in indices:
                    if m_rate is None:
                        class_data, class_targets = self._select(
                            x[domain], y[domain], low_range=idx, high_range=idx + 1
                        )
                    else:
                        class_data, class_targets = self._select_rmm(
                            x[domain], y[domain], low_range=idx, high_range=idx + 1, m_rate=m_rate
                        )
                    data[domain].append(class_data)
                    targets[domain].append(class_targets)
                if data[domain]:
                    data[domain] = np.concatenate(data[domain])
                    targets[domain] = np.concatenate(targets[domain])
            
            if appendent is not None and len(appendent) != 0:
                dat, tar, dom = appendent
                for i, (dt, t, d) in enumerate(zip(dat, tar, dom)):
                    if len(data[d]) == 0:
                        data[d] = np.array([dt])
                        targets[d] = np.array([t])
                    else:
                        data[d] = np.concatenate((data[d], [dt]), axis=0)
                        targets[d] = np.concatenate((targets[d], [t]), axis=0)
            dataset_dict = {
                domain: DummyDataset(data[domain], targets[domain], trsf, self.use_path)
                for domain in data
            }
            if combined:
                combined_data, combined_targets, combined_domains= [],[], []
                for domain in data:
                    combined_data.append(data[domain])
                    combined_targets.append(targets[domain])
                    combined_domains.append(np.array([domain]*len(data[domain]), dtype= object))
                combined_data, combined_targets, combined_domains = np.concatenate(combined_data), np.concatenate(combined_targets), np.concatenate(combined_domains)
                        
                n_samples = len(combined_data)
                shuffle_indices = np.random.permutation(n_samples)
                combined_data = combined_data[shuffle_indices]
                combined_targets = combined_targets[shuffle_indices]
                combined_domains = combined_domains[shuffle_indices]
                combined_dataset_dict = DummyDataset(combined_data, combined_targets, trsf, self.use_path, domains=combined_domains)
                if ret_data:
                    return combined_data, combined_targets, combined_domains, combined_dataset_dict
                else:
                    return combined_dataset_dict
            else:
                if ret_data:
                    return data, targets, dataset_dict
                else:
                    return dataset_dict
        
        elif source == "test":
            new_indices = set(indices) - self.cached_indices
            if not new_indices:
                if set(indices) == self.cached_indices:
                    data = {
                        domain: np.concatenate(self.cached_data[domain]) for domain in self.cached_data
                    }
                    targets = {
                        domain: np.concatenate(self.cached_targets[domain]) for domain in self.cached_targets
                    }

                    dataset_dict = {
                        domain: DummyDataset(data[domain], targets[domain], trsf, self.use_path)
                        for domain in data
                    }
                    if combined:
                        combined_data, combined_targets, combined_domains= [],[], []
                        for domain in data:
                            combined_data.append(data[domain])
                            combined_targets.append(targets[domain])
                            combined_domains.append(np.array([domain]*data[domain].shape[0], dtype= object))
                        combined_data, combined_targets, combined_domains = np.concatenate(combined_data), np.concatenate(combined_targets), np.concatenate(combined_domains)
                        
                        n_samples = len(combined_data)
                        shuffle_indices = np.random.permutation(n_samples)
                        combined_data = combined_data[shuffle_indices]
                        combined_targets = combined_targets[shuffle_indices]
                        combined_domains = combined_domains[shuffle_indices]

                        combined_dataset_dict = DummyDataset(combined_data, combined_targets, trsf, self.use_path, domains=combined_domains)
                        if ret_data:
                            return combined_data, combined_targets, combined_domains, combined_dataset_dict
                        else:
                            return combined_dataset_dict
                    else:
                        if ret_data:
                            return data, targets, dataset_dict
                        else:
                            return dataset_dict
                else:
                    data_dict , targets_dict = {}, {}
                    indices_set = set(indices)
                    for domain in x.keys():
                        data_dict[domain] = []
                        targets_dict[domain] = []

                        domain_data, domain_targets = [], []
                        for idx in indices_set:
                            if m_rate is None:
                                class_data, class_targets = self._select(
                                    x[domain], y[domain], low_range=idx, high_range=idx + 1
                                )
                            else:
                                class_data, class_targets = self._select_rmm(
                                    x[domain], y[domain], low_range=idx, high_range=idx + 1, m_rate=m_rate
                                )
                            domain_data.append(class_data)
                            domain_targets.append(class_targets)

                        if domain_data:
                            data_dict[domain].append(np.concatenate(domain_data))
                            targets_dict[domain].append(np.concatenate(domain_targets))

                    merged_data = {
                        domain: np.concatenate(data_dict[domain]) for domain in data_dict
                    }
                    merged_targets = {
                        domain: np.concatenate(targets_dict[domain]) for domain in targets_dict
                    }
                    dataset_dict = {
                        domain: DummyDataset(merged_data[domain], merged_targets[domain], trsf, self.use_path)
                        for domain in merged_data
                    }
                    if combined:
                        combined_data, combined_targets, combined_domains= [], [], []
                        for domain in data:
                            combined_data.append(data[domain])
                            combined_targets.append(targets[domain])
                            combined_domains.append(np.array([domain]*data[domain].shape[0], dtype= object))
                        combined_data, combined_targets, combined_domains = np.concatenate(combined_data), np.concatenate(combined_targets), np.concatenate(combined_domains)
                        
                        n_samples = len(combined_data)
                        shuffle_indices = np.random.permutation(n_samples)
                        combined_data = combined_data[shuffle_indices]
                        combined_targets = combined_targets[shuffle_indices]
                        combined_domains = combined_domains[shuffle_indices]
    
                        combined_dataset_dict = DummyDataset(combined_data, combined_targets, trsf, self.use_path, domains=combined_domains)
                        if ret_data:
                            return combined_data, combined_targets, combined_domains, combined_dataset_dict
                        else:
                            return combined_dataset_dict
                    else:
                        if ret_data:
                            return data, targets, dataset_dict
                        else:
                            return dataset_dict
                
                
            for domain in x.keys():
                if domain not in self.cached_data:
                    self.cached_data[domain] = []
                    self.cached_targets[domain] = []

                domain_data, domain_targets = [], []
                for idx in new_indices:
                    if m_rate is None:
                        class_data, class_targets = self._select(
                            x[domain], y[domain], low_range=idx, high_range=idx + 1
                        )
                    else:
                        class_data, class_targets = self._select_rmm(
                            x[domain], y[domain], low_range=idx, high_range=idx + 1, m_rate=m_rate
                        )
                    domain_data.append(class_data)
                    domain_targets.append(class_targets)

                if domain_data:
                    self.cached_data[domain].append(np.concatenate(domain_data))
                    self.cached_targets[domain].append(np.concatenate(domain_targets))

            self.cached_indices.update(new_indices)

            merged_data = {
                domain: np.concatenate(self.cached_data[domain]) for domain in self.cached_data
            }
            merged_targets = {
                domain: np.concatenate(self.cached_targets[domain]) for domain in self.cached_targets
            }
            dataset_dict = {
                domain: DummyDataset(merged_data[domain], merged_targets[domain], trsf, self.use_path)
                for domain in merged_data
            }
            if combined:
                combined_data, combined_targets, combined_domains= [],[], []
                for domain in data:
                    combined_data.append(data[domain])
                    combined_targets.append(targets[domain])
                    combined_domains.append(np.array([domain]*data[domain].shape[0], dtype= object))
                combined_data, combined_targets, combined_domains = np.concatenate(combined_data), np.concatenate(combined_targets), np.concatenate(combined_domains)
                        
                n_samples = len(combined_data)
                shuffle_indices = np.random.permutation(n_samples)
                combined_data = combined_data[shuffle_indices]
                combined_targets = combined_targets[shuffle_indices]
                combined_domains = combined_domains[shuffle_indices]

                combined_dataset_dict = {DummyDataset(combined_data, combined_targets, trsf, self.use_path, domains=combined_domains)}
                if ret_data:
                    return combined_data, combined_targets, combined_domains, combined_dataset_dict
                else:
                    return combined_dataset_dict
            else:
                if ret_data:
                    return data, targets, dataset_dict
                else:
                    return dataset_dict

        
    def get_finetune_dataset(self,known_classes,total_classes,source,mode,appendent,type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx+1)
            num=len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot*(total_classes-known_classes)/known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot/(total_classes-known_classes))
        for idx in range(known_classes,total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data),new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data=np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0, ret_data=False, m_rate=None, combined = False
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []

        if source == "train":
            train_data, train_targets = {}, {}
            val_data, val_targets = {}, {}
            for domain in x.keys():
                train_data[domain], train_targets[domain] = [], []
                val_data[domain], val_targets[domain] = [], []
                for idx in indices:
                    if m_rate is None:
                        class_data, class_targets = self._select(
                            x[domain], y[domain], low_range=idx, high_range=idx + 1
                        )
                    else:
                        class_data, class_targets = self._select_rmm(
                            x[domain], y[domain], low_range=idx, high_range=idx + 1
                        )
                    val_indx = np.random.choice(
                        len(class_data), val_samples_per_class, replace=False
                    )
                    train_indx = list(set(np.arange(len(class_data)))-set(val_indx))
                    val_data[domain].append(class_data[val_indx])
                    val_targets[domain].append(class_targets[val_indx])
                    train_data[domain].append(class_data[train_indx])
                    train_targets[domain].append(class_targets[train_indx])
                if train_data[domain]:
                    train_data[domain] = np.concatenate(train_data[domain])
                    train_targets[domain] = np.concatenate(train_targets[domain])
                    val_data[domain] = np.concatenate(val_data[domain])
                    val_targets[domain] = np.concatenate(val_targets[domain])
            
            if appendent is not None and len(appendent) != 0:
                dat, tar, dom = appendent
                for i, (dt, t, d) in enumerate(zip(dat, tar, dom)):
                    train_data[d] = np.concatenate((train_data[d], [dt]), axis=0)
                    train_targets[d] = np.concatenate((train_targets[d], [t]), axis=0)
                    val_data[d] = np.concatenate((val_data[d], [dt]), axis=0)
                    val_targets[d] = np.concatenate((val_targets[d], [t]), axis=0)

            train_dataset_dict = {
                domain: DummyDataset(train_data[domain], train_targets[domain], trsf, self.use_path)
                for domain in train_data
            }

            val_dataset_dict = {
                domain: DummyDataset(val_data[domain], val_targets[domain], trsf, self.use_path)
                for domain in val_data
            }
            if combined:
                train_combined_data, train_combined_targets, train_combined_domains= [],[], []
                for domain in train_data:
                    train_combined_data.append(train_data[domain])
                    train_combined_targets.append(train_targets[domain])
                    train_combined_domains.append(np.array([domain]*len(train_data[domain]), dtype= object))
                train_combined_data, train_combined_targets, train_combined_domains = np.concatenate(train_combined_data), np.concatenate(train_combined_targets), np.concatenate(train_combined_domains)
                        
                n_samples = len(train_combined_data)
                shuffle_indices = np.random.permutation(n_samples)
                train_combined_data = train_combined_data[shuffle_indices]
                train_combined_targets = train_combined_targets[shuffle_indices]
                train_combined_domains = train_combined_domains[shuffle_indices]
                train_combined_dataset_dict = DummyDataset(train_combined_data, train_combined_targets, trsf, self.use_path, domains=train_combined_domains)


                val_combined_data, val_combined_targets, val_combined_domains= [],[], []
                for domain in train_data:
                    val_combined_data.append(val_data[domain])
                    val_combined_targets.append(val_targets[domain])
                    val_combined_domains.append(np.array([domain]*len(val_data[domain]), dtype= object))
                val_combined_data, val_combined_targets, val_combined_domains = np.concatenate(val_combined_data), np.concatenate(val_combined_targets), np.concatenate(val_combined_domains)
                        
                n_samples = len(val_combined_data)
                shuffle_indices = np.random.permutation(n_samples)
                val_combined_data = val_combined_data[shuffle_indices]
                val_combined_targets = val_combined_targets[shuffle_indices]
                val_combined_domains = val_combined_domains[shuffle_indices]
                val_combined_dataset_dict = DummyDataset(val_combined_data, val_combined_targets, trsf, self.use_path, domains=val_combined_domains)

                if ret_data:
                    return (train_combined_data, train_combined_targets, train_combined_domains, train_combined_dataset_dict),\
                            (val_combined_data, val_combined_targets, val_combined_domains, val_combined_dataset_dict)
                else:
                    return train_combined_dataset_dict, val_combined_dataset_dict
            else:
                if ret_data:
                    return (train_data, train_targets, train_dataset_dict),\
                            (val_data, val_targets, val_dataset_dict)
                else:
                    return train_dataset_dict, val_dataset_dict
    

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)

        # Data
        self._train_data, self._train_targets = {}, {}
        for domain in self.source_domain:
            self._train_data[domain] = idata._train_data[domain]
            self._train_targets[domain] = idata._train_targets[domain]
        self._test_data, self._test_targets = idata._test_data, idata._test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        self.domains=idata.domains

        # Order
        self.classes=idata.classes
        order = np.arange(len(self.classes))
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = {
            domain: _map_new_class_index(targets, self._class_order)
            for domain, targets in self._train_targets.items()
        }

        self._test_targets = {
            domain: _map_new_class_index(targets, self._class_order)
            for domain, targets in self._test_targets.items()
        }

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

    def getlen_foster(self, index):
        total_count = 0
        
        for domain, targets in self._train_targets.items():
            domain_count = np.sum(np.where(targets == index, 1, 0))
            total_count += domain_count
        
        return total_count


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, domains = None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.domains = domains
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        if self.domains is not None:
            domain = self.domains[idx]
            return idx, image, label, domain
        else:
            return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "domainnet":
        return dataset(root='data', dataset_name='DomainNet')
    elif name == "officehome":
        return dataset(root='data', dataset_name='OfficeHome')
    elif name == "pacs":
        return dataset(root='data', dataset_name='PACS')
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
