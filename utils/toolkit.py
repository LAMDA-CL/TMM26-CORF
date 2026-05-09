import os
import numpy as np
import torch
import  json
from enum import Enum
from pytorch_grad_cam import GradCAM
from torch import nn
import copy

class GradCAMWrapper(torch.nn.Module):
    def __init__(self, dernet_model):
        super(GradCAMWrapper, self).__init__()
        self.dernet = dernet_model
        
        self.target_backbone = self.dernet.convnet

    def forward(self, x):
        result = self.target_backbone(x)
        
        if isinstance(result, dict) and 'features' in result:
            return result['features']
        return result


class NetGradCAM:
    def __init__(self, original_model, device=None):
        self.original_model = original_model
        self.device = device
        self._model_copy = None
        self._extractor = None
    
    def _init_extractor(self):
        if self._extractor is None:
            self._model_copy = copy.deepcopy(self.original_model)
            self._model_copy.eval()
            
            with torch.no_grad():
                for src_param, copy_param in zip(self.original_model.parameters(), 
                                               self._model_copy.parameters()):
                    copy_param.copy_(src_param.detach())
            
            wrapper = GradCAMWrapper(self._model_copy)
            target_layers = [self._model_copy.convnet.layer4[-1]]
            
            from pytorch_grad_cam import GradCAM
            self._extractor = GradCAM(model=wrapper, target_layers=target_layers)
    
    def __call__(self, inputs, targets=None):
        try:
            self._init_extractor()
            
            with torch.set_grad_enabled(True):
                cam = self._extractor(inputs, targets)
            
            return torch.tensor(cam, dtype=torch.float32, device=self.device)
        finally:
            self._cleanup_extractor()
    
    def _cleanup_extractor(self):
        if self._extractor is not None:
            if hasattr(self._extractor, 'activations_and_grads'):
                self._extractor.activations_and_grads.release()
            
            self._extractor = None
            self._model_copy = None
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class DERNetGradCAMWrapper(torch.nn.Module):
    def __init__(self, dernet_model, backbone_index=-1):
        super(DERNetGradCAMWrapper, self).__init__()
        
        with torch.no_grad():
            self.dernet = copy.deepcopy(dernet_model)
            self.dernet.eval()
        
        self.num_backbones = len(self.dernet.convnets)
        
        if backbone_index < 0:
            backbone_index = self.num_backbones + backbone_index
        
        assert 0 <= backbone_index < self.num_backbones, f"Backbone索引{backbone_index}超出范围[0,{self.num_backbones-1}]"
        
        self.backbone_index = backbone_index
        self.target_backbone = self.dernet.convnets[backbone_index]
    
    def forward(self, x):
        with torch.set_grad_enabled(True):
            result = self.target_backbone(x)
            
            if isinstance(result, dict) and 'features' in result:
                return result['features']
            
            return result
    
    def update_weights(self, original_model):
        with torch.no_grad():
            target_orig = original_model.convnets[self.backbone_index]
            target_copy = self.dernet.convnets[self.backbone_index]
            
            for p_src, p_tgt in zip(target_orig.parameters(), target_copy.parameters()):
                p_tgt.copy_(p_src.detach())

class DERNetEnsembleGradCAM:
    def __init__(self, dernet_model, device=None, weights=None):
        self.original_model = dernet_model
        self.device = device
        self.num_backbones = len(dernet_model.convnets)
        
        if weights is None:
            self.weights = torch.ones(self.num_backbones, device=self.device) / self.num_backbones
        else:
            assert len(weights) == self.num_backbones, "The number of weights must be equal to the number of backbones"
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, device=self.device)
            self.weights = weights / weights.sum()
        
        self.target_layer_indices = []
        for i in range(self.num_backbones):
            backbone = dernet_model.convnets[i]
            if hasattr(backbone, 'layer4') and len(backbone.layer4) > 0:
                self.target_layer_indices.append((i, 'layer4[-1]'))
            else:
                self.target_layer_indices.append((i, None))
        
        self._model_copy = None
        self._wrappers = []
        self._extractors = None
    
    def _init_extractors(self):

        if self._extractors is None:
            from pytorch_grad_cam import GradCAM

            self._model_copy = copy.deepcopy(self.original_model)
            self._model_copy.eval()

            for param in self._model_copy.parameters():
                param.requires_grad = True
            
            with torch.no_grad():
                for src_param, copy_param in zip(self.original_model.parameters(), 
                                               self._model_copy.parameters()):
                    copy_param.copy_(src_param.detach())
                
            self._extractors = []
            self._wrappers = []
            
            for i, layer_desc in self.target_layer_indices:
                wrapper = DERNetGradCAMWrapper(self._model_copy, backbone_index=i)
                self._wrappers.append(wrapper)
                
                if layer_desc is None:
                    target_layer = self._find_last_conv_layer(wrapper.target_backbone)
                else:
                    target_layer = eval(f"wrapper.target_backbone.{layer_desc}")
                
                extractor = GradCAM(
                    model=wrapper,
                    target_layers=[target_layer]
                )
                self._extractors.append(extractor)
    
    def _cleanup_extractors(self):

        if hasattr(self, '_extractors') and self._extractors:
            for extractor in self._extractors:
                try:
                    if hasattr(extractor, 'activations_and_grads'):
                        extractor.activations_and_grads.release()
                        
                        if hasattr(extractor.activations_and_grads, 'activations'):
                            for i in range(len(extractor.activations_and_grads.activations)):
                                extractor.activations_and_grads.activations[i] = None
                            extractor.activations_and_grads.activations = []
                        
                        if hasattr(extractor.activations_and_grads, 'gradients'):
                            for i in range(len(extractor.activations_and_grads.gradients)):
                                extractor.activations_and_grads.gradients[i] = None
                            extractor.activations_and_grads.gradients = []
                        
                        if hasattr(extractor.activations_and_grads, 'handles'):
                            for handle in extractor.activations_and_grads.handles:
                                handle.remove()
                            extractor.activations_and_grads.handles = []
                except Exception as e:
                    print(f"Warning: error in cleaning GradCAM: {e}")
        
        # 清除所有引用
        if hasattr(self, '_wrappers') and self._wrappers:
            for wrapper in self._wrappers:
                if hasattr(wrapper, 'dernet'):
                    wrapper.dernet = None
                if hasattr(wrapper, 'target_backbone'):
                    wrapper.target_backbone = None
        
        if hasattr(self, '_model_copy') and self._model_copy is not None:
            self._model_copy = None
        
        self._extractors = None
        self._wrappers = []
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _find_last_conv_layer(self, model):
        conv_layers = []
        
        def find_conv(module):
            for name, m in module.named_children():
                if isinstance(m, nn.Conv2d):
                    conv_layers.append(m)
                find_conv(m)
        
        find_conv(model)
        if not conv_layers:
            raise ValueError("fail to find conv layer")
        return conv_layers[-1]
 

    def __call__(self, inputs, targets=None):

        inputs_copy = inputs.clone().detach()
        
        try:
            self._init_extractors()
            
            
            all_cams = []
            batch_size = inputs_copy.shape[0]
            
            with torch.set_grad_enabled(True):
                for i, extractor in enumerate(self._extractors):
                    try:
                        inputs_copy.requires_grad_(True)
                        cam = extractor(inputs_copy, targets)
                        torch_cam = torch.tensor(cam, dtype=torch.float32, device=self.device)
                        all_cams.append(torch_cam)
                    except Exception as e:
                        print(f"Warning: error when calculating backbone {i} GradCAM: {e}")
                        zeros = torch.zeros((batch_size,) + inputs_copy.shape[2:], 
                                         device=self.device)
                        all_cams.append(zeros)
            
            if all_cams:
                stacked_cams = torch.stack(all_cams)
                weights_expanded = self.weights.view(-1, 1, 1, 1)
                ensemble_cam = (stacked_cams * weights_expanded).sum(dim=0)
                return ensemble_cam
            else:
                return torch.zeros((batch_size,) + inputs_copy.shape[2:], 
                                 device=self.device)
            
        finally:
            self._cleanup_extractors()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_extractors()
        return False
    
    def cleanup(self):
        self._cleanup_extractors()

    def update_model(self, new_model):

        self.original_model = new_model
        self._cleanup_extractors()


class MEMOGradCAMWrapper(torch.nn.Module):

    def __init__(self, memo_model, extractor_index=-1):

        super(MEMOGradCAMWrapper, self).__init__()
        
        with torch.no_grad():
            self.model = copy.deepcopy(memo_model)
            self.model.eval()
        
        self.TaskAgnosticExtractor = self.model.TaskAgnosticExtractor
        
        self.extractor_index = extractor_index
        
        assert extractor_index < len(self.model.AdaptiveExtractors), \
            f"Extractor索引{extractor_index}超出范围[0,{len(self.model.AdaptiveExtractors)-1}]"
        self.target_extractor = self.model.AdaptiveExtractors[extractor_index]
    
    def forward(self, x):
        with torch.set_grad_enabled(True):
            inter_feature = self.TaskAgnosticExtractor(x)
            final_feature = self.target_extractor(inter_feature)
            return final_feature
    
    def update_weights(self, original_model):
        with torch.no_grad():
            agnosticsrc = original_model.TaskAgnosticExtractor
            agnosticdst = self.TaskAgnosticExtractor
            for p_src, p_dst in zip(agnosticsrc.parameters(), agnosticdst.parameters()):
                p_dst.copy_(p_src.detach())

            src = original_model.AdaptiveExtractors[self.extractor_index]
            dst = self.target_extractor
            
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.copy_(p_src.detach())

class MEMOEnsembleGradCAM:
    def __init__(self, memo_model, device=None, weights=None):
        self.original_model = memo_model
        self.device = device
        self.num_extractors = len(memo_model.AdaptiveExtractors)
        
        if weights is None:
            self.weights = torch.ones(self.num_extractors, device=self.device) / self.num_extractors
        else:
            assert len(weights) == self.num_extractors
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, device=self.device)
            self.weights = weights / weights.sum()
        
        self.target_layer_indices = []
        
        
        for i in range(self.num_extractors):
            if hasattr(memo_model.AdaptiveExtractors[i], 'layer4'):
                self.target_layer_indices.append((i, 'layer4[-1]'))
            else:
                self.target_layer_indices.append((i, None))
        
        self._model_copy = None
        self._wrappers = []
        self._extractors = None

    def _init_extractors(self):

        if self._extractors is None:
            
            from pytorch_grad_cam import GradCAM

            self._model_copy = copy.deepcopy(self.original_model)
            self._model_copy.eval()

            for param in self._model_copy.parameters():
                param.requires_grad = True

            with torch.no_grad():
                for src_param, copy_param in zip(self.original_model.parameters(), 
                                               self._model_copy.parameters()):
                    copy_param.copy_(src_param.detach())
            
            
            self._extractors = []
            self._wrappers = []
            
            for i, layer_desc in self.target_layer_indices:
                wrapper = MEMOGradCAMWrapper(self._model_copy, extractor_index=i)
                self._wrappers.append(wrapper)
                
                if layer_desc is None:
                    target_layer = self._find_last_conv_layer(wrapper.target_backbone)
                else:
                    target_layer = eval(f"wrapper.target_extractor.{layer_desc}")
                
                extractor = GradCAM(
                    model=wrapper,
                    target_layers=[target_layer]
                )
                self._extractors.append(extractor)
    
    def _cleanup_extractors(self):

        if hasattr(self, '_extractors') and self._extractors:
            for extractor in self._extractors:
                try:
                    if hasattr(extractor, 'activations_and_grads'):
                        extractor.activations_and_grads.release()
                        
                        if hasattr(extractor.activations_and_grads, 'activations'):
                            for i in range(len(extractor.activations_and_grads.activations)):
                                extractor.activations_and_grads.activations[i] = None
                            extractor.activations_and_grads.activations = []
                        
                        if hasattr(extractor.activations_and_grads, 'gradients'):
                            for i in range(len(extractor.activations_and_grads.gradients)):
                                extractor.activations_and_grads.gradients[i] = None
                            extractor.activations_and_grads.gradients = []
                        
                        if hasattr(extractor.activations_and_grads, 'handles'):
                            for handle in extractor.activations_and_grads.handles:
                                handle.remove()
                            extractor.activations_and_grads.handles = []
                except Exception as e:
                    print(f"Warning: {e}")
        
        if hasattr(self, '_wrappers') and self._wrappers:
            for wrapper in self._wrappers:
                if hasattr(wrapper, 'dernet'):
                    wrapper.dernet = None
                if hasattr(wrapper, 'target_backbone'):
                    wrapper.target_backbone = None
        
        if hasattr(self, '_model_copy') and self._model_copy is not None:
            self._model_copy = None
        
        self._extractors = None
        self._wrappers = []
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _find_last_conv_layer(self, model):
        conv_layers = []
        
        def find_conv(module):
            for name, m in module.named_children():
                if isinstance(m, nn.Conv2d):
                    conv_layers.append(m)
                find_conv(m)
        
        find_conv(model)
        if not conv_layers:
            raise ValueError("no conv")
        return conv_layers[-1]
    
    def __call__(self, inputs, targets=None):

        inputs_copy = inputs.clone().detach()
        
        try:
            self._init_extractors()
            
            all_cams = []
            batch_size = inputs_copy.shape[0]

            with torch.set_grad_enabled(True):
                for i, extractor in enumerate(self._extractors):
                    try:
                        inputs_copy.requires_grad_(True)
                        cam = extractor(inputs_copy, targets)
                        
                        torch_cam = torch.tensor(cam, dtype=torch.float32, device=self.device)
                        
                        all_cams.append(torch_cam)
                    except Exception as e:
                        print(f"警告: 计算backbone {i}的GradCAM时出错: {e}")
                        zeros = torch.zeros((batch_size,) + inputs_copy.shape[2:], device=self.device)
                        all_cams.append(zeros)
            
            if all_cams:
                stacked_cams = torch.stack(all_cams)
                
                weights_expanded = self.weights.view(-1, 1, 1, 1)
                
                ensemble_cam = (stacked_cams * weights_expanded).sum(dim=0)
                
                return ensemble_cam
            else:
                return torch.zeros((batch_size,) + inputs_copy.shape[2:], device=self.device)
            
        finally:
            self._cleanup_extractors()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_extractors()
        return False
    
    def cleanup(self):
        self._cleanup_extractors()

    def update_model(self, new_model):

        self.original_model = new_model
        self._cleanup_extractors()



class FOSTERGradCAMWrapper(torch.nn.Module):

    def __init__(self, net_model, backbone_index=-1):

        super(FOSTERGradCAMWrapper, self).__init__()
        
        with torch.no_grad():
            self.net = copy.deepcopy(net_model)
            self.net.eval()
        
        self.num_backbones = len(self.net.convnets)
        
        if backbone_index < 0:
            backbone_index = self.num_backbones + backbone_index
        
        assert 0 <= backbone_index < self.num_backbones, f"Backbone索引{backbone_index}超出范围[0,{self.num_backbones-1}]"
        
        self.backbone_index = backbone_index
        self.target_backbone = self.net.convnets[backbone_index]
    
    def forward(self, x):

        with torch.set_grad_enabled(True):
            result = self.target_backbone(x)
            
            if isinstance(result, dict) and 'features' in result:
                return result['features']
            
            return result
    
    def update_weights(self, original_model):

        with torch.no_grad():
            target_orig = original_model.convnets[self.backbone_index]
            target_copy = self.dernet.convnets[self.backbone_index]
            
            for p_src, p_tgt in zip(target_orig.parameters(), target_copy.parameters()):
                p_tgt.copy_(p_src.detach())

class FOSTEREnsembleGradCAM:

    def __init__(self, net_model, device=None, weights=None):

        self.original_model = net_model
        self.device = device
        self.num_backbones = len(net_model.convnets)
        
        if weights is None:
            self.weights = torch.ones(self.num_backbones, device=self.device) / self.num_backbones
        else:
            assert len(weights) == self.num_backbones, "num of weights must be equal to mum of backbone"
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, device=self.device)
            self.weights = weights / weights.sum()
        
        self.target_layer_indices = []
        for i in range(self.num_backbones):
            backbone = net_model.convnets[i]
            if hasattr(backbone, 'layer4') and len(backbone.layer4) > 0:
                self.target_layer_indices.append((i, 'layer4[-1]'))
            else:
                self.target_layer_indices.append((i, None))
        
        self._model_copy = None
        self._wrappers = []
        self._extractors = None
    
    def _init_extractors(self):

        if self._extractors is None:
            from pytorch_grad_cam import GradCAM

            self._model_copy = copy.deepcopy(self.original_model)
            self._model_copy.eval()

            for param in self._model_copy.parameters():
                param.requires_grad = True
            
            with torch.no_grad():
                for src_param, copy_param in zip(self.original_model.parameters(), 
                                               self._model_copy.parameters()):
                    copy_param.copy_(src_param.detach())
                
            self._extractors = []
            self._wrappers = []
            
            for i, layer_desc in self.target_layer_indices:
                wrapper = FOSTERGradCAMWrapper(self._model_copy, backbone_index=i)
                self._wrappers.append(wrapper)
                
                if layer_desc is None:
                    target_layer = self._find_last_conv_layer(wrapper.target_backbone)
                else:
                    target_layer = eval(f"wrapper.target_backbone.{layer_desc}")
                
                extractor = GradCAM(
                    model=wrapper,
                    target_layers=[target_layer]
                )
                self._extractors.append(extractor)
    
    def _cleanup_extractors(self):

        if hasattr(self, '_extractors') and self._extractors:
            for extractor in self._extractors:
                try:
                    if hasattr(extractor, 'activations_and_grads'):
                        extractor.activations_and_grads.release()
                        
                        if hasattr(extractor.activations_and_grads, 'activations'):
                            for i in range(len(extractor.activations_and_grads.activations)):
                                extractor.activations_and_grads.activations[i] = None
                            extractor.activations_and_grads.activations = []
                        
                        if hasattr(extractor.activations_and_grads, 'gradients'):
                            for i in range(len(extractor.activations_and_grads.gradients)):
                                extractor.activations_and_grads.gradients[i] = None
                            extractor.activations_and_grads.gradients = []
                        
                        if hasattr(extractor.activations_and_grads, 'handles'):
                            for handle in extractor.activations_and_grads.handles:
                                handle.remove()
                            extractor.activations_and_grads.handles = []
                except Exception as e:
                    print(f"error: {e}")
        
        if hasattr(self, '_wrappers') and self._wrappers:
            for wrapper in self._wrappers:
                if hasattr(wrapper, 'dernet'):
                    wrapper.dernet = None
                if hasattr(wrapper, 'target_backbone'):
                    wrapper.target_backbone = None
        
        if hasattr(self, '_model_copy') and self._model_copy is not None:
            self._model_copy = None
        
        self._extractors = None
        self._wrappers = []
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _find_last_conv_layer(self, model):
        conv_layers = []
        
        def find_conv(module):
            for name, m in module.named_children():
                if isinstance(m, nn.Conv2d):
                    conv_layers.append(m)
                find_conv(m)
        
        find_conv(model)
        if not conv_layers:
            raise ValueError("failed to find conv layer")
        return conv_layers[-1]
 

    def __call__(self, inputs, targets=None):
        inputs_copy = inputs.clone().detach()
        
        try:
            self._init_extractors()
            
            
            all_cams = []
            batch_size = inputs_copy.shape[0]
            
            with torch.set_grad_enabled(True):
                for i, extractor in enumerate(self._extractors):
                    try:
                        inputs_copy.requires_grad_(True)
                        cam = extractor(inputs_copy, targets)
                        torch_cam = torch.tensor(cam, dtype=torch.float32, device=self.device)
                        all_cams.append(torch_cam)
                    except Exception as e:
                        print(f"warning: calculatin backbone {i} GradCAM error: {e}")
                        zeros = torch.zeros((batch_size,) + inputs_copy.shape[2:], 
                                         device=self.device)
                        all_cams.append(zeros)
            
            if all_cams:
                stacked_cams = torch.stack(all_cams)
                weights_expanded = self.weights.view(-1, 1, 1, 1)
                ensemble_cam = (stacked_cams * weights_expanded).sum(dim=0)
                return ensemble_cam
            else:
                return torch.zeros((batch_size,) + inputs_copy.shape[2:], 
                                 device=self.device)
            
        finally:
            self._cleanup_extractors()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_extractors()
        return False  # 不抑制异常
    
    def cleanup(self):
        self._cleanup_extractors()

    def update_model(self, new_model):
        self.original_model = new_model
        self._cleanup_extractors()







class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=15):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = float(np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    ))

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = float(np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        ))

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else float(np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        ))
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = float(np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    ))

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1: 
        fc_weight = model._network.fc.weight.data    
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")

def save_model(args, model):
    #used in PODNet
    _path = os.path.join(args['logfilename'], "model.pt")
    if len(args['device']) > 1:
        weight = model._network   
    else:
        weight = model._network.cpu()
    torch.save(weight, _path)