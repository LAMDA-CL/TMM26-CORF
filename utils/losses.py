from __future__ import print_function

import torch
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F

class AugmentedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, norm=2, device=None):
        super(AugmentedTripletLoss, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.margin = torch.tensor(margin, device=self.device)
        self.norm = torch.tensor(norm, device=self.device)
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin.item()).to(self.device)

    def forward(self, inputs, targets, center, classes):
        n = inputs.size(0)  # batch_size

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        num_proto = classes
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            if dist[i][mask[i] == 0].numel() == 0:
                dist_an.append((dist[i][mask[i]].max() + self.margin).unsqueeze(0))
            else:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap).to(self.device)
        if num_proto > 0:
            center_slice = center[:num_proto]
            center_norm = center_slice / (center_slice.norm(dim=1, keepdim=True) + 1e-12)
            for i in range(n):
                for j in range(num_proto):
                    distp = torch.norm(inputs[i].unsqueeze(0) - center_norm[j], self.norm.item()).clamp(min=1e-12)
                    dist_an[i] = torch.min(dist_an[i].squeeze(0), distp).unsqueeze(0)
        dist_an = torch.cat(dist_an).to(self.device)
        y = torch.ones_like(dist_an, device=self.device)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss




class KernelBasedDistillation(nn.Module):
    def __init__(self, device, cosine_kernel=True, tstd_kernel=True):
        super().__init__()
        assert cosine_kernel or tstd_kernel, "At least one kernel must be enabled."
        self.device = device
        self.cosine_kernel = cosine_kernel
        self.tstd_kernel = tstd_kernel

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0
        kernel_count = 0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)

            if self.cosine_kernel:
                t_cos = F.normalize(t_flat, dim=1)
                s_cos = F.normalize(s_flat, dim=1)
                p_t = self._cosine_kernel(t_cos)
                p_s = self._cosine_kernel(s_cos)
                total_loss += self._symmetric_log_loss(p_t, p_s)
                kernel_count += 1

            if self.tstd_kernel:
                p_t = self._tstudent_kernel(t_flat)
                p_s = self._tstudent_kernel(s_flat)
                total_loss += self._symmetric_log_loss(p_t, p_s)
                kernel_count += 1

        return total_loss / (kernel_count * len(teacher_maps))

    def _cosine_kernel(self, x):
        sim = torch.matmul(x, x.T)  # (B, B)
        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _tstudent_kernel(self, x, d=1.0):
        # Compute pairwise squared distances
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        dist2 = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)
        sim = 1.0 / (1.0 + dist2 / d)
        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss


class RBFKernelDistillation(nn.Module):
    def __init__(self, device, sigma=1.0):
        super().__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)  # (B, C)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)  # (B, C)

            p_t = self._rbf_kernel(t_flat)
            p_s = self._rbf_kernel(s_flat)

            total_loss += self._symmetric_log_loss(p_t, p_s)

        return total_loss / len(teacher_maps)

    def _rbf_kernel(self, x):
        # Compute pairwise squared distances
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        dist2 = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)  # (B, B)

        sim = torch.exp(-dist2 / (2 * self.sigma ** 2))  # RBF kernel
        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)  # Row-wise normalization
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss

class LaplacianKernelDistillation(nn.Module):
    def __init__(self, device, sigma=1.0):
        super().__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)

            p_t = self._laplacian_kernel(t_flat)
            p_s = self._laplacian_kernel(s_flat)

            total_loss += self._symmetric_log_loss(p_t, p_s)

        return total_loss / len(teacher_maps)

    def _laplacian_kernel(self, x):
        # Compute pairwise L2 distances (not squared)
        dist = torch.cdist(x, x, p=2)  # (B, B), p=2 is Euclidean norm

        sim = torch.exp(-dist / self.sigma)
        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss



class PolynomialKernelDistillation(nn.Module):
    def __init__(self, device, degree=2, coef0=1.0):
        super().__init__()
        self.device = device
        self.degree = degree
        self.coef0 = coef0  # c in the formula

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)

            p_t = self._polynomial_kernel(t_flat)
            p_s = self._polynomial_kernel(s_flat)

            total_loss += self._symmetric_log_loss(p_t, p_s)

        return total_loss / len(teacher_maps)

    def _polynomial_kernel(self, x):
        sim = torch.matmul(x, x.T)  # (B, B)
        sim = (sim + self.coef0).pow(self.degree)
        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss


class SigmoidKernelDistillation(nn.Module):
    def __init__(self, device, alpha=1.0, coef0=1.0):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.coef0 = coef0

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)

            p_t = self._sigmoid_kernel(t_flat)
            p_s = self._sigmoid_kernel(s_flat)

            total_loss += self._symmetric_log_loss(p_t, p_s)

        return total_loss / len(teacher_maps)

    def _sigmoid_kernel(self, x):
        sim = torch.matmul(x, x.T)  # (B, B)
        sim = torch.tanh(self.alpha * sim + self.coef0)
        sim.fill_diagonal_(0)
        sim = sim - sim.min(dim=1, keepdim=True)[0]
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss



class CauchyKernelDistillation(nn.Module):
    def __init__(self, device, sigma=1.0):
        super().__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)

            p_t = self._cauchy_kernel(t_flat)
            p_s = self._cauchy_kernel(s_flat)

            total_loss += self._symmetric_log_loss(p_t, p_s)

        return total_loss / len(teacher_maps)

    def _cauchy_kernel(self, x):
        # Compute pairwise squared Euclidean distances
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        dist2 = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)  # (B, B)

        sim = 1.0 / (1.0 + dist2 / (self.sigma ** 2))
        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss


class IMQKernelDistillation(nn.Module):
    def __init__(self, device, c=1.0):
        super().__init__()
        self.device = device
        self.c = c

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)

            p_t = self._imq_kernel(t_flat)
            p_s = self._imq_kernel(s_flat)

            total_loss += self._symmetric_log_loss(p_t, p_s)

        return total_loss / len(teacher_maps)

    def _imq_kernel(self, x):
        # Pairwise squared Euclidean distance
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        dist2 = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)  # (B, B)

        sim = 1.0 / torch.sqrt(dist2 + self.c ** 2)
        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss


class ExponentialKernelDistillation(nn.Module):
    def __init__(self, device, sigma=1.0):
        super().__init__()
        self.device = device
        self.sigma = sigma

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)

            p_t = self._exp_kernel(t_flat)
            p_s = self._exp_kernel(s_flat)

            total_loss += self._symmetric_log_loss(p_t, p_s)

        return total_loss / len(teacher_maps)

    def _exp_kernel(self, x):
        dist = torch.cdist(x, x, p=2)  # (B, B), Euclidean distance
        sim = torch.exp(-dist / (2 * self.sigma))
        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss








class UnifiedKernelDistillation(nn.Module):
    def __init__(self, device, kernel_type='cosine', **kwargs):
        super().__init__()
        self.device = device
        self.kernel_type = kernel_type.lower()
        self.kernel_params = kwargs

    def forward(self, teacher_maps, student_maps):
        assert len(teacher_maps) == len(student_maps), "Mismatch in number of layers"
        total_loss = 0.0

        for t_map, s_map in zip(teacher_maps, student_maps):
            B, C, H, W = t_map.shape
            t_flat = t_map.view(B, C, -1).mean(dim=2).to(self.device)
            s_flat = s_map.view(B, C, -1).mean(dim=2).to(self.device)

            if self.kernel_type == 'cosine':
                t_flat = F.normalize(t_flat, dim=1)
                s_flat = F.normalize(s_flat, dim=1)

            p_t = self._compute_kernel(t_flat)
            p_s = self._compute_kernel(s_flat)

            total_loss += self._symmetric_log_loss(p_t, p_s)

        return total_loss / len(teacher_maps)

    def _compute_kernel(self, x):
        k = self.kernel_type

        if k == 'cosine':
            sim = torch.matmul(x, x.T)
        elif k == 'student-t':
            d = self.kernel_params.get('d', 1.0)
            x_norm = (x ** 2).sum(dim=1, keepdim=True)
            dist2 = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)
            sim = 1.0 / (1.0 + dist2 / d)
        elif k == 'rbf':
            sigma = self.kernel_params.get('sigma', 1.0)
            x_norm = (x ** 2).sum(dim=1, keepdim=True)
            dist2 = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)
            sim = torch.exp(-dist2 / (2 * sigma ** 2))
        elif k == 'laplacian':
            sigma = self.kernel_params.get('sigma', 1.0)
            dist = torch.cdist(x, x, p=2)
            sim = torch.exp(-dist / sigma)
        elif k == 'polynomial':
            degree = self.kernel_params.get('degree', 2)
            coef0 = self.kernel_params.get('coef0', 1.0)
            sim = (torch.matmul(x, x.T) + coef0).pow(degree)
        elif k == 'sigmoid':
            alpha = self.kernel_params.get('alpha', 1.0)
            coef0 = self.kernel_params.get('coef0', 1.0)
            sim = torch.tanh(alpha * torch.matmul(x, x.T) + coef0)
            sim = sim - sim.min(dim=1, keepdim=True)[0]  # shift to non-negative
        elif k == 'cauchy':
            sigma = self.kernel_params.get('sigma', 1.0)
            x_norm = (x ** 2).sum(dim=1, keepdim=True)
            dist2 = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)
            sim = 1.0 / (1.0 + dist2 / (sigma ** 2))
        elif k == 'imq':
            c = self.kernel_params.get('c', 1.0)
            x_norm = (x ** 2).sum(dim=1, keepdim=True)
            dist2 = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)
            sim = 1.0 / torch.sqrt(dist2 + c ** 2)
        elif k == 'exponential':
            sigma = self.kernel_params.get('sigma', 1.0)
            dist = torch.cdist(x, x, p=2)
            sim = torch.exp(-dist / (2 * sigma))
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

        sim.fill_diagonal_(0)
        sim = sim.clamp(min=1e-6)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def _symmetric_log_loss(self, p_teacher, p_student):
        p_teacher = p_teacher.clamp(min=1e-6)
        p_student = p_student.clamp(min=1e-6)
        diff = p_teacher - p_student
        log_diff = torch.log(p_teacher) - torch.log(p_student)
        loss = (diff * log_diff).sum()
        return loss
