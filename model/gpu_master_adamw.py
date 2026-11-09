"""BF16 working weights with GPU-resident FP32 master weights and AdamW.

ZeRO-1 passes this optimizer only its local working-parameter shard. Native
PyTorch AdamW updates those FP32 masters on the same GPU; no optimizer tensor
is copied to CPU during a step. The public state_dict retains the legacy
master_param/exp_avg/exp_avg_sq keys so existing two-rank checkpoints restore
without losing FP32 precision or changing the partition.
"""
from __future__ import annotations

import copy
from collections import defaultdict

import torch


class GPUMasterAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, foreach=False, fused=True,
                 allow_cpu_for_tests=False):
        if foreach:
            raise ValueError('Use fused AdamW or its scalar reference, not foreach.')
        if lr < 0 or eps < 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError('Invalid AdamW hyperparameters.')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, foreach=False, fused=bool(fused)))
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self.use_fused = bool(fused)
        self._adam = None

    def _check_working_parameter(self, parameter):
        if parameter.dtype != torch.bfloat16:
            raise TypeError('Working parameter must be BF16.')
        if parameter.device.type != 'cuda' and not self.allow_cpu_for_tests:
            raise TypeError('Production master/Adam updates must reside on CUDA GPUs.')

    @torch.no_grad()
    def prime_master(self, parameter, initial_fp32, *, take_ownership=False):
        del take_ownership  # Ownership is always independent of the CPU initializer.
        self._check_working_parameter(parameter)
        if initial_fp32.dtype != torch.float32 or initial_fp32.shape != parameter.shape:
            raise TypeError('Initial FP32 master has the wrong precision or shape.')
        if self.state[parameter] or self._adam is not None:
            raise RuntimeError('Master was already initialized or Adam is active.')
        self.state[parameter]['master_param'] = initial_fp32.detach().to(
            device=parameter.device, dtype=torch.float32, copy=True)

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict would cast FP32 masters/moments to the
        # BF16 working dtype. Restore explicitly instead.
        saved_groups = state_dict['param_groups']
        if len(saved_groups) != len(self.param_groups):
            raise ValueError('Optimizer parameter-group count changed.')
        parameters_by_id = {}
        execution_keys = {'params', 'foreach', 'fused', 'capturable', 'differentiable'}
        for saved, live in zip(saved_groups, self.param_groups):
            if len(saved['params']) != len(live['params']):
                raise ValueError('Optimizer local partition changed.')
            for saved_id, parameter in zip(saved['params'], live['params']):
                parameters_by_id[saved_id] = parameter
            for key, value in saved.items():
                if key not in execution_keys:
                    live[key] = copy.deepcopy(value)
        restored = defaultdict(dict)
        for saved_id, saved in state_dict['state'].items():
            if saved_id not in parameters_by_id:
                raise ValueError('Checkpoint contains an unknown local parameter.')
            parameter = parameters_by_id[saved_id]
            self._check_working_parameter(parameter)
            values = {}
            for key in ('master_param', 'exp_avg', 'exp_avg_sq'):
                value = saved.get(key)
                if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != parameter.shape:
                    raise ValueError(f'Checkpoint {key} is missing or has the wrong precision/shape.')
                values[key] = value.detach().to(parameter.device, dtype=torch.float32, copy=True)
            step = saved.get('step', 0)
            if isinstance(step, torch.Tensor):
                step = step.detach().to(parameter.device, dtype=torch.float32, copy=True).reshape(())
            else:
                step = torch.tensor(float(step), device=parameter.device, dtype=torch.float32)
            values['step'] = step
            restored[parameter] = values
        if set(restored) != set(parameters_by_id.values()):
            raise ValueError('Checkpoint does not contain every local FP32 master.')
        self.state = restored
        self._adam = None

    def _ensure_native_adam(self):
        if self._adam is not None:
            return
        groups = []
        for group in self.param_groups:
            masters = []
            for parameter in group['params']:
                self._check_working_parameter(parameter)
                state = self.state[parameter]
                master = state.get('master_param')
                if master is None:
                    raise RuntimeError('FP32 master was not primed or restored.')
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(master)
                if 'exp_avg_sq' not in state:
                    state['exp_avg_sq'] = torch.zeros_like(master)
                if 'step' not in state:
                    state['step'] = torch.zeros((), device=parameter.device, dtype=torch.float32)
                masters.append(master)
            groups.append(dict(params=masters, lr=group['lr'], betas=group['betas'],
                               eps=group['eps'], weight_decay=group['weight_decay'],
                               fused=self.use_fused, foreach=False,
                               capturable=bool(masters and masters[0].device.type == 'cuda')))
        self._adam = torch.optim.AdamW(groups)
        for group in self.param_groups:
            for parameter in group['params']:
                state = self.state[parameter]
                self._adam.state[state['master_param']] = {
                    key: state[key] for key in ('step', 'exp_avg', 'exp_avg_sq')}

    def assert_gpu_resident(self):
        for group in self.param_groups:
            for parameter in group['params']:
                self._check_working_parameter(parameter)
                for key, value in self.state.get(parameter, {}).items():
                    if isinstance(value, torch.Tensor) and value.device != parameter.device:
                        raise RuntimeError(f'{key} is outside its owning GPU.')
                    if key in ('master_param', 'exp_avg', 'exp_avg_sq') and value.dtype != torch.float32:
                        raise RuntimeError(f'{key} is not FP32.')

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._ensure_native_adam()
        active = []
        for live_group, adam_group in zip(self.param_groups, self._adam.param_groups):
            for key in ('lr', 'betas', 'eps', 'weight_decay'):
                adam_group[key] = live_group[key]
            for parameter in live_group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or gradient.dtype != torch.bfloat16:
                    raise TypeError('Expected dense BF16 working gradients.')
                master = self.state[parameter]['master_param']
                master.grad = gradient.float()
                active.append(parameter)
        try:
            self._adam.step()
            for parameter in active:
                parameter.copy_(self.state[parameter]['master_param'].to(torch.bfloat16))
        finally:
            # FP32 gradients are needed only for the update; they are not
            # retained for the following accumulation window.
            for parameter in active:
                self.state[parameter]['master_param'].grad = None
        return loss
