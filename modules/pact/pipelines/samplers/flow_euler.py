"""
Flow Euler Samplers for Generative Models

This file implements samplers for flow-matching generative models using the Euler integration method.
It contains three main sampler classes:
1. FlowEulerSampler: Base implementation of Euler sampling for flow-matching models
2. FlowEulerCfgSampler: Adds classifier-free guidance to the Euler sampler
3. FlowEulerGuidanceIntervalSampler: Enhances the sampler with both classifier-free guidance and guidance intervals

Flow-matching models define continuous paths from noise to data, and these samplers implement
ODE solvers (specifically Euler method) to follow these paths and generate samples.
"""

from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import (
    ClassifierFreeGuidanceSamplerMixin,
    ClassifierFreeGuidanceSamplerMixinArticulation,
)
from .guidance_interval_mixin import (
    GuidanceIntervalSamplerMixin,
    GuidanceIntervalSamplerArticulationMixin,
)


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """

    def __init__(
        self,
        sigma_min: float,
    ):
        # sigma_min controls the minimum noise level in the flow
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        """
        Convert noise prediction (epsilon) to predicted clean data (x_0).

        Args:
            x_t: Current noisy tensor at timestep t
            t: Current timestep
            eps: Predicted noise

        Returns:
            Predicted clean data x_0
        """
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        """
        Convert predicted clean data (x_0) to noise prediction (epsilon).

        Args:
            x_t: Current noisy tensor at timestep t
            t: Current timestep
            x_0: Predicted clean data

        Returns:
            Implied noise prediction epsilon
        """
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        """
        Convert velocity prediction (v) to predicted clean data (x_0) and noise (epsilon).

        Args:
            x_t: Current noisy tensor at timestep t
            t: Current timestep
            v: Predicted velocity

        Returns:
            Tuple of (x_0, epsilon) derived from velocity
        """
        # print("V2:", x_t.shape, v.shape)
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (
            self.sigma_min + (1 - self.sigma_min) * t
        ) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        """
        Run inference with the model.

        Args:
            model: The flow model
            x_t: Current noisy tensor at timestep t
            t: Current timestep (will be scaled by 1000)
            cond: Conditional information
            kwargs: Additional arguments for model

        Returns:
            Model's predicted velocity
        """
        # Scale timestep by 1000 for model input
        t = torch.tensor(
            [1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32
        )
        # Broadcast single condition to match batch size if needed
        # print(f"cond shape: {cond.shape}")
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        # print(f"cond shape after repeat: {cond.shape}")
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        """
        Get model predictions and convert to various formats.

        Args:
            model: The flow model
            x_t: Current noisy tensor at timestep t
            t: Current timestep
            cond: Conditional information
            kwargs: Additional arguments for model

        Returns:
            Tuple of (x_0, epsilon, velocity) predictions
        """
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, cond: Optional[Any] = None, **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.

        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        # Get model predictions
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(
            model, x_t, t, cond, **kwargs
        )
        # Euler step: x_{t-1} = x_t - (t - t_prev) * v_t
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        # Create a linearly spaced timestep sequence from 1 to 0
        t_seq = np.linspace(1, 0, steps + 1)
        # Apply rescaling to timesteps if needed
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        # Create pairs of consecutive timesteps
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        # Initialize return dictionary
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        # print(f"shape of cond: {cond.shape}") # shape of cond: torch.Size([4, 1374, 1024])
        # Perform Euler sampling steps
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret


class FlowEulerSamplerArticulation(FlowEulerSampler):

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor(
            [1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32
        )
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        output = model(x_t, t, cond, **kwargs)

        return output

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        output = self._inference_model(model, x_t, t, cond, **kwargs)
        if isinstance(output, tuple) and len(output) == 2:
            pred_v, arti = output
        else:
            pred_v = output
            arti = None
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        if "arti_out_mode" in kwargs and kwargs["arti_out_mode"] == "flow_matching":
            pred_v_arti = arti
            x_t_arti = kwargs["x_t_arti"]
            pred_x_0_arti, pred_eps_arti = self._v_to_xstart_eps(
                x_t=x_t_arti, t=t, v=pred_v_arti
            )
            return (
                pred_x_0,
                pred_eps,
                pred_v,
                (pred_x_0_arti, pred_eps_arti, pred_v_arti),
            )

        return pred_x_0, pred_eps, pred_v, arti

    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, cond: Optional[Any] = None, **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.

        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
            - 'positions': The predicted positions.
        """
        pred_x_0, pred_eps, pred_v, arti = self._get_model_prediction(
            model, x_t, t, cond, **kwargs
        )
        pred_x_prev = x_t - (t - t_prev) * pred_v
        if "arti_out_mode" in kwargs and kwargs["arti_out_mode"] == "flow_matching":
            x_t_arti = kwargs["x_t_arti"]
            (pred_x_0_arti, pred_eps_arti, pred_v_arti) = arti
            pred_x_prev_arti = x_t_arti - (t - t_prev) * pred_v_arti
            return edict(
                {
                    "pred_x_prev": pred_x_prev,
                    "pred_x_0": pred_x_0,
                    "pred_x_prev_arti": pred_x_prev_arti,
                    "pred_x_0_arti": pred_x_0_arti,
                    "arti": pred_x_prev_arti,
                }
            )
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "arti": arti})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
            - 'pred_arti_t': a list of predicted positions at each timestep.
            - 'arti': the final predicted positions.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict(
            {
                "samples": None,
                "pred_x_t": [],
                "pred_x_0": [],
                "pred_arti_t": [],
                "arti": None,
            }
        )
        sample_arti = None
        if "arti_out_mode" in kwargs and kwargs["arti_out_mode"] == "flow_matching":
            sample_arti = kwargs["noise_arti"]
            kwargs.pop("noise_arti")
            kwargs["x_t_arti"] = sample_arti
            ret.pred_x_t_arti = []
            ret.pred_x_0_arti = []
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            if "arti_out_mode" in kwargs and kwargs["arti_out_mode"] == "flow_matching":
                sample_arti = out.pred_x_prev_arti
                kwargs["x_t_arti"] = sample_arti
                ret.pred_x_t_arti.append(out.pred_x_prev_arti)
                ret.pred_x_0_arti.append(out.pred_x_0_arti)

            else:
                ret.pred_arti_t.append(out.arti)
        ret.samples = sample
        if "arti_out_mode" in kwargs and kwargs["arti_out_mode"] == "flow_matching":
            ret.arti = sample_arti
        else:
            ret.arti = ret.pred_arti_t[-1] if ret.pred_arti_t else None

        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.

    This class adds classifier-free guidance to the Euler sampler, enabling conditional
    generation with guidance strength control.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # Call the parent sample method with CFG parameters
        return super().sample(
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            **kwargs,
        )


class FlowEulerCfgSamplerArticulation(
    ClassifierFreeGuidanceSamplerMixinArticulation, FlowEulerSamplerArticulation
):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.

    This class adds classifier-free guidance to the Euler sampler, enabling conditional
    generation with guidance strength control.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # Call the parent sample method with CFG parameters
        return super().sample(
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            **kwargs,
        )


class FlowEulerGuidanceIntervalSampler(
    GuidanceIntervalSamplerMixin, FlowEulerSampler
):  ### the infer pipeline uses this one
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.

    This class extends the Euler sampler with both classifier-free guidance and the ability
    to specify timestep intervals where guidance is applied.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # Call the parent sample method with CFG and interval parameters
        return super().sample(
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            **kwargs,
        )


class FlowEulerGuidanceIntervalSamplerArticulation(
    GuidanceIntervalSamplerArticulationMixin, FlowEulerSamplerArticulation
):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.

    This class extends the Euler sampler with both classifier-free guidance and the ability
    to specify timestep intervals where guidance is applied.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # Call the parent sample method with CFG and interval parameters
        return super().sample(
            model,
            noise,
            cond,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            **kwargs,
        )
