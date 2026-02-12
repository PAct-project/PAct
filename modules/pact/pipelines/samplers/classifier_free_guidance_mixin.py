from typing import *
import torch


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        if 'masks' in kwargs:
            kwargs['masks'] = torch.zeros_like(kwargs['masks'])
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred
class ClassifierFreeGuidanceSamplerMixinArticulation:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred_output = super()._inference_model(model, x_t, t, cond, **kwargs)
        pred, pred_arti = pred_output if isinstance(pred_output, tuple) else (pred_output, None)
        
        if 'masks' in kwargs:
            kwargs['masks'] = torch.zeros_like(kwargs['masks'])
        neg_pred_output = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        neg_pred, neg_pred_arti = neg_pred_output if isinstance(neg_pred_output, tuple) else (neg_pred_output, None)
        if hasattr(self, 'arti_info_diffusion') and self.arti_info_diffusion:
            return  (1 + cfg_strength) * pred - cfg_strength * neg_pred, \
                        (1 + cfg_strength) * pred_arti - cfg_strength * neg_pred_arti 
                        
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred, pred_arti

