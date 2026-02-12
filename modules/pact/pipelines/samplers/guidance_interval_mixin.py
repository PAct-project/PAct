from typing import *
import torch


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(
        self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs
    ):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            kwargs["is_null_guidance"] = False
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            if "masks" in kwargs:
                kwargs["masks"] = torch.zeros_like(kwargs["masks"])
            kwargs["is_null_guidance"] = True
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            kwargs["is_null_guidance"] = False
            return super()._inference_model(model, x_t, t, cond, **kwargs)


class GuidanceIntervalSamplerArticulationMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(
        self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs
    ):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            kwargs["ordered_mask_dino"] = kwargs["masks"]
            pred_output = super()._inference_model(model, x_t, t, cond, **kwargs)
            pred, pred_arti = (
                pred_output if isinstance(pred_output, tuple) else (pred_output, None)
            )

            # if 'masks' in kwargs:
            kwargs["ordered_mask_dino"] = kwargs[
                "neg_mask"
            ]  ### TODO: clean the code heare
            neg_pred_output = super()._inference_model(
                model, x_t, t, neg_cond, **kwargs
            )
            neg_pred, neg_pred_arti = (
                neg_pred_output
                if isinstance(neg_pred_output, tuple)
                else (neg_pred_output, None)
            )
            if "arti_out_mode" in kwargs and kwargs["arti_out_mode"] == "flow_matching":
                assert (
                    neg_pred_arti is not None and pred_arti is not None
                ), "Articulation prediction missing!"
                return (1 + cfg_strength) * pred - cfg_strength * neg_pred, (
                    1 + cfg_strength
                ) * pred_arti - cfg_strength * neg_pred_arti

            return (1 + cfg_strength) * pred - cfg_strength * neg_pred, pred_arti
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
