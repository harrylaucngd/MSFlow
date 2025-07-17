import torch
from utils import *
from configs import *
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper


class WrappedModel_Cond(ModelWrapper):
    def __init__(self, model, temperature=1.0, guidance_scale=1.0):
        """
        model: your FlowMolBERT_Cond
        temperature: softmax temperature
        guidance_scale: scale for class-free guidance (0 = no guidance)
        """
        super().__init__(model)
        self.temperature = temperature
        self.guidance_scale = guidance_scale

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        """
        x: input tokens [B, L]
        t: timesteps [B]
        """
        cond = extras.get('cond', None)    #cond: condition tensor [B, cond_dim] or None for unconditional

        # If guidance_scale is zero OR cond is None (no conditioning),
        # just return conditional/unconditional output directly:

        if self.guidance_scale == 0.0 or cond is None:
            logits = self.model(x, t, cond=cond)
            return torch.softmax(logits / self.temperature, dim=-1)

        # Otherwise do classifier-free guidance:
        logits_cond = self.model(x, t, cond=cond)
        logits_uncond = self.model(x, t, cond=None)

        guided_logits = logits_uncond + self.guidance_scale * (logits_cond - logits_uncond)

        return torch.softmax(guided_logits / self.temperature, dim=-1)




@torch.no_grad()
def sample_flow_cond(
    num_samples,
    model,
    cond,
    path,
    seq_len,
    vocab_size,
    mask_token_id=0,
    source_distribution="masked",
    steps=128,
    epsilon=1e-3,
    device="cpu",
    return_intermediates=False,
    temperature=1.0,
    guidance_scale = 0.0,
    uncond_prob=0.0,
):
    """
    Conditional sampling wrapper for FlowMolBERT_Cond.

    cond: Tensor of shape [num_samples, cond_dim]
    force_uncond_prob: chance to drop cond for unconditional samples during sampling
    """
    step_size = 1.0 / steps
    time_grid = torch.linspace(0, 1 - epsilon, steps, device=device)

    # Initialize tokens
    if source_distribution == "uniform":
        x_init = torch.randint(size=(num_samples, seq_len), high=vocab_size, device=device)
    elif source_distribution == "masked":
        x_init = torch.full(size=(num_samples, seq_len), fill_value=mask_token_id, device=device)
    else:
        raise NotImplementedError(f"Unknown source_distribution: {source_distribution}")

    # Ensure cond is on the right device and matches batch size
    if cond is not None:
        cond = cond.to(device)
    if cond.size(0) == num_samples:
        print("Sampling 1 mol per condition")
    elif cond.dim() == 1:
        cond = cond.unsqueeze(0)
        cond = cond.repeat(num_samples, 1)
        # assert cond.size(0) == num_samples, "Condition batch size must match num_samples"
    extras = {}
    extras['cond'] = cond
    
    # Wrap model with condition and temperature
    wrapped_model = WrappedModel_Cond(model, temperature=temperature, guidance_scale= guidance_scale)

    # Create solver with wrapped model
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_model,
        path=path,
        vocabulary_size=vocab_size
    )

    samples = solver.sample(
        x_init=x_init,
        step_size=step_size,
        time_grid=time_grid,
        return_intermediates=return_intermediates,
        verbose=True,
        **extras
    )

    return samples.detach().cpu()
