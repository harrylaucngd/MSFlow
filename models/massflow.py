import torch
import torch.nn as nn
from DiffMS.src.mist.models.spectra_encoder import SpectraEncoderGrowing
from modules.cond_lit_model import CondFlowMolBERTLitModule

class SpectraEncoderModule(nn.Module):
    """Wrapper for pretrained SpectraEncoderGrowing."""
    def __init__(self, checkpoint_path=None, hidden_dim=256, magma_modulo=512):
        super().__init__()
        self.encoder = SpectraEncoderGrowing(
            inten_transform='float',
            inten_prob=0.1,
            remove_prob=0.5,
            peak_attn_layers=2,
            num_heads=8,
            pairwise_featurization=True,
            embed_instrument=False,
            cls_type='ms1',
            set_pooling='cls',
            spec_features='peakformula',
            mol_features='fingerprint',
            form_embedder='pos-cos',
            output_size=4096,
            hidden_size=hidden_dim,
            spectra_dropout=0.1,
            top_layers=1,
            refine_layers=4,
            magma_modulo=magma_modulo,
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location='cuda')
            self.encoder.load_state_dict(state_dict)

    def forward(self, batch):
        out, aux = self.encoder(batch)
        return out, aux


class SpectraFlowModel(nn.Module):
    """
    Combines SpectraEncoder outputs with Flow model.
    Assumes CondFlowMolBERTLitModule takes encoder output as `cond` input.
    """
    def __init__(self, encoder_ckpt=None, encoder_hidden_dim=256, encoder_magma_modulo=512,
                 flow_config=None):
        super().__init__()
        self.encoder_module = SpectraEncoderModule(
            checkpoint_path=encoder_ckpt,
            hidden_dim=encoder_hidden_dim,
            magma_modulo=encoder_magma_modulo
        )

        # Initialize Flow model
        self.flow_model = CondFlowMolBERTLitModule(**flow_config)

    def forward(self, batch, labels=None):
        """
        batch: spectra input batch for encoder
        labels: optional target for flow model
        """
        encoder_out, aux = self.encoder_module(batch)
        
        # Feed encoder output as conditional input to flow model
        flow_out = self.flow_model(encoder_out, labels=labels)
        return flow_out, encoder_out, aux
