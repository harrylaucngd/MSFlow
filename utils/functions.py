import torch


def transfer_weights(uncond_model, cond_model, freeze_pretrained=False):
    with torch.no_grad():
        cond_model.model.tok_emb.weight.copy_(uncond_model.tok_emb.weight)
        cond_model.model.pos_emb.weight.copy_(uncond_model.pos_emb.weight)
        cond_model.model.time_emb.weight.copy_(uncond_model.time_emb.weight)
        cond_model.model.encoder.load_state_dict(uncond_model.encoder.state_dict())
        cond_model.model.lm_head.weight.copy_(uncond_model.lm_head.weight)



def transfer_weights_with_adaptive_ln(uncond_model, cond_model, freeze_pretrained=False):
    """
    Transfer pretrained weights and initialize AdaptiveLayerNorm from pretrained LayerNorm.
    """
    # 1️⃣ Embeddings
    cond_model.tok_emb.weight.data.copy_(uncond_model.tok_emb.weight)
    cond_model.pos_emb.weight.data.copy_(uncond_model.pos_emb.weight)

    # Time embedding if shapes match
    if cond_model.time_emb.weight.shape == uncond_model.time_emb.weight.shape:
        cond_model.time_emb.weight.data.copy_(uncond_model.time_emb.weight)
        if hasattr(cond_model.time_emb, 'bias') and hasattr(uncond_model.time_emb, 'bias'):
            cond_model.time_emb.bias.data.copy_(uncond_model.time_emb.bias)

    # 2️⃣ Encoder layers
    for l_pre, l_cond in zip(uncond_model.encoder.layers, cond_model.encoder.layers):
        # Multihead attention
        try:
            l_cond.self_attn.in_proj_weight.data.copy_(l_pre.self_attn.in_proj_weight)
            l_cond.self_attn.in_proj_bias.data.copy_(l_pre.self_attn.in_proj_bias)
            l_cond.self_attn.out_proj.weight.data.copy_(l_pre.self_attn.out_proj.weight)
            l_cond.self_attn.out_proj.bias.data.copy_(l_pre.self_attn.out_proj.bias)
        except Exception as e:
            print("⚠️ Skipping attention weights for a layer:", e)

        # Feedforward layers: can only copy if dimensions match
        if l_cond.linear1.weight.shape == l_pre.linear1.weight.shape:
            l_cond.linear1.weight.data.copy_(l_pre.linear1.weight)
            l_cond.linear1.bias.data.copy_(l_pre.linear1.bias)
        if l_cond.linear2.weight.shape == l_pre.linear2.weight.shape:
            l_cond.linear2.weight.data.copy_(l_pre.linear2.weight)
            l_cond.linear2.bias.data.copy_(l_pre.linear2.bias)

        # Initialize AdaptiveLayerNorm from pretrained LayerNorm
        if hasattr(l_pre, 'norm1') and hasattr(l_cond, 'norm1'):
            l_cond.norm1.ln.weight.data.copy_(l_pre.norm1.weight)
            l_cond.norm1.ln.bias.data.copy_(l_pre.norm1.bias)
        if hasattr(l_pre, 'norm2') and hasattr(l_cond, 'norm2'):
            l_cond.norm2.ln.weight.data.copy_(l_pre.norm2.weight)
            l_cond.norm2.ln.bias.data.copy_(l_pre.norm2.bias)

        # Optional: zero the adaptive MLP to start with identity
        # if hasattr(l_cond.norm1, 'mlp') and l_cond.norm1.mlp is not None:
        #     for m in l_cond.norm1.mlp[-1]:
        #         if isinstance(m, torch.nn.Linear):
        #             torch.nn.init.zeros_(m.weight)
        #             torch.nn.init.zeros_(m.bias)
        # if hasattr(l_cond.ln.norm2, 'mlp') and l_cond.norm2.mlp is not None:
        #     for m in l_cond.norm2.mlp[-1]:
        #         if isinstance(m, torch.nn.Linear):
        #             torch.nn.init.zeros_(m.weight)
        #             torch.nn.init.zeros_(m.bias)

    # 3️⃣ LM head
    try:
        cond_model.lm_head.weight.data.copy_(uncond_model.lm_head.weight)
    except Exception as e:
        print("⚠️ Skipping LM head copy:", e)

    # 4️⃣ Freeze pretrained layers if requested
    if freeze_pretrained:
        for name, param in cond_model.named_parameters():
            # Keep conditional adapters and lm_head trainable
            if ('cond' in name) or ('lm_head' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False


    print("✅ Transfer completed. AdaptiveLayerNorm initialized from pretrained LayerNorm.")