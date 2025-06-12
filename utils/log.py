import torch

def save_checkpoint_every_n_epochs(epoch, step, model_diff, optim_diff, scheduler_diff, model_mfm, optim_mfm, scheduler_mfm, output_path, n=5):
    if (epoch + 1) % n == 0:
        # Save diffusion checkpoint
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state': model_diff.state_dict(),
            'optimizer_state': optim_diff.state_dict(),
            'scheduler_state': scheduler_diff.state_dict()
        }, output_path + f'last_diff.pth')

        # Save MFM checkpoint
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state': model_mfm.state_dict(),
            'optimizer_state': optim_mfm.state_dict(),
            'scheduler_state': scheduler_mfm.state_dict()
        }, output_path + f'last_mfm.pth')

        print(f"[Checkpoint] Saved models at epoch {epoch+1}")
