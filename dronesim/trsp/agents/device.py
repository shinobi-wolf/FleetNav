import torch


def get_device() -> torch.device:
    """Return the CUDA device only if it can actually execute a kernel.

    `torch.cuda.is_available()` is True even when the installed PyTorch
    build has no compiled kernels for the GPU's compute capability (e.g.
    older Maxwell/Pascal cards vs. current PyTorch wheels), which raises
    `no kernel image is available for execution on the device` at the
    first real op. Probing with a tiny op catches that up front so
    training falls back to CPU instead of crashing mid-run.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        torch.zeros(1, device='cuda') + 1
        return torch.device('cuda')
    except RuntimeError:
        return torch.device('cpu')
