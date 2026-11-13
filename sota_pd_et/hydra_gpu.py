"""Optional CUDA execution of the already fitted aeon 1.6.0 HYDRA map.

aeon's HYDRA weights and channel indices are ordinary tensor attributes, not
registered module buffers, and its forward method creates CPU accumulators.
Consequently, calling ``module.to('cuda')`` is insufficient.  This adapter
copies the *fitted* weights and indices to the selected device and places its
accumulators there too.  It never fits a map or draws random numbers.

Inputs and convolutions use float32, with TF32 disabled for CUDA convolutions.
CUDA reductions (in particular floating-point scatter additions) can still
produce small numerical differences from CPU execution; bitwise agreement and
identical predictions at a decision boundary are not guaranteed.

The forward computation is adapted from aeon 1.6.0's
``aeon.transformations.collection.convolution_based._hydra._HydraInternal``.
The original implementation is distributed under the following license:

BSD 3-Clause License

Copyright (c) The aeon developers.
Copyright (c) 2022 The sktime developers.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a usable device name.

    ``auto`` chooses CUDA when it is available and CPU otherwise.  Explicit
    ``cuda`` requests fail clearly when PyTorch cannot use CUDA; they never
    silently switch to CPU.  Apple MPS is not a supported backend here.
    """
    import torch

    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of 'auto', 'cpu', or 'cuda'.")
    if requested == "cpu":
        return "cpu"
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot use an NVIDIA GPU. "
            "Select a GPU runtime in Colab, or choose device='cpu' or 'auto'."
        )
    return "cuda" if cuda_available else "cpu"


class HydraDeviceTransform:
    """A device cache for one fitted aeon ``HydraTransformer``.

    Parameters
    ----------
    fitted_transformer
        A fitted aeon 1.6.0 ``HydraTransformer``.  Its CPU tensors remain the
        authoritative portable model state and are never modified.
    device
        ``'auto'``, ``'cpu'``, or ``'cuda'``.  CPU is useful for verification
        against aeon's reference implementation.

    This object is an execution cache.  The owning feature map should omit it
    from serialization and recreate it for the destination machine.  All
    outputs are moved back to CPU before they are returned so the existing
    training-fold scaler and ridge classifier retain their CPU behavior.
    """

    def __init__(self, fitted_transformer, device: str = "cuda"):
        import torch

        self.device = resolve_device(device)
        internal = getattr(fitted_transformer, "_hydra", None)
        if internal is None:
            raise ValueError("Fit the aeon HydraTransformer before creating its device cache.")
        self.k = int(internal.k)
        self.h = int(internal.h)
        self.divisor = int(internal.divisor)
        self.dilations = tuple(int(value.item()) for value in internal.dilations)
        self.paddings = tuple(int(value.item()) for value in internal.paddings)
        # Copy existing values, preserving the exact fitted kernels and channel
        # selection.  In particular, do not instantiate another HYDRA module.
        self.weights = internal.W.detach().to(
            device=self.device, dtype=torch.float32, copy=True
        ).contiguous()
        self.indices = tuple(
            value.detach().to(device=self.device, dtype=torch.long, copy=True)
            for value in internal.idx
        )

    def transform(self, X: np.ndarray):
        """Return one batch's unscaled HYDRA features as a CPU float32 tensor.

        ``X`` has shape ``(participants, channels, timepoints)``.  Batch size
        is controlled by the caller; a CUDA allocation failure is allowed to
        propagate so an explicit GPU request does not silently change device.
        """
        import torch
        import torch.nn.functional as F

        array = np.asarray(X, dtype=np.float32)
        if array.ndim != 3 or min(array.shape) < 1:
            raise ValueError("Expected a nonempty (participants, channels, timepoints) batch.")
        # A copy also handles a read-only array or a view with negative strides.
        tensor = torch.from_numpy(np.array(array, dtype=np.float32, order="C", copy=True))
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        n_examples, n_channels, _ = tensor.shape

        # Deterministic convolution selection and full float32 arithmetic avoid
        # the additional TF32 approximation.  Scatter sums can still differ in
        # their floating-point addition order on CUDA.
        convolution_context = (
            torch.backends.cudnn.flags(
                enabled=True, benchmark=False, deterministic=True, allow_tf32=False
            )
            if self.device == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), convolution_context:
            differenced = torch.diff(tensor) if self.divisor > 1 else None
            features = []
            for dilation_index, (dilation, padding) in enumerate(
                zip(self.dilations, self.paddings)
            ):
                for diff_index in range(self.divisor):
                    series = tensor if diff_index == 0 else differenced
                    weights = self.weights[dilation_index, diff_index]
                    if n_channels > 1:
                        selected = series[:, self.indices[dilation_index][diff_index]].sum(2)
                        responses = F.conv1d(
                            selected, weights, dilation=dilation,
                            padding=padding, groups=self.h,
                        )
                    else:
                        responses = F.conv1d(
                            series, weights, dilation=dilation, padding=padding,
                        )
                    responses = responses.view(n_examples, self.h, self.k, -1)
                    max_values, max_indices = responses.max(2)
                    min_values, min_indices = responses.min(2)
                    count_max = responses.new_zeros(n_examples, self.h, self.k)
                    count_min = responses.new_zeros(n_examples, self.h, self.k)
                    count_max.scatter_add_(-1, max_indices, max_values)
                    count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))
                    features.extend((count_max, count_min))
            return torch.cat(features, 1).view(n_examples, -1).cpu()
