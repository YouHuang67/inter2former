import time
from typing import Union, List, Tuple

import numpy as np
import torch
from engine.utils import rearrange, reduce, repeat

# Click modes
CLK_POSITIVE = "positive"
CLK_NEGATIVE = "negative"

# Reference label values
REF_DEFINITELY_BACKGROUND = 0
REF_POSSIBLY_BACKGROUND = 1
REF_UNKNOWN = 2
REF_POSSIBLY_FOREGROUND = 3
REF_DEFINITELY_FOREGROUND = 4

REF_MODES = [
    REF_DEFINITELY_BACKGROUND,
    REF_POSSIBLY_BACKGROUND,
    REF_UNKNOWN,
    REF_POSSIBLY_FOREGROUND,
    REF_DEFINITELY_FOREGROUND,
]

REF_INVERSE_MODES = {
    REF_POSSIBLY_BACKGROUND: REF_POSSIBLY_FOREGROUND,
    REF_POSSIBLY_FOREGROUND: REF_POSSIBLY_BACKGROUND,
}


def fast_points_to_ref_label(label: Union[torch.Tensor, np.ndarray],
                             points: List[Tuple[int, int, str]],
                             inner_radius: float,
                             outer_radius: float
                             ) -> Union[torch.Tensor, np.ndarray]:
    """
    Vectorized ref label generation.
    1. Build coordinate grid.
    2. Compute min distances from pixels to clicks.
    3. Mark outer and inner regions.
    4. Resolve conflicts.
    Supports numpy and torch (CPU/GPU).
    """
    if not isinstance(label, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Cannot handle type: {type(label)}")
    if label.ndim != 2:
        raise ValueError("Label must be 2D, got shape " + str(label.shape))
    use_torch = isinstance(label, torch.Tensor)
    H, W = label.shape

    if use_torch:
        device = label.device
        out = torch.full((H, W), REF_UNKNOWN, dtype=torch.int64,
                         device=device)
        Y = torch.arange(H, device=device).view(H, 1).float()
        X = torch.arange(W, device=device).view(1, W).float()
    else:
        out = np.full((H, W), REF_UNKNOWN, dtype=label.dtype)
        Y, X = np.indices((H, W))
        Y = Y.astype(np.float32)
        X = X.astype(np.float32)

    pos_points = []
    neg_points = []
    pos_first = None
    neg_first = None
    for i, (y, x, mode) in enumerate(points):
        if mode == CLK_POSITIVE:
            pos_points.append((y, x))
            if pos_first is None:
                pos_first = i
        elif mode == CLK_NEGATIVE:
            neg_points.append((y, x))
            if neg_first is None:
                neg_first = i
        else:
            raise ValueError(f"Invalid mode: {(y, x, mode)}")

    outer_thresh_sq = (inner_radius + outer_radius) ** 2
    inner_thresh_sq = inner_radius ** 2

    if pos_points:
        if use_torch:
            pos_arr = torch.tensor(pos_points, dtype=torch.float32,
                                   device=device)
            pos_y = pos_arr[:, 0].view(-1, 1, 1)
            pos_x = pos_arr[:, 1].view(-1, 1, 1)
            pos_dist = ((Y.view(1, H, 1) - pos_y) ** 2 +
                        (X.view(1, 1, W) - pos_x) ** 2)
            pos_min, _ = torch.min(pos_dist, dim=0)
        else:
            pos_arr = np.array(pos_points, dtype=np.float32)
            pos_y = rearrange(pos_arr[:, 0], "n -> n () ()")
            pos_x = rearrange(pos_arr[:, 1], "n -> n () ()")
            Y_grid = rearrange(Y, "h w -> () h w")
            X_grid = rearrange(X, "h w -> () h w")
            pos_dist = ((Y_grid - pos_y)**2 +
                        (X_grid - pos_x)**2)
            pos_min = np.min(pos_dist, axis=0)
        outer_pos = pos_min <= outer_thresh_sq
        inner_pos = pos_min <= inner_thresh_sq
    else:
        outer_pos = inner_pos = None

    if neg_points:
        if use_torch:
            neg_arr = torch.tensor(neg_points, dtype=torch.float32,
                                   device=device)
            neg_y = neg_arr[:, 0].view(-1, 1, 1)
            neg_x = neg_arr[:, 1].view(-1, 1, 1)
            neg_dist = ((Y.view(1, H, 1) - neg_y) ** 2 +
                        (X.view(1, 1, W) - neg_x) ** 2)
            neg_min, _ = torch.min(neg_dist, dim=0)
        else:
            neg_arr = np.array(neg_points, dtype=np.float32)
            neg_y = rearrange(neg_arr[:, 0], "n -> n () ()")
            neg_x = rearrange(neg_arr[:, 1], "n -> n () ()")
            Y_grid = rearrange(Y, "h w -> () h w")
            X_grid = rearrange(X, "h w -> () h w")
            neg_dist = ((Y_grid - neg_y)**2 +
                        (X_grid - neg_x)**2)
            neg_min = np.min(neg_dist, axis=0)
        outer_neg = neg_min <= outer_thresh_sq
        inner_neg = neg_min <= inner_thresh_sq
    else:
        outer_neg = inner_neg = None

    if outer_pos is not None:
        out[outer_pos] = REF_POSSIBLY_FOREGROUND
    if outer_neg is not None:
        out[outer_neg] = REF_POSSIBLY_BACKGROUND
    if outer_pos is not None and outer_neg is not None:
        conflict = outer_pos & outer_neg
        out[conflict] = REF_UNKNOWN

    inner_order = []
    if inner_pos is not None:
        inner_order.append(("pos", inner_pos, pos_first))
    if inner_neg is not None:
        inner_order.append(("neg", inner_neg, neg_first))
    inner_order.sort(key=lambda x: x[2] if x[2] is not None
                     else float("inf"))
    for typ, mask, _ in inner_order:
        if typ == "pos":
            out[mask] = REF_DEFINITELY_FOREGROUND
        else:
            out[mask] = REF_DEFINITELY_BACKGROUND

    if use_torch:
        return out.to(label.dtype)
    return out.astype(label.dtype)


def fast_point_list_to_ref_labels(
        labels: Union[torch.Tensor, np.ndarray],
        point_list: List[List[Tuple[int, int, str]]],
        inner_radius: float,
        outer_radius: float
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a list of point lists to ref labels using fast vectorized
    computation.

    :param labels: Tensor/array of shape (*, H, W).
    :param point_list: List of point lists; each list contains
                       (y, x, class) tuples.
    :param inner_radius: Inner radius for ref labels.
    :param outer_radius: Outer radius for ref labels.
    :return: Tensor/array of shape (*, H, W).
    """
    if not isinstance(labels, (torch.Tensor, np.ndarray)):
        raise TypeError("Cannot handle type: " + str(type(labels)))
    # Check batch size using the top-left pixel of each image.
    flat_labels = labels[..., 0, 0].reshape((-1,))
    if len(point_list) != len(flat_labels):
        raise ValueError("Mismatch between number of point lists and labels, "
                         f"got {len(point_list)} vs "
                         f"{tuple(labels[..., 0, 0].shape)}")
    ori_labels = labels
    # Reshape labels to (N, H, W) without device transfer.
    flat_shape = (-1,) + labels.shape[-2:]
    labels = labels.reshape(flat_shape)
    ref_labels = []
    for idx, label in enumerate(labels):
        pts = point_list[idx]
        ref = fast_points_to_ref_label(label, pts,
                                       inner_radius, outer_radius)
        ref_labels.append(ref)
    if isinstance(labels, torch.Tensor):
        ref_labels = torch.stack(ref_labels, dim=0)
        ref_labels = ref_labels.view_as(ori_labels)
    else:
        ref_labels = np.stack(ref_labels, axis=0)
        ref_labels = ref_labels.reshape(ori_labels.shape)
    return ref_labels


if __name__ == "__main__":
    from inter2former.utils import points_to_ref_label
    # Numerical tests
    H, W = 1024, 1024
    np_label = np.zeros((H, W), dtype=np.int32)
    torch_label = torch.zeros((H, W), dtype=torch.int32)

    test_points = [
        (50, 50, CLK_POSITIVE),
        (80, 60, CLK_POSITIVE),
        (150, 150, CLK_NEGATIVE),
        (140, 160, CLK_NEGATIVE),
        (100, 100, CLK_POSITIVE),
        (102, 102, CLK_NEGATIVE),
        (300, 300, CLK_POSITIVE),
        (305, 305, CLK_NEGATIVE),
        (500, 500, CLK_POSITIVE)
    ]
    inner_radius = 5.0
    outer_radius = 0.0

    ref_np = points_to_ref_label(np_label, test_points, inner_radius,
                                 outer_radius)
    ref_torch = points_to_ref_label(torch_label, test_points,
                                    inner_radius, outer_radius)

    fast_ref_np = fast_points_to_ref_label(np_label, test_points,
                                           inner_radius, outer_radius)
    if torch.cuda.is_available():
        torch_label = torch_label.cuda()
    fast_ref_torch = fast_points_to_ref_label(torch_label, test_points,
                                              inner_radius, outer_radius)
    fast_ref_torch = fast_ref_torch.cpu()

    print(f"NumPy numerical test abs diff: "
          f"{float(np.mean(np.abs(ref_np - fast_ref_np).astype(np.float32)))}")
    print(f"Torch numerical test abs diff: "
          f"{float(torch.mean(torch.abs(ref_torch - fast_ref_torch).float()).item())}")

    # Performance tests
    ITER = 100
    t0 = time.time()
    for _ in range(ITER):
        _ = points_to_ref_label(np_label, test_points,
                                inner_radius, outer_radius)
    t_orig = time.time() - t0

    t0 = time.time()
    for _ in range(ITER):
        _ = fast_points_to_ref_label(np_label, test_points,
                                     inner_radius, outer_radius)
    t_fast = time.time() - t0

    print("Orig NumPy avg time: {:.3f} ms".format(t_orig / ITER * 1e3))
    print("Fast NumPy avg time: {:.3f} ms".format(t_fast / ITER * 1e3))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_label_gpu = torch.zeros((H, W), dtype=torch.int32,
                                      device=device)
        _ = fast_points_to_ref_label(torch_label_gpu, test_points,
                                     inner_radius, outer_radius)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(ITER):
            _ = fast_points_to_ref_label(torch_label_gpu, test_points,
                                         inner_radius, outer_radius)
        torch.cuda.synchronize()
        t_fast_gpu = time.time() - t0
        print("Fast Torch GPU avg time: {:.3f} ms".format(t_fast_gpu / ITER * 1e3))

    # Add batch processing tests
    batch_points_list = [
        [  # First sample
            (50, 50, CLK_POSITIVE),
            (80, 60, CLK_POSITIVE),
            (150, 150, CLK_NEGATIVE)
        ],
        [  # Second sample
            (200, 200, CLK_NEGATIVE),
            (220, 220, CLK_POSITIVE)
        ]
    ]

    # Test NumPy batch consistency
    np_batch_labels = np.zeros((2, H, W), dtype=np.int32)
    batch_ref_np = fast_point_list_to_ref_labels(np_batch_labels,
                                                 batch_points_list,
                                                 inner_radius, outer_radius)
    # Verify against single processing
    for i in range(2):
        single_ref = fast_points_to_ref_label(np_batch_labels[i],
                                              batch_points_list[i],
                                              inner_radius, outer_radius)
        assert np.allclose(batch_ref_np[i],
                           single_ref), f"Batch mismatch at index {i}"

    # Test Torch GPU if available
    if torch.cuda.is_available():
        torch_batch = torch.zeros((2, H, W), dtype=torch.int32).cuda()
        batch_ref_torch = fast_point_list_to_ref_labels(torch_batch,
                                                        batch_points_list,
                                                        inner_radius,
                                                        outer_radius)
        # Shape check
        assert batch_ref_torch.shape == torch_batch.shape, f"Shape error {batch_ref_torch.shape} vs {torch_batch.shape}"

    # Print final verification
    print("Batch processing verification passed")

