import torch
import torch.nn as nn
import numpy as np


def _gaussian_1d_kernel(ksize=5, sigma=1.0):
    """Generate a 1D Gaussian kernel."""
    x = np.arange(ksize) - ksize // 2
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    return g.astype(np.float32)


def _sobel_filters():
    """Return 2D Sobel filters (horizontal & vertical)."""
    sf = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    return sf, sf.T


def _directional_filters():
    """Return 8 directional filters for non-max suppression."""
    f0 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float32)
    f45 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    f90 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
    f135 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    f180 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
    f225 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    f270 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    f315 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    return np.stack([f0, f45, f90, f135, f180, f225, f270, f315])


class Canny(nn.Module):
    
    def __init__(self, threshold=2.0):
        super(Canny, self).__init__()
        self.threshold = threshold

        # 1D Gaussian kernel
        g1d = _gaussian_1d_kernel(ksize=5, sigma=1.0)

        # Gaussian conv (horizontal & vertical)
        self.g_h = nn.Conv2d(1, 1, (1, 5), padding=(0, 2), bias=False)
        self.g_v = nn.Conv2d(1, 1, (5, 1), padding=(2, 0), bias=False)

        # Reshape & copy for gh.weight => shape [1,1,1,5]
        self.g_h.weight.data.copy_(
            torch.from_numpy(g1d).reshape(1, 1, 1, 5)
        )
        # Reshape & copy for gv.weight => shape [1,1,5,1]
        self.g_v.weight.data.copy_(
            torch.from_numpy(g1d).reshape(1, 1, 5, 1)
        )

        # Sobel filters
        sfh, sfv = _sobel_filters()
        self.sobh = nn.Conv2d(1, 1, (3, 3), padding=1, bias=False)
        self.sobv = nn.Conv2d(1, 1, (3, 3), padding=1, bias=False)
        self.sobh.weight.data.copy_(
            torch.from_numpy(sfh).unsqueeze(0).unsqueeze(0)
        )
        self.sobv.weight.data.copy_(
            torch.from_numpy(sfv).unsqueeze(0).unsqueeze(0)
        )

        # Directional filters for non-max suppression
        df_all = _directional_filters()
        self.dirf = nn.Conv2d(1, 8, (3, 3), padding=1, bias=False)
        self.dirf.weight.data.copy_(
            torch.from_numpy(df_all[:, None])
        )

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, img):
        """
        params img: (B, 3, H, W)
        return: blurred, grad_mag, grad_ori, thin, thresh, early
        """
        r, g, b = img.chunk(3, dim=1)

        # Gaussian blur
        r_blur = self.g_v(self.g_h(r))
        g_blur = self.g_v(self.g_h(g))
        b_blur = self.g_v(self.g_h(b))
        blurred = torch.cat([r_blur, g_blur, b_blur], dim=1)  # (B, 3, H, W)

        # Gradient
        rx, ry = self.sobh(r_blur), self.sobv(r_blur)
        gx, gy = self.sobh(g_blur), self.sobv(g_blur)
        bx, by = self.sobh(b_blur), self.sobv(b_blur)
        grad_mag = (rx ** 2 + ry ** 2).sqrt() \
                 + (gx ** 2 + gy ** 2).sqrt() \
                 + (bx ** 2 + by ** 2).sqrt()  # (B, 1, H, W)

        # Orientation
        gx_sum = rx + gx + bx
        gy_sum = ry + gy + by
        grad_ori = torch.atan2(gy_sum, gx_sum) * (180.0 / np.pi)
        grad_ori += 180.0
        grad_ori = torch.round(grad_ori / 45.0) * 45.0  # (B, 1, H, W)

        # Non-maximum suppression
        allf = self.dirf(grad_mag)  # (B, 8, H, W)

        pos_idx = ((grad_ori / 45) % 8).squeeze(1).long()  # (B, H, W)
        neg_idx = (((grad_ori / 45) + 4) % 8).squeeze(1).long()  # (B, H, W)

        B, _, H, W = allf.shape

        allf_flat = allf.view(B, -1)  # (B, 8*H*W)

        pixel_idx = torch.arange(H*W, device=img.device).unsqueeze(0)
        pixel_idx = pixel_idx.expand(B, -1)  # (B, H*W)

        pos_idx_flat = pos_idx.view(B, -1)  # (B, H*W)
        neg_idx_flat = neg_idx.view(B, -1)  # (B, H*W)

        i1 = pos_idx_flat * (H * W) + pixel_idx  # (B, H*W)
        i2 = neg_idx_flat * (H * W) + pixel_idx  # (B, H*W)

        pf = torch.gather(allf_flat, dim=1, index=i1)  # (B, H*W)
        nf = torch.gather(allf_flat, dim=1, index=i2)  # (B, H*W)

        pf = pf.view(B, 1, H, W)
        nf = nf.view(B, 1, H, W)

        stacked = torch.stack([pf, nf], dim=1)  # (B, 2, 1, H, W)
        is_max = stacked.min(dim=1).values > 0      # (B, 1, H, W)
        thin = grad_mag.clone()
        thin[~is_max] = 0.0                     # non-max suppressed

        # Threshold
        thresh = thin.clone()
        thresh[thin < self.threshold] = 0.0
        early = grad_mag.clone()
        early[grad_mag < self.threshold] = 0.0

        return blurred, grad_mag, grad_ori, thin, thresh, early


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import data  # noqa

    def _tensor_to_np(t):
        t = t.detach().cpu().numpy()
        if t.shape[0] == 1 and t.ndim == 4:
            t = t.squeeze(0)

        if t.ndim == 3 and t.shape[0] == 1:
            t = t.squeeze(0)

        if t.ndim == 3 and t.shape[0] == 3:
            t = np.transpose(t, (1, 2, 0))

        return t


    def _plot_results(img, blurred, grad_mag, ori, thin, thresh, early):  # noqa
        """Plot and visualize results in a grid."""
        fig, axs = plt.subplots(2, 4, figsize=(12, 6))
        axs[0, 0].imshow(_tensor_to_np(img).clip(0, 1))
        axs[0, 0].set_title("Input")

        axs[0, 1].imshow(_tensor_to_np(blurred[:, 0:1]), cmap='gray')
        axs[0, 1].set_title("Blur(R)")

        axs[0, 2].imshow(_tensor_to_np(grad_mag), cmap='gray')
        axs[0, 2].set_title("Gradient Mag")

        axs[0, 3].imshow(_tensor_to_np(ori), cmap='hsv')
        axs[0, 3].set_title("Orientation")

        axs[1, 0].imshow(_tensor_to_np(thin), cmap='gray')
        axs[1, 0].set_title("Thin Edges")

        axs[1, 1].imshow(_tensor_to_np(thresh), cmap='gray')
        axs[1, 1].set_title("Thresholded")

        axs[1, 2].imshow(_tensor_to_np(early), cmap='gray')
        axs[1, 2].set_title("Early Threshold")

        axs[1, 3].axis('off')

        for row in axs:
            for c in row:
                c.axis('off')
        plt.tight_layout()
        plt.show()

    img_np = data.astronaut()

    img_np = img_np.astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_np, (2, 0, 1)))
    img_t = img_t.unsqueeze(0)

    net = Canny(threshold=2.0)
    blurred, gmag, gori, thin, th, early = net(img_t)
    print(f'blurred range: {blurred.min().item()} - {blurred.max().item()}')
    print(f'grad_mag range: {gmag.min().item()} - {gmag.max().item()}')
    print(f'grad_ori range: {gori.min().item()} - {gori.max().item()}')
    print(f'thin range: {thin.min().item()} - {thin.max().item()}')
    print(f'th range: {th.min().item()} - {th.max().item()}')
    print(f'early range: {early.min().item()} - {early.max().item()}')
    _plot_results(img_t, blurred, gmag, gori, thin, th, early)
