# PPO attacked image export

Each contact sheet contains the same 16 eligible held-out FAKE images. Within each cell, the original is on the left and the attacked image is on the right. Pixel values are shown directly without montage normalization. Full-resolution PNG pairs and per-image metrics are stored in each experiment folder.

- basic_ppo: seven-action policy
- expanded_ppo: ten-action reference policy
- quality_ppo: hard SSIM >= 0.90 policy
