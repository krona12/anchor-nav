# SAM ViT-H checkpoint

Put Meta Segment Anything checkpoints here when running
`hm3d-online/refhm3d-nav-sequence-baseline-sam.py`.

The SAM baseline defaults to ViT-H:

```bash
wget -O hm3d-online/SAM/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

If ViT-H runs out of GPU memory, lower the per-batch point count:

```bash
SAM_POINTS_PER_BATCH=32 python hm3d-online/refhm3d-nav-sequence-baseline-sam.py
```

or point to any checkpoint explicitly:

```bash
SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth python hm3d-online/refhm3d-nav-sequence-baseline-sam.py
```
