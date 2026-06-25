

import gc
import json
import os
from pathlib import Path

import numpy as np
import quaternion
from sklearn.cluster import KMeans
import torch
from torch_scatter import scatter_mean
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
import cv2
from FastSAM.fastsam import FastSAM
from data.datasets.constant import PromptType
from model.embodied_pq3d_instseg import EmbodiedPQ3DInstSegModel
from model.query3d_vle import Query3DVLE
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import open3d as o3d
import torch.nn.functional as F
import albumentations as A
import MinkowskiEngine as ME
from data.datasets.embodied_instseg_wrapper import EmbodiedInstSegDatasetWrapper
from data.data_utils import pad_sequence
from torch.utils.data import default_collate
from merge_utils import RepresentationManager
import time

# 与仓库内 hm3d-online/FastSAM/FastSAM-x.pt 对齐，不依赖 cwd，避免误用相对路径导致加载失败或走网络
_FASTSAM_WEIGHT = Path(__file__).resolve().parent / "FastSAM" / "FastSAM-x.pt"


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)
        end_time = time.time()  # 记录结束时间
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def make_intrinsic_hfov(hfov, aspect_ratio):
    intrinsic = np.eye(4)
    hfov = np.radians(hfov)
    intrinsic[0][0] = 1 / np.tan(hfov / 2.0)
    intrinsic[1][1] = 1 / np.tan(hfov / 2.0) / aspect_ratio
    return intrinsic

def format_result(result):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        annotation['id'] = i
        annotation['segmentation'] = mask.cpu().numpy()
        annotation['bbox'] = result.boxes.data[i]
        annotation['score'] = result.boxes.conf[i]
        annotation['area'] = annotation['segmentation'].sum()
        annotations.append(annotation)
    return annotations

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]
    
def convert_from_uvd(u, v, depth, intr, pose):
    z = depth / 1000.0

    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    padding = np.ones_like(u)
    # padding = -padding
    
    uv = np.concatenate([u,v,-padding], axis=0) * np.expand_dims(z,axis=0)
    xyz = (np.linalg.inv(intr[:3,:3]) @ uv) 
    xyz = np.concatenate([xyz,padding], axis=0)
    xyz = pose @ xyz
    xyz[:3,:] /= xyz[3,:] 
    # import pdb; pdb.set_trace()
    return xyz[:3, :].T

def average_pooling_by_group(img_feat, idxs, grid_idxs, valid_cnt):
    # Get max group ids
    group_ids = torch.from_numpy(idxs).cuda()
    grid = torch.from_numpy(grid_idxs).to(img_feat.dtype).cuda()
    num_groups = torch.max(group_ids) + 1

    img_feat = img_feat.unsqueeze(0)
    feat = F.grid_sample(img_feat, grid[:valid_cnt].unsqueeze(1).unsqueeze(0), mode='bilinear', align_corners=True)
    feat = feat.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()
    feat = torch.cat([feat, torch.zeros(20000-valid_cnt, feat.size(1), device=feat.device, dtype=feat.dtype)], dim=0)

    # Create a tensor to accumulate the sum of values for each group
    pooled_matrix = torch.zeros(num_groups, feat.size(1), device=feat.device, dtype=feat.dtype)

    # Scatter the matrix rows into the pooled matrix based on group ids
    pooled_matrix.index_add_(0, group_ids, feat)

    # Count the number of elements in each group
    group_counts = torch.bincount(group_ids, minlength=num_groups).float()

    group_counts = torch.where(group_counts == 0, torch.ones_like(group_counts), group_counts)

    # Compute the average by dividing by the counts
    pooled_matrix /= group_counts.unsqueeze(1)

    return pooled_matrix.cpu().numpy()

def stage1_collote_fn(batch):
    new_batch = {}

    # sparse collate voxel features
    input_dict = {
        "coords": [sample.pop('voxel_coordinates') for sample in batch], 
        "feats": [sample.pop('voxel_features') for sample in batch],
    }
    voxel_coordinates, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.int32)
    new_batch['voxel_coordinates'] = voxel_coordinates
    new_batch['voxel_features'] = voxel_features
            
    # list collate
    list_keys = ['voxel2segment', 'coordinates', 'voxel_to_full_maps', 'segment_to_full_maps', 'raw_coordinates', 'instance_ids', 'instance_labels', 'instance_boxes', 'instance_ids_ori', 'full_masks', 'segment_masks', 'scan_id', 'segment_labels', 'query_selection_ids', 'instance_hm3d_labels', 'instance_hm3d_text_embeds']
    list_keys = [k for k in list_keys if k in batch[0].keys()]
    for k in list_keys:
        new_batch[k] = [sample.pop(k) for sample in batch]
        
    # pad collate
    padding_keys = ['coord_min', 'coord_max', 'obj_center', 'obj_pad_masks', 'seg_center', 'seg_pad_masks', 'seg_point_count', 'query_locs', 'query_pad_masks',
                    'voxel_seg_pad_masks', 'mv_seg_fts', 'mv_seg_pad_masks', 'pc_seg_fts', 'pc_seg_pad_masks', 'prompt', 'prompt_pad_masks']
    padding_keys =  [k for k in padding_keys if k in batch[0].keys()]
    for k in padding_keys:
        tensors = [sample.pop(k) for sample in batch]
        padded_tensor = pad_sequence(tensors)
        new_batch[k] = padded_tensor
    
    # default collate
    new_batch.update(default_collate(batch))
    return new_batch

def batch_to_cuda(batch):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()  # Query3DSingleFrame inference
        elif isinstance(batch[key], list) and isinstance(batch[key][0], torch.Tensor):
            batch[key] = [tensor.cuda() for tensor in batch[key]]  # Handle list of tensors  with torch.no_grad():
    return batch


def _model_xyz_to_habitat_xyz(p):
    """query_locs / frontier_list 使用 [x, z, y]（与 frontier_list 构造一致），转 Habitat [x, y, z]。"""
    p = np.asarray(p, dtype=np.float64).reshape(-1)[:3]
    return [float(p[0]), float(p[2]), float(p[1])]


def _dump_panorama_sam_views(analysis_output_dir, view_idx, color_rgb_hwc, group_ids_hw, fastsam_masks):
    pano = os.path.join(analysis_output_dir, "panorama")
    os.makedirs(pano, exist_ok=True)

    def _fs_conf(m):
        c = m["score"]
        return float(c.detach().item()) if torch.is_tensor(c) else float(c)

    sam_meta = [{"area": int(m["area"]), "fastsam_conf": _fs_conf(m)} for m in fastsam_masks[:64]]
    with open(os.path.join(pano, f"view_{view_idx:02d}_fastsam_meta.json"), "w", encoding="utf-8") as f:
        json.dump(sam_meta, f, ensure_ascii=False, indent=2)
    cv2.imwrite(
        os.path.join(pano, f"view_{view_idx:02d}_rgb.jpg"),
        cv2.cvtColor(np.ascontiguousarray(color_rgb_hwc), cv2.COLOR_RGB2BGR),
    )
    gid = np.asarray(group_ids_hw, dtype=np.int32)
    mx = int(gid.max()) if gid.size else 0
    rng = np.random.RandomState(view_idx * 10007 + 17)
    pal = rng.randint(30, 255, size=(max(mx + 1, 1), 3), dtype=np.uint8)
    base = color_rgb_hwc.astype(np.float32)
    overlay = base.copy()
    valid = gid >= 0
    if np.any(valid):
        overlay[valid] = base[valid] * 0.42 + pal[gid[valid]].astype(np.float32) * 0.58
    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(pano, f"view_{view_idx:02d}_sam_instance_overlay.jpg"),
        cv2.cvtColor(overlay_u8, cv2.COLOR_RGB2BGR),
    )
    inst_png = np.clip(gid + 1, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(pano, f"view_{view_idx:02d}_sam_instance_ids.png"), inst_png)


def _dump_stage1_per_frame_json(analysis_output_dir, pred_dict_list):
    d = os.path.join(analysis_output_dir, "stage1_per_frame")
    os.makedirs(d, exist_ok=True)
    for fi, p in enumerate(pred_dict_list):
        masks = p.get("pred_masks")
        n_inst = int(masks.shape[1]) if masks is not None and getattr(masks, "ndim", 0) == 2 else 0
        n_pts = int(p["point_cloud"].shape[0]) if p.get("point_cloud") is not None else 0
        rec = {
            "frame_index": fi,
            "n_points": n_pts,
            "n_instances": n_inst,
            "pred_scores": [float(x) for x in np.asarray(p["pred_scores"]).reshape(-1)],
            "pred_classes": [int(x) for x in np.asarray(p["pred_classes"]).reshape(-1)],
            "pred_mask_scores": [float(x) for x in np.asarray(p["pred_mask_scores"]).reshape(-1)],
        }
        with open(os.path.join(d, f"frame_{fi:02d}_detections.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)


def _build_stage2_decision_payload(
    *,
    sentence,
    decision_num,
    frontier_waypoints_habitat,
    query_locs_model,
    query_scores,
    real_obj_pad_masks,
    decision_logits,
    goto_frontier_probability,
    is_object_decision,
    real_object_decision_idx,
    frontier_decision_idx,
    target_position_habitat_xyz,
    n_frontiers,
):
    n_real = int(real_obj_pad_masks.sum().item())
    ql = query_locs_model.detach().cpu().numpy()
    scores = query_scores.detach().cpu().numpy()
    dlog = decision_logits.detach().cpu().numpy().astype(float)
    objs = []
    for i in range(n_real):
        objs.append(
            {
                "slot_index": i,
                "merged_object_score": float(scores[i]),
                "center_habitat_xyz": _model_xyz_to_habitat_xyz(ql[i]),
                "og3d_logit": float(dlog[i]),
            }
        )
    frs = []
    for j in range(n_frontiers):
        gi = n_real + j
        frs.append(
            {
                "frontier_index": j,
                "center_habitat_xyz": _model_xyz_to_habitat_xyz(ql[gi]),
                "og3d_logit": float(dlog[gi]),
            }
        )
    chosen = {
        "branch": "object" if is_object_decision else "frontier",
        "target_habitat_xyz": [float(target_position_habitat_xyz[0]), float(target_position_habitat_xyz[1]), float(target_position_habitat_xyz[2])],
    }
    if is_object_decision:
        chosen["object_slot_index"] = int(real_object_decision_idx)
    else:
        chosen["frontier_argmax_index"] = int(frontier_decision_idx)
    payload = {
        "decision_num": int(decision_num),
        "sentence": sentence,
        "goto_frontier_probability": float(goto_frontier_probability),
        "n_objects_in_memory": n_real,
        "n_frontiers": int(n_frontiers),
        "frontier_waypoints_habitat_xyz": [[float(a[0]), float(a[1]), float(a[2])] for a in frontier_waypoints_habitat],
        "object_candidates": objs,
        "frontier_candidates": frs,
        "chosen": chosen,
    }
    return payload


def _dump_stage2_decision_json(
    analysis_output_dir,
    **kwargs,
):
    payload = _build_stage2_decision_payload(**kwargs)
    out_path = os.path.join(analysis_output_dir, "stage2_decision.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


class PQ3DModel:
    def __init__(self, stage1_dir, stage2_dir, min_decision_num=None):
        # get four models, sam, dino, pq3d stage1, pq3d stage2
        # dino
        dinov2_local_path = '/home/chenlin/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'
        processor = AutoImageProcessor.from_pretrained(dinov2_local_path)
        model = AutoModel.from_pretrained(dinov2_local_path).cuda()
        model.eval()
        img_backbone = [processor, model]
        self.image_backbone = img_backbone
        # sam：默认使用本文件同目录下 FastSAM/FastSAM-x.pt；可用环境变量 FASTSAM_WEIGHT 覆盖
        fastsam_ckpt = Path(os.environ.get("FASTSAM_WEIGHT", str(_FASTSAM_WEIGHT))).expanduser()
        if not fastsam_ckpt.is_file():
            raise FileNotFoundError(
                f"FastSAM 权重未找到: {fastsam_ckpt.resolve()}。"
                "请将 FastSAM-x.pt 放到 hm3d-online/FastSAM/ 或设置环境变量 FASTSAM_WEIGHT。"
            )
        mask_generator = FastSAM(str(fastsam_ckpt))
        self.mask_generator = mask_generator
        # pq3d stage1
        config_path = "../configs/embodied-pq3d-final"
        config_name = "embodied_scan_instseg.yaml"
        GlobalHydra.instance().clear() 
        hydra.initialize(config_path=config_path)
        cfg = hydra.compose(config_name=config_name)
        self.pq3d_stage1 = EmbodiedPQ3DInstSegModel(cfg)
        self.pq3d_stage1.load_state_dict(torch.load(os.path.join(stage1_dir, 'pytorch_model.bin'), map_location='cpu'))
        self.pq3d_stage1.eval()
        self.pq3d_stage1.cuda()
        # merge manager
        self.representation_manager = RepresentationManager()
        # pq3d stage2
        config_path = "../configs/embodied-pq3d-final"
        config_name = "embodied_vle.yaml"
        GlobalHydra.instance().clear() 
        hydra.initialize(config_path=config_path)
        cfg = hydra.compose(config_name=config_name)
        self.pq3d_stage2 = Query3DVLE(cfg)
        self.pq3d_stage2.load_state_dict(torch.load(os.path.join(stage2_dir, 'pytorch_model.bin'), map_location='cpu'), strict=False)
        self.pq3d_stage2.eval()
        self.pq3d_stage2.cuda()
        clip_local_path = '/home/chenlin/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
        self.tokenizer = AutoTokenizer.from_pretrained(clip_local_path)
        # decision params
        self.frontier_selection_mode = 'model'
        self.min_decision_num = min_decision_num if min_decision_num is not None else 3
        self.last_decision_aux = {}
        self.last_stage2_decision = {}

    def reset(self):
        self.representation_manager.reset()
        self.last_decision_aux = {}
        self.last_stage2_decision = {}
        
    def decision(self, color_list, depth_list, agent_state_list, frontier_waypoints, sentence, decision_num, image_feat=None, analysis_output_dir=None, task_level: str = ""):
        torch.cuda.empty_cache()
        gc.collect()  
        torch.cuda.ipc_collect()
        batch_size = 6
        # get image feature
        FEAT_DIM = 1024
        processer = self.image_backbone[0]
        image_backbone = self.image_backbone[1]
        img_feats_list = []
        for i in range(0, len(color_list), batch_size):
            batch_colors = color_list[i:i + batch_size]
            image_inputs = processer(batch_colors, return_tensors="pt").to(image_backbone.device)
            with torch.no_grad():
                outputs = image_backbone(**image_inputs)
            img_feats = outputs.last_hidden_state.detach()
            img_feats = img_feats[:, 1:, :]
            img_feats = img_feats.reshape(-1, 16, 16, FEAT_DIM)
            img_feats = img_feats.permute(0, 3, 1, 2)
            img_feats_list.append(img_feats)
        img_feats = torch.cat(img_feats_list, dim=0)
        torch.cuda.empty_cache()
        # SamFastSAM accepts task_level for per-level profiles; legacy FastSAM
        # rejects unknown YOLO kwargs, so only pass it to the wrapper.
        mask_kwargs = dict(device='cuda', retina_masks=True, imgsz=640, conf=0.1, iou=0.9)
        if hasattr(self.mask_generator, "_level_generators"):
            mask_kwargs["task_level"] = task_level
        everything_result = self.mask_generator(color_list, **mask_kwargs)
        # process to esam format, points, superpoints, img_feat
        points_list = []
        super_points_list = []
        img_feat_list = []
        for idx, (color, depth, agent_state) in enumerate(zip(color_list, depth_list, agent_state_list)):
            # get image feat
            img_feat = img_feats[idx]
            color_hw_rgb = np.ascontiguousarray(color)
            # get sam result, group_ids
            try:
                masks = format_result(everything_result[idx])
            except:
                mask_kwargs = dict(device='cuda', retina_masks=True, imgsz=640, conf=0.1, iou=0.7)
                if hasattr(self.mask_generator, "_level_generators"):
                    mask_kwargs["task_level"] = task_level
                everything_result = self.mask_generator(color, **mask_kwargs)
                masks = format_result(everything_result[0])
            masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
            group_ids = np.full((color.shape[0], color.shape[1]), -1, dtype=int)
            num_masks = len(masks)
            group_counter = 0
            for i in range(num_masks):
                mask_now = masks[i]["segmentation"]
                group_ids[mask_now] = group_counter
                group_counter += 1
            if analysis_output_dir:
                _dump_panorama_sam_views(analysis_output_dir, idx, color_hw_rgb, group_ids, masks)
            # get pose and intrinsic
            sensor_state = agent_state.sensor_states['color_sensor']
            sensor_rot = quaternion.as_rotation_matrix(sensor_state.rotation)
            sensor_pos = sensor_state.position
            pose_mat = np.eye(4)
            pose_mat[:3, :3] = sensor_rot
            pose_mat[:3, 3] = sensor_pos
            intrinsic = make_intrinsic_hfov(42, 640 / 360)     
            # convert depth to point cloud
            depth = depth * 1000
            # get grid_idx, and ww_ind, hh_ind
            height, width = depth.shape[:2]    
            grid_idx = np.stack(np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height)), axis=-1)
            w_ind = np.linspace(-1, 1, width)
            h_ind = np.linspace(1, -1, height)
            ww_ind, hh_ind = np.meshgrid(w_ind, h_ind)
            # reshape
            ww_ind = ww_ind.reshape(-1)
            hh_ind = hh_ind.reshape(-1)
            depth = depth.reshape(-1)
            group_ids = group_ids.reshape(-1)
            color = color.reshape(-1, 3)
            grid_idx = grid_idx.reshape(-1, 2)
            # filter out invalid depth
            valid = np.where(depth > 0)[0]
            invalid_ratio = round((1 - 1.0 * len(valid) / (width * height)) * 100, 2)
            if invalid_ratio > 50:
                continue
            ww_ind = ww_ind[valid]
            hh_ind = hh_ind[valid]
            depth = depth[valid]
            group_ids = group_ids[valid]
            rgb = color[valid]
            grid_idx = grid_idx[valid]
            # get point cloud
            xyz = convert_from_uvd(ww_ind, hh_ind, depth, intrinsic, pose_mat)
            xyz = np.concatenate([xyz, rgb], axis=-1)
            xyz_all = np.concatenate([xyz, group_ids.reshape(-1,1), grid_idx], axis=-1)
            valid_cnt = len(xyz_all)
            # downsample
            if len(xyz_all) >= 20000:
                xyz_all = random_sampling(xyz_all, 20000)
                valid_cnt = 20000
            assert valid_cnt > 1000
            xyz, group_ids, grid_idx = xyz_all[:, :6], xyz_all[:, 6], xyz_all[:, 7:]
            # assign points without group
            points_without_seg = xyz[group_ids == -1]
            if len(points_without_seg) > 0:
                if len(points_without_seg) < 20:
                    other_ins = np.zeros(len(points_without_seg), dtype=np.int64) + group_ids.max() + 1
                else:
                    other_ins = KMeans(n_clusters=20, n_init=10).fit(points_without_seg).labels_ + group_ids.max() + 1
                group_ids[group_ids == -1] = other_ins
            # make group ids continuous
            unique_ids = np.unique(group_ids)
            new_group_ids = np.zeros_like(group_ids)
            for i, ids in enumerate(unique_ids):
                new_group_ids[group_ids == ids] = i
            group_ids = new_group_ids
            group_ids = group_ids.astype(np.int64)
            # pool image feature
            pooled_feat = average_pooling_by_group(img_feat, group_ids, grid_idx, valid_cnt)
            # add to list
            points_list.append(xyz.astype(np.float32))
            super_points_list.append(group_ids.astype(np.int64))
            img_feat_list.append(pooled_feat)
        torch.cuda.empty_cache()
        # pq3d stage1
        # Process img_feat_list, points_list, super_points_list to batched input for Query3DSingleFrame inference
        batch = []
        for points, super_points, img_feat in zip(points_list, super_points_list, img_feat_list):
            # process points
            coordinates = points[:, :3]
            coordinates[:, [1,2]] = coordinates[:, [2,1]] # swap y,z
            # process colors
            color = points[:, 3:]
            color_mean = [0.47793125906962, 0.4303257521323044, 0.3749598901421883]
            color_std = [0.2834475483823543, 0.27566157565723015, 0.27018971370874995]
            normalize_color = A.Normalize(mean=color_mean, std=color_std)
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(normalize_color(image=pseudo_image)["image"])
            # put points and colors to features
            features = np.hstack((color, coordinates))
            features = torch.from_numpy(features).float()
            coordinates = torch.from_numpy(coordinates).float()
            # process segment
            point2seg_id = super_points
            point2seg_id = torch.from_numpy(point2seg_id).long()
            seg_center = scatter_mean(coordinates, point2seg_id, dim=0)
            # voxelize
            voxel_size= 0.02
            voxel_coordinates = np.floor(coordinates / voxel_size)
            _, unique_map, inverse_map = ME.utils.sparse_quantize(coordinates=voxel_coordinates, return_index=True, return_inverse=True)
            voxel_coordinates = voxel_coordinates[unique_map]
            voxel_features = features[unique_map]
            voxel2seg_id = point2seg_id[unique_map]
            # process image feat
            img_feat = torch.from_numpy(img_feat).float()
            data_dict = {
                # for voxel encoder
                'voxel_coordinates': voxel_coordinates,
                'voxel_features': voxel_features,
                "voxel2segment": voxel2seg_id, # list collate
                'coordinates': voxel_features[:, -3:],
                # raw and mapping
                "raw_coordinates":  np.concatenate([coordinates.numpy(), points[:, 3:6]], axis=1), # list collate, numpy
                "coord_min": coordinates.min(0)[0],
                "coord_max": coordinates.max(0)[0],
                "voxel_to_full_maps": inverse_map, # list collate
                "segment_to_full_maps": point2seg_id, # list collate
                # segment info
                "seg_center": seg_center,
                "seg_pad_masks": torch.ones(len(seg_center), dtype=torch.bool),
                # query info
                'query_locs': seg_center.clone(),
                'query_pad_masks': torch.ones(len(seg_center), dtype=torch.bool),
                'query_selection_ids': torch.arange(len(seg_center)), # list collate
                # image feat
                'mv_seg_fts': img_feat,
                'mv_seg_pad_masks': torch.ones(len(seg_center), dtype=torch.bool),  
            }
            batch.append(data_dict)
        batch = stage1_collote_fn(batch)
        batch = batch_to_cuda(batch)
        with torch.no_grad():
            stage1_output_data_dict = self.pq3d_stage1(batch) 
        # get all predictions
        pred_dict_list = []
        pred_masks = stage1_output_data_dict['predictions_mask'][-1] # (B, S, N)
        pred_logits = torch.functional.F.softmax(stage1_output_data_dict['predictions_class'][-1], dim=-1) # ignore last logit (201 for no class), (B, N, 201)
        pred_boxes = stage1_output_data_dict['predictions_box'][-1] # (B, N, 6)
        pred_scores = torch.functional.F.softmax(stage1_output_data_dict['predictions_score'][-1], dim=-1)[:, :, 1] # (B, N)
        pred_query = stage1_output_data_dict['query_feat']
        pred_embeds = stage1_output_data_dict['openvocab_query_feat']
        query_pad_masks = batch['query_pad_masks']
        voxel2segment = batch['voxel2segment']
        voxel_to_full_maps = batch['voxel_to_full_maps']
        segment_to_full_maps = batch['segment_to_full_maps']
        raw_coordinates = batch['raw_coordinates']
        for bid in range(len(pred_masks)):
            masks = pred_masks[bid].detach().cpu()[voxel2segment[bid].cpu()][:, query_pad_masks[bid].cpu()]
            logits = pred_logits[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 201)
            boxes = pred_boxes[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 6)
            scores = pred_scores[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q)
            query = pred_query[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 768)
            embeds = pred_embeds[bid].detach().cpu()[query_pad_masks[bid].cpu()] # (q, 768)
            # filter out wall floor ceiling
            valid_query_mask = ~torch.isin(torch.argmax(logits, dim=-1), torch.tensor([0, 2, 35]))
            masks = masks[:, valid_query_mask]
            logits = logits[valid_query_mask][..., :-1]
            boxes = boxes[valid_query_mask]
            scores = scores[valid_query_mask]
            query = query[valid_query_mask]
            embeds = embeds[valid_query_mask]
            if masks.shape[1] == 0:
                continue
            # get masks and scores
            heatmap = masks.float().sigmoid()
            masks = (masks > 0).float()
            mask_scores = (heatmap * masks).sum(0) / (masks.sum(0) + 1e-6)
            classes = torch.argmax(logits, dim=1)
            # polish mask
            masks = masks.detach().cpu()[voxel_to_full_maps[bid].cpu()]  # full res
            masks = scatter_mean(masks, segment_to_full_maps[bid].cpu(), dim=0)  # full res segments
            masks = (masks > 0.5).float()
            masks = masks.detach().cpu()[segment_to_full_maps[bid].cpu()]  # full res points
            # add to dict
            masks = masks.numpy()
            classes = classes.numpy()
            boxes = boxes.numpy()
            mask_scores = mask_scores.numpy()
            scores = scores.numpy()
            query = query.numpy()
            embeds = embeds.numpy()
            pred_dict_list.append({'point_cloud': raw_coordinates[bid], 'pred_masks': masks, 'pred_classes': classes, 'pred_boxes': boxes, 'pred_scores': scores, 'pred_mask_scores': mask_scores, 'pred_feats': query, 'open_vocab_feats': embeds})
        if analysis_output_dir:
            _dump_stage1_per_frame_json(analysis_output_dir, pred_dict_list)
        # start to merge（传入每帧 RGB 以记录各物体首次检测图像）
        self.representation_manager.merge(pred_dict_list, frame_rgbs=color_list)
        torch.cuda.empty_cache()
        # pq3d stage2
        batch = []
        query_feat = self.representation_manager.object_feat
        query_box = self.representation_manager.object_box
        query_scores = self.representation_manager.object_score
        obj_openvocab_feat = self.representation_manager.open_vocab_feat
        fw_iter = frontier_waypoints if frontier_waypoints is not None else []
        if isinstance(fw_iter, np.ndarray):
            fw_iter = [fw_iter[i] for i in range(len(fw_iter))]
        frontier_habitat_xyz = [np.asarray(fw, dtype=np.float64).reshape(3) for fw in fw_iter]
        frontier_list = [[float(p[0]), float(p[2]), float(p[1])] for p in frontier_habitat_xyz]
        # build object
        obj_boxes = torch.from_numpy(query_box).float()
        obj_locs = obj_boxes.clone()
        obj_scores = torch.from_numpy(query_scores).float()
        obj_pad_masks = torch.ones(len(obj_locs), dtype=torch.bool) # N
        real_obj_pad_masks = torch.ones(len(obj_locs), dtype=torch.bool) # N
        # build segment
        seg_center = obj_locs.clone()
        seg_pad_masks = obj_pad_masks.clone()
        mv_seg_fts = torch.from_numpy(query_feat).float()
        mv_seg_pad_masks = obj_pad_masks.clone()
        vocab_seg_fts = torch.from_numpy(obj_openvocab_feat).float()
        vocab_seg_pad_masks = obj_pad_masks.clone()
        # extend object with frontier
        num_frontiers = len(frontier_list)
        if num_frontiers > 0: 
            frontier_centers = torch.tensor([frontier[:3] for frontier in frontier_list])
            frontier_boxes = torch.cat((frontier_centers, torch.zeros(num_frontiers, 3)), dim=1)
            obj_boxes = torch.cat((obj_boxes, frontier_boxes), dim=0)
            obj_scores = torch.cat((obj_scores, torch.ones(num_frontiers)), dim=0)
            obj_locs = torch.cat((obj_locs, frontier_boxes), dim=0)
            obj_pad_masks = torch.cat((obj_pad_masks, torch.ones(num_frontiers, dtype=torch.bool)), dim=0)
            real_obj_pad_masks = torch.cat((real_obj_pad_masks, torch.zeros(num_frontiers, dtype=torch.bool)), dim=0)
        # build query
        query_locs = obj_locs.clone()
        query_pad_masks = obj_pad_masks.clone()
        query_scores = obj_scores.clone()
        # build pseudu tgt_object_id and obj_labels
        obj_labels = torch.zeros(len(obj_locs), dtype=torch.long)
        tgt_object_id = torch.LongTensor([])
        # build prompt
        encoded_input = self.tokenizer([sentence], add_special_tokens=True, truncation=True)
        tokenized_txt = encoded_input.input_ids[0]
        prompt = torch.FloatTensor(tokenized_txt)
        prompt_pad_masks = torch.ones((len(tokenized_txt))).bool()
        prompt_type = PromptType.TXT
        # build data_dict
        data_dict = {
            # query
            'query_pad_masks': query_pad_masks,
            'query_locs': query_locs,
            "query_scores": query_scores,
            'real_obj_pad_masks': real_obj_pad_masks,
            # segment
            'seg_center': seg_center,
            'seg_pad_masks': seg_pad_masks,
            'mv_seg_fts': mv_seg_fts,
            'mv_seg_pad_masks': mv_seg_pad_masks,
            'vocab_seg_fts': vocab_seg_fts,
            'vocab_seg_pad_masks': vocab_seg_pad_masks,
            # label
            'obj_labels': obj_labels,
            'tgt_object_id': tgt_object_id,
            'decision_label': 1,
            # prompt
            'prompt': prompt,
            'prompt_pad_masks': prompt_pad_masks,
            'prompt_type': prompt_type,
        }
        # replace with image feat
        if image_feat is not None:
            data_dict['prompt'] = image_feat
            data_dict['prompt_pad_masks'] = torch.ones((1)).bool()
            data_dict['prompt_type'] = PromptType.IMAGE
        # collate
        batch.append(data_dict)
        batch = default_collate(batch)
        batch = batch_to_cuda(batch)
        s2_query_scores_cpu = batch["query_scores"][0].detach().cpu()
        # stage2 forward
        with torch.no_grad():
            stage2_output_data_dict = self.pq3d_stage2(batch)
        # convert output to decision
        decision_logits = stage2_output_data_dict['og3d_logits'].detach().cpu()[0]
        real_obj_pad_masks = stage2_output_data_dict['real_obj_pad_masks'].bool().detach().cpu()[0]
        query_locs = stage2_output_data_dict['query_locs'].detach().cpu()[0]
        obj_frontier_decision_logits = stage2_output_data_dict['decision_logits'][0]
        goto_frontier_probability = obj_frontier_decision_logits.softmax(dim=-1).detach().cpu()[0].item()
        # get real and frontier
        real_object_decision_logits = decision_logits[real_obj_pad_masks]
        frontier_decision_logits = decision_logits[~real_obj_pad_masks]
        real_object_decision_idx = torch.argmax(real_object_decision_logits).item()
        if len(frontier_list) > 0:
            frontier_decision_idx = torch.argmax(frontier_decision_logits).item()
        else:
            frontier_decision_idx = 0
        # real_object_decision_prob = real_object_decision_logits[real_object_decision_idx].sigmoid().item()
        # frontier_decision_prob = frontier_decision_logits[frontier_decision_idx].sigmoid().item()
        real_object_locs = query_locs[real_obj_pad_masks]
        frontier_locs = query_locs[~real_obj_pad_masks]
        # get decision
        if (goto_frontier_probability <= 0.5 and decision_num > self.min_decision_num) or len(frontier_list) == 0:
            is_object_decision = True
            target_position = real_object_locs[real_object_decision_idx].numpy()[:3]
        else:
            is_object_decision = False
            # select frontier with highest probability
            if self.frontier_selection_mode == 'model':
                target_position = frontier_locs[frontier_decision_idx].numpy()[:3]
            # Select frontier closest to agent_state[-1]
            elif self.frontier_selection_mode == 'closest':
                agent_position = np.array([agent_state_list[-1].position[0], agent_state_list[-1].position[2], agent_state_list[-1].position[1]])
                distances = np.linalg.norm(frontier_locs.numpy()[:, :3] - agent_position, axis=1)
                closest_frontier_idx = np.argmin(distances)
                target_position = frontier_locs[closest_frontier_idx].numpy()[:3]
            # Select random frontier
            elif self.frontier_selection_mode == 'random':
                random_frontier_idx = np.random.randint(len(frontier_locs))
                target_position = frontier_locs[random_frontier_idx].numpy()[:3]
        target_position[[1, 2]] = target_position[[2, 1]]
        target_habitat_xyz = np.asarray(target_position, dtype=np.float64).reshape(3)
        n_real = int(real_obj_pad_masks.sum().item())
        rdl = real_object_decision_logits.float()
        if rdl.numel() >= 2:
            vals, _ = torch.topk(rdl.flatten(), k=2)
            obj_top1_top2_logit_gap = float(vals[0] - vals[1])
        elif rdl.numel() == 1:
            obj_top1_top2_logit_gap = float("inf")
        else:
            obj_top1_top2_logit_gap = 0.0
        self.last_decision_aux = {
            "goto_frontier_probability": float(goto_frontier_probability),
            "is_object_decision": bool(is_object_decision),
            "real_object_decision_idx": int(real_object_decision_idx),
            "n_real_objects": n_real,
            "object_top1_top2_logit_gap": obj_top1_top2_logit_gap,
        }
        stage2_payload_kwargs = dict(
            sentence=sentence,
            decision_num=decision_num,
            frontier_waypoints_habitat=frontier_habitat_xyz,
            query_locs_model=query_locs,
            query_scores=s2_query_scores_cpu,
            real_obj_pad_masks=real_obj_pad_masks,
            decision_logits=decision_logits,
            goto_frontier_probability=float(goto_frontier_probability),
            is_object_decision=bool(is_object_decision),
            real_object_decision_idx=int(real_object_decision_idx),
            frontier_decision_idx=int(frontier_decision_idx),
            target_position_habitat_xyz=target_habitat_xyz,
            n_frontiers=int(num_frontiers),
        )
        self.last_stage2_decision = _build_stage2_decision_payload(**stage2_payload_kwargs)
        if analysis_output_dir is not None:
            _dump_stage2_decision_json(analysis_output_dir, **stage2_payload_kwargs)
            self.last_decision_aux["analysis_stage2_json"] = os.path.join(analysis_output_dir, "stage2_decision.json")
        return target_position, is_object_decision
