# AnchorNav 代码实现指南（code-instruct.md）

> 本文档是面向实现 Agent 的详细代码指南。在 MTU3D 基础上，通过 plugin_hooks.py 插件机制实现 AnchorNav 三个模块，**不修改** Stage 1 / Stage 2 核心模型。
>
> **代码库根目录**：`D:\DeepLearning\vln\mtu3d\MTU3D\`
> **主要工作目录**：`hm3d-online/`

---

## 0. 文件结构总览

```
hm3d-online/
├── goat-nav.py              ← 修改：插入 5 处钩子调用
├── data_utils.py            ← 修改：新增 clip_text_encoder 属性
├── plugin_hooks.py          ← 修改：填充 stubs 实现
│
├── anchor_nav/              ← 新建目录
│   ├── __init__.py
│   ├── gap.py               ← GAP 模块
│   ├── registry.py          ← Anchor Registry 模块
│   ├── afs.py               ← ADO Phase 1 + Phase 2
│   └── proj_utils.py        ← 3D→2D 投影工具
│
└── vlm/                     ← 新建目录
    ├── __init__.py
    └── client.py            ← Qwen VLM 封装
```

---

## 1. vlm/client.py

**功能**：封装 Qwen VLM（通过阿里云 DashScope OpenAI-compatible API）的文本和视觉调用。

```python
# vlm/client.py
import os
import base64
import json
import numpy as np
import cv2
from openai import OpenAI
from typing import List, Optional, Dict, Any

VLM_API_KEY = "sk-d95ce20c2bcc475a8eb4054bd183307d"
VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLM_MODEL_FAST = "qwen3.5-flash"    # Phase 1 用，速度优先
VLM_MODEL_BEST = "qwen3.5-plus"    # GAP 和 Phase 2 用，精度优先


def _init_client() -> OpenAI:
    return OpenAI(api_key=VLM_API_KEY, base_url=VLM_BASE_URL)


def encode_image_to_base64(img: np.ndarray) -> str:
    """将 numpy RGB 图像编码为 base64 字符串（JPEG）。"""
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def call_vlm_text(prompt: str, model: str = VLM_MODEL_BEST) -> str:
    """纯文本调用（用于 GAP）。"""
    client = _init_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def call_vlm_image(
    images: List[np.ndarray],
    text_prompt: str,
    model: str = VLM_MODEL_FAST,
) -> str:
    """
    多图像 + 文本调用（用于 ADO Phase 1/2）。
    images: List[np.ndarray]，RGB，uint8，shape (H,W,3)
    返回 VLM 的文本响应。
    """
    client = _init_client()
    content = []
    for img in images:
        b64 = encode_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": text_prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content


def parse_json_response(response: str) -> Any:
    """从 VLM 响应中提取 JSON，容错处理代码块包裹。"""
    text = response.strip()
    # 去掉可能的 ```json ... ``` 包裹
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找到第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise
```

---

## 2. anchor_nav/proj_utils.py

**功能**：将 3D 世界坐标点投影到 2D 像素坐标。坐标系与 `data_utils.py` 的 `convert_from_uvd` 保持一致。

```python
# anchor_nav/proj_utils.py
import numpy as np
import quaternion
from typing import Optional, Tuple

IMG_WIDTH = 640
IMG_HEIGHT = 360
HFOV_DEG = 42.0


def make_intrinsic(hfov_deg: float = HFOV_DEG, w: int = IMG_WIDTH, h: int = IMG_HEIGHT) -> np.ndarray:
    """返回 4×4 归一化内参矩阵（与 data_utils.make_intrinsic_hfov 一致）。"""
    intr = np.eye(4)
    hfov = np.radians(hfov_deg)
    aspect = w / h
    intr[0][0] = 1.0 / np.tan(hfov / 2.0)
    intr[1][1] = 1.0 / np.tan(hfov / 2.0) / aspect
    return intr


def get_cam_to_world(agent_state) -> np.ndarray:
    """从 habitat AgentState 获取 color_sensor 的 4×4 相机-to-世界变换矩阵。"""
    sensor_state = agent_state.sensor_states["color_sensor"]
    rot = quaternion.as_rotation_matrix(sensor_state.rotation)
    pos = sensor_state.position
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat


def project_world_to_pixel(
    world_xyz: np.ndarray,   # shape (3,)，世界坐标 [x, y, z]
    agent_state,
    w: int = IMG_WIDTH,
    h: int = IMG_HEIGHT,
) -> Optional[Tuple[int, int]]:
    """
    将世界坐标点投影到像素坐标 (px, py)。
    若点在相机后方或超出图像范围则返回 None。

    坐标系说明：
    - habitat 世界坐标：y 轴朝上
    - data_utils 中深度图的 u/v 是归一化 NDC 坐标，x 向右 [-1,1]，y 向上 [1,-1]
    - habitat 相机坐标：x 右，y 上，-z 为前方（camera looks along -z）
    """
    intr = make_intrinsic(HFOV_DEG, w, h)
    cam_to_world = get_cam_to_world(agent_state)
    world_to_cam = np.linalg.inv(cam_to_world)

    # 世界坐标转相机坐标
    p_world = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    p_cam = world_to_cam @ p_world  # [x_c, y_c, z_c, 1]

    # habitat 相机沿 -z 方向看，因此 z_c < 0 表示在前方
    # data_utils 中使用 -padding（即 z 方向取反），等效于 depth = -z_c
    depth = -p_cam[2]
    if depth <= 0.1:  # 点在相机后方或太近
        return None

    # NDC 坐标（参考 convert_from_uvd 的逆过程）
    u_ndc = (p_cam[0] / depth) * intr[0][0]   # x方向
    v_ndc = (p_cam[1] / depth) * intr[1][1]   # y方向（朝上为正）

    # 转像素（参考 data_utils 中的 linspace：w_ind=[-1,1], h_ind=[1,-1]）
    # u_ndc = px * 2/w - 1  → px = (u_ndc + 1) / 2 * w
    # v_ndc = 1 - py * 2/h  → py = (1 - v_ndc) / 2 * h
    px = int((u_ndc + 1.0) / 2.0 * w)
    py = int((1.0 - v_ndc) / 2.0 * h)

    if 0 <= px < w and 0 <= py < h:
        return (px, py)
    return None


def draw_som_markers(
    image: np.ndarray,              # RGB uint8 (H,W,3)，会被原地修改（传副本）
    positions_2d: list,             # List[Optional[Tuple[int,int]]]，None 表示不可见
    labels: list,                   # List[str or int]，标记文字
    color=(0, 255, 0),
    radius=15,
    font_scale=0.7,
) -> np.ndarray:
    """在图像上绘制 SoM 标记（圆圈+数字），返回标注后的图像。"""
    import cv2
    img = image.copy()
    for pos, label in zip(positions_2d, labels):
        if pos is None:
            continue
        px, py = pos
        cv2.circle(img, (px, py), radius, color, 2)
        cv2.putText(img, str(label), (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    return img
```

---

## 3. anchor_nav/gap.py

**功能**：Goal-Anchor Parser，在 episode 开始时批量解析所有子目标描述。

```python
# anchor_nav/gap.py
import json
from dataclasses import dataclass, field
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vlm.client import call_vlm_text, parse_json_response, VLM_MODEL_BEST


@dataclass
class AnchorItem:
    type: str   # "spatial_obj" | "room" | "area_cue"
    name: str   # 英文名称

@dataclass
class GoalAnchor:
    target: str
    anchors: List[AnchorItem]
    raw_desc: str
    sub_idx: int  # 在 episode 中的索引

    def get_missing_anchors(self, registry) -> List[AnchorItem]:
        """返回 registry 中仍 missing 的锚列表。"""
        return [a for a in self.anchors if registry.get_status(a.name) != "found"]

    def get_found_anchors(self, registry) -> List[AnchorItem]:
        return [a for a in self.anchors if registry.get_status(a.name) == "found"]


BATCH_PROMPT_TEMPLATE = """You are a navigation assistant. Below are {n} sub-goal descriptions (in order) for an indoor navigation task:
{numbered_descs}

For each sub-goal, identify:
- target: the final object to find (English noun, singular)
- anchors: reference objects that help locate the target (NOT the target itself). Each anchor has:
  - type: "spatial_obj" (specific object), "room" (room type), or "area_cue" (visual texture/area feature)
  - name: English name

Return a JSON array of length {n}. Use empty anchors list if none. Example:
[{{"target": "armchair", "anchors": [{{"type": "spatial_obj", "name": "bed"}}, {{"type": "spatial_obj", "name": "balcony door"}}]}}]

Descriptions:
{numbered_descs}
"""


class GoalAnchorParser:
    def parse_all(self, descriptions: List[str]) -> List[GoalAnchor]:
        """
        输入: N 个子目标的自然语言描述列表
        输出: N 个 GoalAnchor 对象

        对于 object 类型（仅 goal_category，无 lang_desc），
        调用方应传入 goal_category 作为描述，GAP 将构造空锚集合。
        """
        if not descriptions:
            return []

        # 检测是否为简单 object 类型（单词，无空格）
        # 若所有描述都是单词，直接构造空锚（无需 VLM）
        if all(len(d.split()) <= 2 for d in descriptions):
            return [
                GoalAnchor(target=d.strip(), anchors=[], raw_desc=d, sub_idx=i)
                for i, d in enumerate(descriptions)
            ]

        numbered = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(descriptions))
        prompt = BATCH_PROMPT_TEMPLATE.format(
            n=len(descriptions),
            numbered_descs=numbered,
        )
        try:
            raw = call_vlm_text(prompt, model=VLM_MODEL_BEST)
            parsed = parse_json_response(raw)
            if not isinstance(parsed, list):
                parsed = [parsed]
        except Exception as e:
            print(f"[GAP] VLM call failed: {e}, using empty anchors fallback")
            parsed = [{"target": d.split()[0], "anchors": []} for d in descriptions]

        result = []
        for i, (item, desc) in enumerate(zip(parsed, descriptions)):
            anchors = [
                AnchorItem(type=a.get("type", "spatial_obj"), name=a.get("name", ""))
                for a in item.get("anchors", [])
                if a.get("name")
            ]
            result.append(GoalAnchor(
                target=item.get("target", desc.split()[0]),
                anchors=anchors,
                raw_desc=desc,
                sub_idx=i,
            ))
        return result
```

---

## 4. anchor_nav/registry.py

**功能**：轻量锚状态注册表，依托 RepresentationManager 的 `open_vocab_feat` 进行 CLIP 相似度匹配。

```python
# anchor_nav/registry.py
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor_nav.gap import AnchorItem, GoalAnchor

# 房间-物体先验（用于 room 类型锚推断）
ROOM_PRIOR = {
    "bedroom": ["bed", "wardrobe", "nightstand", "pillow", "lamp"],
    "kitchen": ["stove", "fridge", "refrigerator", "sink", "oven", "microwave"],
    "living_room": ["sofa", "couch", "tv", "television", "coffee table"],
    "bathroom": ["toilet", "bathtub", "shower", "sink", "mirror"],
    "study": ["desk", "bookshelf", "computer", "chair"],
}

ANCHOR_SIM_THRESHOLD = 0.25   # CLIP 相似度阈值，视觉嵌入与文本嵌入的余弦相似度
ANCHOR_SIM_THRESHOLD_ROOM = 0.20


@dataclass
class AnchorEntry:
    name: str
    anchor_type: str      # "spatial_obj" | "room" | "area_cue"
    mtuid: Optional[int]
    position: Optional[np.ndarray]  # shape (3,), 世界坐标 (x,y,z)
    status: str           # "found" | "missing"


class AnchorRegistry:
    def __init__(self, clip_text_encoder, clip_tokenizer):
        """
        clip_text_encoder: CLIPTextModel（来自 transformers）
        clip_tokenizer: CLIPTokenizer（来自 transformers，PQ3DModel 已有）
        """
        self.clip_text_encoder = clip_text_encoder
        self.clip_tokenizer = clip_tokenizer
        self._entries: Dict[str, AnchorEntry] = {}  # name → AnchorEntry
        self._text_embed_cache: Dict[str, np.ndarray] = {}

    def reset(self):
        self._entries.clear()
        self._text_embed_cache.clear()

    def register_anchors(self, goal_anchor: "GoalAnchor"):
        """注册当前子目标的锚（已存在的保留状态）。"""
        for a in goal_anchor.anchors:
            if a.name not in self._entries:
                self._entries[a.name] = AnchorEntry(
                    name=a.name,
                    anchor_type=a.type,
                    mtuid=None,
                    position=None,
                    status="missing",
                )

    def get_status(self, anchor_name: str) -> str:
        entry = self._entries.get(anchor_name)
        return entry.status if entry else "missing"

    def get_position(self, anchor_name: str) -> Optional[np.ndarray]:
        entry = self._entries.get(anchor_name)
        return entry.position if entry and entry.status == "found" else None

    def _get_text_embed(self, text: str) -> np.ndarray:
        """获取文本的 CLIP 归一化嵌入（有缓存）。"""
        if text not in self._text_embed_cache:
            inputs = self.clip_tokenizer(
                [text], padding=True, return_tensors="pt", truncation=True
            )
            inputs = {k: v.to(next(self.clip_text_encoder.parameters()).device)
                      for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.clip_text_encoder(**inputs)
                # CLIPTextModel 输出 pooler_output 或 last_hidden_state[:,0,:]
                emb = outputs.pooler_output  # shape (1, 768)
            emb = F.normalize(emb, dim=-1).squeeze(0).cpu().numpy()
            self._text_embed_cache[text] = emb
        return self._text_embed_cache[text]

    def update(self, representation_manager):
        """
        每次 representation_manager.merge() 后调用。
        遍历 missing 锚，尝试在当前 open_vocab_feat 中找到匹配。

        representation_manager.open_vocab_feat: shape (M, 768)，归一化 CLIP 视觉嵌入
        representation_manager.object_box: shape (M, 6)，[cx,cy,cz,dx,dy,dz]
        """
        open_vocab_feat = representation_manager.open_vocab_feat  # (M, 768)
        object_box = representation_manager.object_box             # (M, 6)

        if open_vocab_feat.shape[0] == 0:
            return

        # 归一化视觉嵌入
        feat_tensor = torch.from_numpy(open_vocab_feat).float()
        feat_norm = F.normalize(feat_tensor, dim=-1)  # (M, 768)

        for name, entry in self._entries.items():
            if entry.status == "found":
                continue

            if entry.anchor_type in ("spatial_obj", "area_cue"):
                text_emb = torch.from_numpy(self._get_text_embed(name)).float()  # (768,)
                sims = (feat_norm @ text_emb).numpy()  # (M,)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim > ANCHOR_SIM_THRESHOLD:
                    entry.status = "found"
                    entry.mtuid = best_idx
                    # object_box 格式：[cx, cy, cz, dx, dy, dz]，取中心
                    entry.position = object_box[best_idx][:3].copy()

            elif entry.anchor_type == "room":
                # 用 room-object 先验：统计当前所有对象与各房间代表物体的相似度
                room_name = name  # e.g., "kitchen", "bedroom"
                representative_objects = ROOM_PRIOR.get(room_name, [room_name])
                scores = []
                for obj_name in representative_objects:
                    text_emb = torch.from_numpy(self._get_text_embed(obj_name)).float()
                    sims = (feat_norm @ text_emb).numpy()
                    scores.append(float(np.max(sims)))
                avg_score = np.mean(scores)
                if avg_score > ANCHOR_SIM_THRESHOLD_ROOM:
                    # room 类型锚标记为 found，position 设为当前视野中相关对象的质心
                    entry.status = "found"
                    entry.mtuid = None  # room 类型无单一 mtuid
                    entry.position = object_box[:, :3].mean(axis=0)  # 质心作为近似位置
```

---

## 5. anchor_nav/afs.py

**功能**：ADO（Phase 1 + Phase 2）核心逻辑。

```python
# anchor_nav/afs.py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor_nav.gap import GoalAnchor, AnchorItem
    from anchor_nav.registry import AnchorRegistry

from anchor_nav.proj_utils import project_world_to_pixel, draw_som_markers
from vlm.client import call_vlm_image, parse_json_response, VLM_MODEL_FAST, VLM_MODEL_BEST

IMG_WIDTH = 640
IMG_HEIGHT = 360

FRONTIER_VLM_MODEL = VLM_MODEL_FAST   # Phase 1 用 flash
TARGET_VLM_MODEL = VLM_MODEL_BEST     # Phase 2 用 plus
CLIP_GAP_THRESHOLD = 0.15             # Phase 2：top1-top2 差距 > 此值跳过 VLM
PHASE2_CONF_THRESHOLD = 0.6           # Phase 2：VLM 置信度阈值
PHASE2_TOP_K = 5                      # Phase 2：CLIP 预筛选候选数


# ─────────────────────────────────────────────
# Phase 1：锚导向 Frontier 选择
# ─────────────────────────────────────────────

def phase1_anchor_frontier_score(
    frontier_waypoints: List[np.ndarray],   # K 个，每个 shape (3,)，世界坐标 (x,y,z)
    color_list: List[np.ndarray],           # 12 帧 spin RGB 图像，uint8 (H,W,3)
    agent_state_list: List,                 # 对应的 12 个 habitat AgentState
    a_missing: List["AnchorItem"],          # 缺失的锚列表
    agent_position: np.ndarray,             # 当前 agent 3D 位置 shape (3,)
) -> Optional[np.ndarray]:
    """
    对所有 frontier 进行 SoM 打分，返回最优 frontier 的 3D 世界坐标。
    若无法完成（所有 frontier 不可见 / VLM 失败）返回 None（降级为 Stage 2 原始选择）。
    """
    if not frontier_waypoints or not a_missing:
        return None

    K = len(frontier_waypoints)
    n_frames = len(color_list)

    # Step 1：为每个 frontier 找最佳可见帧
    frontier_frame_idx = []    # frontier k → 最佳帧索引
    frontier_pixel = []        # frontier k → (px, py)，None 表示不可见

    for fw in frontier_waypoints:
        best_frame = None
        best_pixel = None
        best_dist_to_center = float("inf")

        for fi, agent_state in enumerate(agent_state_list):
            pix = project_world_to_pixel(fw, agent_state, IMG_WIDTH, IMG_HEIGHT)
            if pix is not None:
                cx, cy = IMG_WIDTH // 2, IMG_HEIGHT // 2
                dist = ((pix[0] - cx) ** 2 + (pix[1] - cy) ** 2) ** 0.5
                if dist < best_dist_to_center:
                    best_dist_to_center = dist
                    best_frame = fi
                    best_pixel = pix

        frontier_frame_idx.append(best_frame)
        frontier_pixel.append(best_pixel)

    # 过滤掉不可见的 frontier
    visible_indices = [k for k in range(K) if frontier_pixel[k] is not None]
    if not visible_indices:
        return None

    # Step 2：选取"最具代表性"的帧集合（取包含最多可见 frontier 的前 3 帧）
    from collections import Counter
    frame_counter = Counter(frontier_frame_idx[k] for k in visible_indices)
    selected_frames = [f for f, _ in frame_counter.most_common(3) if f is not None]

    # Step 3：在选定帧上绘制 SoM 标记
    annotated_images = []
    for fi in selected_frames:
        img = color_list[fi].copy()
        for k in visible_indices:
            if frontier_frame_idx[k] == fi:
                img = draw_som_markers(img, [frontier_pixel[k]], [k + 1])
        annotated_images.append(img)

    # Step 4：构建 VLM Prompt
    anchor_desc = ", ".join(f'"{a.name}" ({a.type})' for a in a_missing)
    prompt = (
        f"I am navigating an indoor environment and need to find: {anchor_desc}.\n"
        f"The images show the current view with {len(visible_indices)} marked directions "
        f"(numbered circles). Each number indicates a potential exploration direction.\n"
        f"Score each marked direction (0.0 to 1.0) based on how likely it leads to "
        f"finding the target objects.\n"
        f"Return ONLY a JSON object: {{\"scores\": [<score for mark 1>, <score for mark 2>, ...]}}\n"
        f"Scores list length must equal the number of visible marks: {len(visible_indices)}."
    )

    try:
        raw = call_vlm_image(annotated_images, prompt, model=FRONTIER_VLM_MODEL)
        parsed = parse_json_response(raw)
        vlm_scores = parsed.get("scores", [])
        if len(vlm_scores) != len(visible_indices):
            # 长度不对，尝试截断或填充
            vlm_scores = (vlm_scores + [0.5] * len(visible_indices))[:len(visible_indices)]
    except Exception as e:
        print(f"[ADO Phase1] VLM failed: {e}, falling back to Stage2 selection")
        return None

    # Step 5：计算效用得分并选最优 frontier
    best_k = None
    best_u = -1.0
    for rank, k in enumerate(visible_indices):
        p_k = float(vlm_scores[rank]) if rank < len(vlm_scores) else 0.5
        fw = frontier_waypoints[k]
        dist = float(np.linalg.norm(fw - agent_position)) + 1e-3
        u_k = p_k / dist
        if u_k > best_u:
            best_u = u_k
            best_k = k

    if best_k is None:
        return None

    return frontier_waypoints[best_k]  # shape (3,)，世界坐标


# ─────────────────────────────────────────────
# Phase 2：锚约束目标验证
# ─────────────────────────────────────────────

def phase2_anchor_target_verify(
    representation_manager,
    current_goal: "GoalAnchor",
    registry: "AnchorRegistry",
    color_list: List[np.ndarray],       # spin 帧
    agent_state_list: List,
    clip_text_encoder,
    clip_tokenizer,
) -> Optional[np.ndarray]:
    """
    在 Stage 2 决定去物体时，用锚约束验证/覆写目标选择。
    返回覆写后的 target_position (3D 世界坐标)，或 None（保持 Stage 2 原始选择）。
    """
    open_vocab_feat = representation_manager.open_vocab_feat  # (M, 768)
    object_box = representation_manager.object_box            # (M, 6)

    if open_vocab_feat.shape[0] == 0:
        return None

    # Step 1：CLIP 预筛选 top-K 候选
    feat_tensor = torch.from_numpy(open_vocab_feat).float()
    feat_norm = F.normalize(feat_tensor, dim=-1)

    inputs = clip_tokenizer(
        [current_goal.target], padding=True, return_tensors="pt", truncation=True
    )
    device = next(clip_text_encoder.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_emb = clip_text_encoder(**inputs).pooler_output  # (1, 768)
    text_emb = F.normalize(text_emb, dim=-1).squeeze(0).cpu()

    sims = (feat_norm @ text_emb).numpy()  # (M,)
    top_k_idx = np.argsort(sims)[-PHASE2_TOP_K:][::-1]  # 降序 top-K

    # Step 2：唯一性判断
    if len(top_k_idx) >= 2:
        gap = float(sims[top_k_idx[0]] - sims[top_k_idx[1]])
        if gap > CLIP_GAP_THRESHOLD:
            # 直接用 top-1，无需 VLM
            pos = object_box[top_k_idx[0]][:3].copy()
            return pos

    # Step 3：在最新帧上绘制候选 SoM 标记
    # 使用 spin 阶段最后一帧（最新视角）
    latest_frame_idx = len(color_list) - 1
    agent_state = agent_state_list[latest_frame_idx]
    img = color_list[latest_frame_idx].copy()

    candidate_pixels = []
    for k_rank, k_idx in enumerate(top_k_idx):
        center = object_box[k_idx][:3]  # [cx, cy, cz]
        pix = project_world_to_pixel(center, agent_state, IMG_WIDTH, IMG_HEIGHT)
        candidate_pixels.append(pix)
        if pix is not None:
            img = draw_som_markers(img, [pix], [k_rank + 1], color=(0, 200, 255))

    # 标注已知锚点位置（用绿色方框）
    found_anchors = current_goal.get_found_anchors(registry)
    for anchor in found_anchors:
        anchor_pos = registry.get_position(anchor.name)
        if anchor_pos is not None:
            pix = project_world_to_pixel(anchor_pos, agent_state, IMG_WIDTH, IMG_HEIGHT)
            if pix is not None:
                cv2.rectangle(img, (pix[0]-10, pix[1]-10), (pix[0]+10, pix[1]+10),
                              (0, 255, 0), 2)
                cv2.putText(img, anchor.name, (pix[0]+12, pix[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Step 4：构建 VLM Prompt
    anchor_desc_parts = []
    for a in found_anchors:
        pos = registry.get_position(a.name)
        pos_str = f"at approximately ({pos[0]:.1f}, {pos[2]:.1f})" if pos is not None else ""
        anchor_desc_parts.append(f'"{a.name}" {pos_str}')
    anchor_desc = "; ".join(anchor_desc_parts) if anchor_desc_parts else "none found yet"

    area_cue_anchors = [a for a in current_goal.anchors if a.type == "area_cue"]
    area_cue_note = ""
    if area_cue_anchors:
        area_cue_desc = ", ".join(f'"{a.name}"' for a in area_cue_anchors)
        area_cue_note = (
            f"\nAlso confirm: the floor/wall/area around the target matches {area_cue_desc}. "
            f"If it does not match, set confidence to 0 for non-matching candidates."
        )

    visible_candidates = [(i+1, top_k_idx[i]) for i in range(len(top_k_idx))
                          if candidate_pixels[i] is not None]
    if not visible_candidates:
        # 没有候选在视野内，降级
        pos = object_box[top_k_idx[0]][:3].copy()
        return pos

    prompt = (
        f"Target object: \"{current_goal.target}\".\n"
        f"Reference objects (green rectangles): {anchor_desc}.\n"
        f"Candidate objects (orange circles, numbered): "
        f"{', '.join(str(v[0]) for v in visible_candidates)}.\n"
        f"Which numbered circle best matches the target based on the description and "
        f"its proximity to the reference objects?{area_cue_note}\n"
        f"Return ONLY JSON: {{\"best_match\": <number or null>, \"confidence\": <0.0-1.0>}}"
    )

    try:
        raw = call_vlm_image([img], prompt, model=TARGET_VLM_MODEL)
        parsed = parse_json_response(raw)
        best_num = parsed.get("best_match")
        confidence = float(parsed.get("confidence", 0.0))
        if best_num is not None and confidence > PHASE2_CONF_THRESHOLD:
            # best_num 是 1-indexed
            rank_idx = int(best_num) - 1
            if 0 <= rank_idx < len(top_k_idx):
                selected_mtuid = top_k_idx[rank_idx]
                return object_box[selected_mtuid][:3].copy()
    except Exception as e:
        print(f"[ADO Phase2] VLM failed: {e}, using CLIP top-1 fallback")

    # 降级：使用 CLIP top-1
    return object_box[top_k_idx[0]][:3].copy()
```

---

## 6. anchor_nav/\_\_init\_\_.py

```python
# anchor_nav/__init__.py
from anchor_nav.gap import GoalAnchorParser, GoalAnchor, AnchorItem
from anchor_nav.registry import AnchorRegistry
from anchor_nav.afs import phase1_anchor_frontier_score, phase2_anchor_target_verify

__all__ = [
    "GoalAnchorParser", "GoalAnchor", "AnchorItem",
    "AnchorRegistry",
    "phase1_anchor_frontier_score", "phase2_anchor_target_verify",
]
```

---

## 7. data_utils.py 修改：新增 CLIP 文本编码器

在 `PQ3DModel.__init__` 末尾添加以下代码：

```python
# 在 self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14") 之后添加：
from transformers import CLIPTextModel
self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
self.clip_text_model.eval()
self.clip_text_model.cuda()
```

> **注意**：`CLIPTextModel` 与 `self.tokenizer`（CLIPTokenizer）配套。`PQ3DModel` 已有 tokenizer，只需新增 text encoder。模型权重与已用的 CLIP backbone 相同，不会引入新的模型下载需求。

---

## 8. plugin_hooks.py 修改：填充实现

**`plugin_hooks.py` 已存在，需要填充 stubs 的具体实现。**

以下展示每个钩子的完整实现：

```python
# plugin_hooks.py 完整替换版（保留原有接口签名不变）

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ── 全局状态（在模块层面维护 episode 级别状态）──
_goal_anchor_list: List[Any] = []         # List[GoalAnchor]
_anchor_registry: Optional[Any] = None   # AnchorRegistry 实例
_gap_parser: Optional[Any] = None        # GoalAnchorParser 实例
_last_mtuid_count: int = 0


def _ensure_modules_loaded(pq3d_model=None):
    """懒加载 anchor_nav 模块，避免循环导入。"""
    global _anchor_registry, _gap_parser
    if _gap_parser is None:
        from anchor_nav.gap import GoalAnchorParser
        _gap_parser = GoalAnchorParser()
    if _anchor_registry is None and pq3d_model is not None:
        from anchor_nav.registry import AnchorRegistry
        _anchor_registry = AnchorRegistry(
            clip_text_encoder=pq3d_model.clip_text_model,
            clip_tokenizer=pq3d_model.tokenizer,
        )


def load_plugin_config(config_path: Optional[str] = None) -> Dict[str, bool]:
    """读取 inference_config.yaml，返回 plugins 字典。"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "inference_config.yaml")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        plugins = cfg.get("plugins", {})
        return {
            "use_gap": plugins.get("use_gap", False),
            "use_registry": plugins.get("use_registry", False),
            "use_ado": plugins.get("use_ado", False),
        }
    except Exception:
        return {"use_gap": False, "use_registry": False, "use_ado": False}


def on_episode_start(
    plugins: Dict[str, bool],
    all_task_descriptions: List[str],   # 当前 episode 所有子目标描述，预读
    pq3d_model=None,                    # 用于获取 clip_text_model
    **kwargs,
) -> List[Any]:
    """
    Episode 开始时调用（pq3d_model.reset() 之后）。
    解析所有子目标描述，初始化 Registry。
    返回 goal_anchor_list（List[GoalAnchor]）。
    """
    global _goal_anchor_list, _anchor_registry
    _ensure_modules_loaded(pq3d_model)

    if _anchor_registry is not None:
        _anchor_registry.reset()

    if plugins.get("use_gap") and _gap_parser is not None:
        try:
            _goal_anchor_list = _gap_parser.parse_all(all_task_descriptions)
            print(f"[GAP] Parsed {len(_goal_anchor_list)} sub-goals")
        except Exception as e:
            print(f"[GAP] Failed: {e}")
            _goal_anchor_list = []
    else:
        _goal_anchor_list = []

    return _goal_anchor_list


def on_sub_episode_start(
    plugins: Dict[str, bool],
    sub_episode_index: int,
) -> None:
    """
    每个子目标开始时调用。注册当前子目标的锚到 Registry。
    """
    if not plugins.get("use_registry"):
        return
    if _anchor_registry is None or not _goal_anchor_list:
        return
    if sub_episode_index < len(_goal_anchor_list):
        _anchor_registry.register_anchors(_goal_anchor_list[sub_episode_index])


def on_after_merge(
    plugins: Dict[str, bool],
    representation_manager: Any,
    **kwargs,
) -> None:
    """每次 Stage1 merge() 后调用，更新 Registry。"""
    if not plugins.get("use_registry"):
        return
    if _anchor_registry is None:
        return
    try:
        _anchor_registry.update(representation_manager)
    except Exception as e:
        print(f"[Registry] update failed: {e}")


def on_after_decision(
    plugins: Dict[str, bool],
    target_position: np.ndarray,
    is_final_decision: bool,
    representation_manager: Any,
    frontier_waypoints: List,
    color_list: List,
    agent_state_list: List,
    sub_episode_index: int,
    pq3d_model=None,
    **kwargs,
) -> Tuple[np.ndarray, bool]:
    """
    decision() 返回后调用；用 ADO 结果覆盖目标或前沿选择。
    返回: (target_position, is_final_decision)
    """
    if not plugins.get("use_ado"):
        return target_position, is_final_decision

    if not _goal_anchor_list or sub_episode_index >= len(_goal_anchor_list):
        return target_position, is_final_decision

    current_goal = _goal_anchor_list[sub_episode_index]
    _ensure_modules_loaded(pq3d_model)

    if not is_final_decision:
        # Phase 1：锚导向 frontier 选择
        a_missing = current_goal.get_missing_anchors(_anchor_registry) if _anchor_registry else []
        if a_missing and frontier_waypoints:
            from anchor_nav.afs import phase1_anchor_frontier_score
            agent_pos = np.array(agent_state_list[-1].position)
            fw_arrays = [np.array(fw) for fw in frontier_waypoints]
            result = phase1_anchor_frontier_score(
                frontier_waypoints=fw_arrays,
                color_list=color_list,
                agent_state_list=agent_state_list,
                a_missing=a_missing,
                agent_position=agent_pos,
            )
            if result is not None:
                # 转换回 goat-nav 期望的 target_position 格式 [x,y,z]
                # （data_utils.py line 534 会再做 y↔z swap，此处保持世界坐标）
                target_position = result
    else:
        # Phase 2：锚约束目标验证
        if _anchor_registry is not None and pq3d_model is not None:
            from anchor_nav.afs import phase2_anchor_target_verify
            result = phase2_anchor_target_verify(
                representation_manager=representation_manager,
                current_goal=current_goal,
                registry=_anchor_registry,
                color_list=color_list,
                agent_state_list=agent_state_list,
                clip_text_encoder=pq3d_model.clip_text_model,
                clip_tokenizer=pq3d_model.tokenizer,
            )
            if result is not None:
                target_position = result

    return target_position, is_final_decision
```

---

## 9. goat-nav.py 修改：插入钩子调用

**修改位置共 5 处**，均为插入代码，不删改原有逻辑。

### 修改 1：导入 plugin_hooks（文件顶部，现有 import 之后）

```python
# 在现有 import 之后添加：
from plugin_hooks import (
    load_plugin_config, on_episode_start, on_sub_episode_start,
    on_after_merge, on_after_decision
)
# 在 pq3d_model = PQ3DModel(...) 之后添加：
_plugins = load_plugin_config()
```

### 修改 2：Episode 开始时调用 GAP

在 `pq3d_model.reset()` **之后**，`for sub_episode_index in range(...)` **之前**：

```python
# 预读所有子目标描述（description 类型取 lang_desc，object 类型取 goal_category）
_all_descs = []
for _t in cur_episode['tasks']:
    _gc, _gt = _t[0], _t[1]
    if _gt == 'description':
        _gobj_id = _t[2]
        _g_list = [g for g in navigation_data_dict[split][scene_id]['goals_by_category'][_gc] if g['object_id'] == _gobj_id]
        _all_descs.append(_g_list[0]['lang_desc'] if _g_list else _gc)
    else:
        _all_descs.append(_gc)

_goal_anchor_list = on_episode_start(
    _plugins,
    all_task_descriptions=_all_descs,
    pq3d_model=pq3d_model,
)
```

### 修改 3：子目标开始时注册锚

在 `for sub_episode_index in range(...)` 循环体的最开始（`cur_sub_episode = ...` 之前）：

```python
on_sub_episode_start(_plugins, sub_episode_index)
```

### 修改 4：merge 后更新 Registry

在 `self.representation_manager.merge(pred_dict_list)` 之后——由于 merge 在 `PQ3DModel.decision()` 内部调用，**最简单的插入点**是在 `target_position, is_final_decision = pq3d_model.decision(...)` 调用**之后**：

```python
# 在 pq3d_model.decision() 调用之后，on_after_decision 之前
on_after_merge(_plugins, pq3d_model.representation_manager)
```

### 修改 5：decision 后执行 ADO 覆写

在 `target_position, is_final_decision = pq3d_model.decision(...)` 调用之后（`decision_num += 1` 之前）：

```python
target_position, is_final_decision = on_after_decision(
    _plugins,
    target_position=target_position,
    is_final_decision=is_final_decision,
    representation_manager=pq3d_model.representation_manager,
    frontier_waypoints=frontier_waypoints,
    color_list=color_list,
    agent_state_list=agent_state_list,
    sub_episode_index=sub_episode_index,
    pq3d_model=pq3d_model,
)
```

---

## 10. inference_config.yaml（新建或修改）

```yaml
# hm3d-online/inference_config.yaml
plugins:
  use_gap: true        # 启用 Goal-Anchor Parser
  use_registry: true   # 启用 Anchor Registry
  use_ado: true        # 启用 Anchor Decision Override (Phase 1 + 2)
```

设为 `false` 时所有模块退化为 no-op，与原始 MTU3D 行为完全一致，便于消融对比。

---

## 11. 注意事项与已知问题

### 坐标系一致性
- `data_utils.py` line 534：`target_position[[1, 2]] = target_position[[2, 1]]`，这是 decision() 返回前的最后一步 y↔z 交换
- `on_after_decision` 的输入 `target_position` 是**已交换后**的坐标（goat-nav 侧的格式）
- `phase1/phase2` 函数内操作的是**世界坐标**（y 轴朝上）
- ADO 覆写后返回的 `target_position` 应与 goat-nav 期望格式一致：需在 `on_after_decision` 中确认格式一致

**具体做法**：`object_box[:, :3]` 是世界坐标 [cx, cy, cz]（y 朝上）。而 `data_utils.py` decision 输出的 target_position 格式在返回前做了 `target_position[[1,2]] = target_position[[2,1]]`——即把内部的 [x,z,y] 变成 [x,y,z]。因此 Phase 2 直接返回 `object_box[idx][:3]`（[x,y,z] world）是正确的。Phase 1 返回 frontier 的世界坐标（已经是 [x,y,z]）也是正确的。

### open_vocab_feat 对齐
- `RepresentationManager` 的 `merge()` 会对对象做 IoU 匹配和 EMA 更新，`open_vocab_feat` 的行索引在不同步会变化
- Registry 中记录的 `mtuid` 在每次 merge 后可能失效（对象被合并/重排）
- **建议**：Registry 的 `update()` 每次都对所有 missing 锚重新匹配（不缓存 mtuid），确保使用最新的 open_vocab_feat
- 或者：Registry 记录位置 `position`（3D 坐标，受 EMA 更新影响较小），而不是依赖 mtuid 的稳定性

### VLM 延迟
- Phase 1 的 VLM 调用在每次 decision 步（约每 12 个物理步）触发一次
- `qwen3.5-flash` 响应时间约 1-3 秒，与 Stage 1/2 的推理时间（约 3-5 秒）相比在可接受范围
- Phase 2 使用 `qwen3.5-plus`，延迟略高（2-5 秒），但触发频率低

### 降级保障
- 所有 VLM 调用均有 `try/except`，失败时自动降级为 Stage 2 原始输出
- `use_ado: false` 时完全退化为原始 MTU3D，便于 ablation
