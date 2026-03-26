from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .ado import phase1_override_frontier, phase2_override_target
from .gap import GoalAnchorParser
from .registry import AnchorRegistry
from .types import GoalAnchor


@dataclass
class AnchorNavContext:
    plugin_flags: dict
    parser: GoalAnchorParser = field(default_factory=GoalAnchorParser)
    registry: Optional[AnchorRegistry] = None
    goals: List[GoalAnchor] = field(default_factory=list)


def on_episode_start(
    ctx: AnchorNavContext,
    all_task_descriptions: List[str],
    pq3d_model,
    has_description_task: bool = False,
) -> None:
    if ctx.registry is None:
        ctx.registry = AnchorRegistry(clip_text_model=pq3d_model.clip_text_model, clip_tokenizer=pq3d_model.tokenizer)
    ctx.registry.reset()
    if ctx.plugin_flags.get("use_gap", False):
        ctx.goals = ctx.parser.parse_all(all_task_descriptions, force_vlm=has_description_task)
        total_anchors = sum(len(g.anchors) for g in ctx.goals)
        print(
            f"[AnchorNav] GAP done: goals={len(ctx.goals)}, anchors={total_anchors}, "
            f"force_vlm={has_description_task}"
        )
    else:
        ctx.goals = []


def on_sub_episode_start(ctx: AnchorNavContext, sub_episode_index: int) -> None:
    if not ctx.plugin_flags.get("use_registry", False):
        return
    if ctx.registry is None:
        return
    if 0 <= sub_episode_index < len(ctx.goals):
        ctx.registry.register_goal(ctx.goals[sub_episode_index])


def on_after_merge(ctx: AnchorNavContext, representation_manager) -> None:
    if not ctx.plugin_flags.get("use_registry", False):
        return
    if ctx.registry is None:
        return
    ctx.registry.update(representation_manager)


def on_after_decision(
    ctx: AnchorNavContext,
    target_position: np.ndarray,
    is_final_decision: bool,
    representation_manager,
    frontier_waypoints: List[np.ndarray],
    color_list,
    agent_state_list,
    sub_episode_index: int,
    pq3d_model,
):
    if not ctx.plugin_flags.get("use_ado", False):
        return target_position, is_final_decision
    if not (0 <= sub_episode_index < len(ctx.goals)):
        return target_position, is_final_decision
    if ctx.registry is None:
        return target_position, is_final_decision

    goal = ctx.goals[sub_episode_index]
    if not is_final_decision:
        missing = goal.missing_anchors(ctx.registry)
        if missing and len(frontier_waypoints) > 0:
            override = phase1_override_frontier(
                frontier_waypoints=[np.array(fw) for fw in frontier_waypoints],
                color_list=color_list,
                agent_state_list=agent_state_list,
                missing_anchor_names=[a.name for a in missing],
                agent_position=np.array(agent_state_list[-1].position),
            )
            if override is not None:
                target_position = override
        return target_position, is_final_decision

    override = phase2_override_target(
        representation_manager=representation_manager,
        goal_anchor=goal,
        registry=ctx.registry,
        color_list=color_list,
        agent_state_list=agent_state_list,
        clip_text_model=pq3d_model.clip_text_model,
        clip_tokenizer=pq3d_model.tokenizer,
    )
    if override is not None:
        target_position = override
    return target_position, is_final_decision

