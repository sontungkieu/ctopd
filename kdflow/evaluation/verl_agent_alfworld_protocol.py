# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted for KDFlow from langfengq/verl-agent at commit
# 20bd331bdbc9026a5668e11362178e10ab7400c8.

"""Dependency-free prompts and projection for verl-agent ALFWorld."""

import re
from typing import Tuple


ALFWORLD_TEMPLATE_NO_HISTORY = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_WITH_HISTORY = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TASK_TYPES = (
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)


def project_alfworld_action(response: str) -> Tuple[str, bool]:
    """Reproduce verl-agent's action projection and format-validity flag."""
    original = response or ""
    lowered = original.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_index = lowered.find(start_tag)
    end_index = lowered.find(end_tag)

    if start_index == -1 or end_index == -1:
        return lowered[-30:], False

    action = lowered[start_index + len(start_tag) : end_index].strip().lower()
    valid = True
    if original.find("<think>") == -1 or original.find("</think>") == -1:
        valid = False
    if re.search(r"[\u4e00-\u9fff]", original):
        valid = False
    return action, valid


def task_type_from_gamefile(gamefile: str) -> str:
    for task_type in TASK_TYPES:
        if task_type in (gamefile or ""):
            return task_type
    return "unknown"
