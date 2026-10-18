"""Transactional, environment-keyed causal feedback adapter for policy serving."""
from copy import deepcopy

import numpy as np

from starVLA.utils.robodojo_memory import RoboDojoMemoryBuffer, finite_array


class RoboDojoMemorySessions:
    def __init__(self, spec):
        self.spec = spec
        self.sessions = {}

    def reset(self):
        self.sessions.clear()

    def prepare(self, examples, normalize_state, namespace=""):
        prepared, updates = [], {}
        for example in examples:
            required = ("env_id", "episode_id", "env_step", "raw_state", "memory_feedback")
            if any(k not in example for k in required):
                raise ValueError(f"RoboDojo memory inference requires {required}; legacy stateless clients must supply execution feedback")
            key = (namespace, str(example["env_id"]))
            if key in updates:
                raise ValueError("A batch may contain only one request per environment")
            episode = str(example["episode_id"])
            current = example["env_step"]
            self.spec.indices(current)
            prior = self.sessions.get(key)
            if prior is None or prior[0] != episode:
                if current != 0:
                    raise ValueError("A new episode must start at env_step=0 with empty feedback")
                buffer = RoboDojoMemoryBuffer(self.spec.recent_records, self.spec.decision_stride, self.spec.use_pose)
            else:
                buffer = deepcopy(prior[1])
            raw_state = finite_array(example["raw_state"], (14,), "raw_state")
            for feedback in example["memory_feedback"]:
                buffer.append(feedback["state"], feedback["executed_action"],
                              source_step=feedback["source_step"],
                              visual_tokens=feedback["visual_tokens"],
                              pose=feedback.get("eef_pose"))
            records = buffer.build(current)
            burnin_records = buffer.build_burnin(current)
            contract = self.spec.contract(current, episode)
            self.spec.validate(records, contract)
            # Only observable inputs reach predict_action. State uses the same
            # training q99 transform, while history explicitly stays raw qpos.
            prepared.append({"image": example["image"], "lang": example["lang"],
                             "state": np.asarray(normalize_state(raw_state), dtype=np.float32),
                             "memory_records": records, "memory_burnin_records": burnin_records,
                             "memory_contract": contract})
            updates[key] = (episode, buffer)
        return prepared, updates

    def commit(self, updates):
        # Commit only after prediction and action unnormalization succeed.
        self.sessions.update(updates)
