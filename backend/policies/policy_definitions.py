# backend/policies/policy_definitions.py

import numpy as np
from dataclasses import dataclass
from scipy.spatial.distance import cosine as cosine_distance

@dataclass
class PolicyParams:
    """
    Hyperparameters for the adaptation policies.
    These values correspond to those used in the paper's primary simulations.
    """
    alpha: float = 0.5
    cap_value: float = 0.25
    deadband_epsilon: float = 0.1
    radius: float = 0.75
    cache_epsilon: float = 0.01

class Policy:
    """Base class for all adaptation policies."""
    def __init__(self, params: PolicyParams):
        self.params = params
        # This attribute is used to track cache hits for analysis.
        self.cache_hit_turn = False

    def step(self, last_vec: np.ndarray, target_vec: np.ndarray, persona_centroid: np.ndarray) -> np.ndarray:
        """
        Calculates the bot's next style vector based on its previous style
        and the user's current (target) style.

        Args:
            last_vec: The bot's style vector from the previous turn (b_{t-1}).
            target_vec: The user's style vector from the current turn (u_t).
            persona_centroid: The bot's core persona vector (b_c).

        Returns:
            The bot's new target style vector for the current turn (b_t).
        """
        raise NotImplementedError

class Uncapped(Policy):
    """
    The baseline adaptive policy that directly mirrors the user's style.
    This policy provides maximum synchrony but minimum stability.
    """
    def step(self, last_vec, target_vec, persona_centroid):
        return target_vec

class Cap(Policy):
    """
    Limits the magnitude of stylistic change in a single turn.
    It computes the change vector (delta) and scales it down if its
    Euclidean norm exceeds a predefined cap value.
    """
    def step(self, last_vec, target_vec, persona_centroid):
        delta = target_vec - last_vec
        magnitude = np.linalg.norm(delta)
        cap = self.params.cap_value

        if magnitude > cap and magnitude > 0:
            delta = delta / magnitude * cap

        return last_vec + delta

class EMA(Policy):
    """
    Exponential Moving Average. Creates a smoother adaptation trajectory by
    blending the user's current style with the bot's previous style,
    controlled by a smoothing factor alpha.
    """
    def step(self, last_vec, target_vec, persona_centroid):
        alpha = self.params.alpha
        return (1 - alpha) * last_vec + alpha * target_vec

class DeadBand(Policy):
    """
    Prevents the bot from reacting to minor, potentially noisy stylistic
    fluctuations. The bot only updates its style if the distance to the
    user's style exceeds a threshold (epsilon).
    """
    def step(self, last_vec, target_vec, persona_centroid):
        # Using cosine distance as described in the simulation script.
        # A higher distance means more different.
        if cosine_distance(target_vec, last_vec) > self.params.deadband_epsilon:
            return target_vec
        return last_vec

class HybridEMACap(Policy):
    """
    A Pareto-efficient hybrid policy that first calculates a smooth EMA
    target and then applies a cap to the resulting change. This combines
    temporal smoothing with a hard limit on instantaneous change.
    """
    def step(self, last_vec, target_vec, persona_centroid):
        alpha, cap = self.params.alpha, self.params.cap_value

        # First, calculate the smoothed target via EMA.
        smooth_target = (1 - alpha) * last_vec + alpha * target_vec

        # Then, calculate the delta from the last vector to this new target.
        delta = smooth_target - last_vec
        magnitude = np.linalg.norm(delta)

        # Apply the cap to this delta.
        if magnitude > cap and magnitude > 0:
            delta = delta / magnitude * cap

        return last_vec + delta

class HybridEMACapRadius(HybridEMACap):
    """
    Extends the Hybrid (EMA+Cap) policy by adding a 'leash' that
    prevents the bot from drifting too far from its core persona.
    If the bot's style vector exceeds a maximum radius from the persona
    centroid, it is pulled back onto that boundary.
    """
    def step(self, last_vec, target_vec, persona_centroid):
        # First, compute the standard Hybrid (EMA+Cap) update.
        updated_vec = super().step(last_vec, target_vec, persona_centroid)

        # Then, apply the radius constraint.
        radius = self.params.radius
        offset_from_centroid = updated_vec - persona_centroid
        distance = np.linalg.norm(offset_from_centroid)

        if distance > radius and distance > 0:
            return persona_centroid + offset_from_centroid / distance * radius

        return updated_vec

class StaticBaseline(Policy):
    """
    A non-adaptive policy that always returns the fixed persona centroid.
    This serves as the anchor for perfect stability and coherence.
    """
    def step(self, last_vec, target_vec, persona_centroid):
        return persona_centroid

class HybridEMACapCache(HybridEMACap):
    """
    Extends the Hybrid (EMA+Cap) policy with a simple caching mechanism.
    If the user's style is stylistically very close to the bot's previous style,
    no update is performed. This reduces minor, unnecessary stylistic adjustments.
    """
    def step(self, last_vec, target_vec, persona_centroid):
        self.cache_hit_turn = False
        
        # Check if the target is already very close to the current state.
        # 1.0 - cosine_distance gives similarity (0 to 2, where 1 is identical).
        similarity = 1.0 - cosine_distance(target_vec, last_vec)
        
        if similarity >= (1.0 - self.params.cache_epsilon):
            self.cache_hit_turn = True
            return last_vec
            
        return super().step(last_vec, target_vec, persona_centroid)

# Dictionary mapping policy names (as used in plots and tables) to their class definitions.
# This allows the simulation harness to easily instantiate any policy by name.
POLICIES = {
    "Static Baseline": StaticBaseline,
    "Uncapped": Uncapped,
    "Cap": Cap,
    "EMA": EMA,
    "Dead-band": DeadBand,
    "Hybrid": HybridEMACap,
    "Hybrid+Radius": HybridEMACapRadius,
    "Hybrid+Cache": HybridEMACapCache,
}
