"""Scoring and selection of passive skills.

The question "what are the best passives for this pal" has no single answer,
so this module refuses to pretend otherwise: it takes an explicit `GoalProfile`
(a weight per stat) and returns the optimal set *for that goal*. The two
built-in profiles, base work and combat companion, are just two particular sets
of weights; callers can pass their own.

Selection is exact, not greedy. Scores are additive and the only structural
constraint is "at most one passive per exclusive group, at most N in total",
so taking the best member of each group and then the N best of those is
provably optimal: any valid set can be improved to that form without ever
losing score.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import PalIndex
from .models import Pal, Passive, Stat, Work

# Reducing a loss is a gain, so the loss stats carry negative weights: a passive
# whose effect value is -15 on SANITY_LOSS scores positively against them.
COMBAT_WEIGHTS: dict[Stat, float] = {
    Stat.ATTACK: 1.0,
    Stat.DEFENSE: 0.45,
    Stat.HP: 0.35,
    Stat.ELEMENT_DAMAGE: 0.8,
    Stat.MOVEMENT_SPEED: 0.15,
    Stat.WORK_SPEED: 0.0,
    Stat.SANITY_LOSS: -0.05,
    Stat.HUNGER_LOSS: -0.1,
    Stat.CAPTURE_RATE: 0.0,
}

BASE_WEIGHTS: dict[Stat, float] = {
    Stat.WORK_SPEED: 1.0,
    # A base pal that burns through sanity stops working, so sanity is worth
    # nearly as much as raw speed rather than being a rounding error.
    Stat.SANITY_LOSS: -0.8,
    Stat.HUNGER_LOSS: -0.35,
    Stat.MOVEMENT_SPEED: 0.2,
    Stat.ATTACK: 0.0,
    Stat.DEFENSE: 0.0,
    Stat.HP: 0.05,
    Stat.ELEMENT_DAMAGE: 0.0,
    Stat.CAPTURE_RATE: 0.0,
}


@dataclass(frozen=True, slots=True)
class GoalProfile:
    """What a pal is *for*, expressed as weights the optimiser maximises."""

    name: str
    weights: dict[Stat, float]
    # Conditional passives ("only at night") are worth less than their sticker
    # value. This is the fraction of the time the condition is assumed to hold.
    condition_reliability: float = 0.5
    max_slots: int = 4
    # Whether to consider passives a pal cannot actually be given.
    include_restricted: bool = False

    def weight_for(self, stat: Stat) -> float:
        return self.weights.get(stat, 0.0)


BASE_PROFILE = GoalProfile(name="base", weights=BASE_WEIGHTS)
COMBAT_PROFILE = GoalProfile(name="combat", weights=COMBAT_WEIGHTS)
PROFILES: dict[str, GoalProfile] = {
    "base": BASE_PROFILE,
    "combat": COMBAT_PROFILE,
}


@dataclass(frozen=True, slots=True)
class ScoredPassive:
    passive_id: str
    name: str
    score: float
    # Score before the conditional discount, so a caller can see what the
    # passive is worth when its condition does hold.
    unconditional_score: float
    condition: str | None
    exclusive_group: str | None
    innate: bool = False

    @property
    def is_conditional(self) -> bool:
        return self.condition is not None


@dataclass(frozen=True, slots=True)
class PassiveRecommendation:
    pal_id: str
    goal: str
    passives: tuple[ScoredPassive, ...]
    total_score: float
    considered: int
    # Passives that scored well but are locked to other pals, reported so the
    # answer does not look arbitrarily short.
    excluded_restricted: tuple[str, ...] = ()


class PassiveEngine:
    """Scores passives against a goal and picks the optimal loadout."""

    def __init__(self, index: PalIndex) -> None:
        self.index = index

    def score_passive(
        self, passive: Passive, profile: GoalProfile, pal: Pal | None = None
    ) -> ScoredPassive:
        """Score one passive. Work-suitability relevance scales base scoring."""
        raw = sum(
            effect.value * profile.weight_for(effect.stat) for effect in passive.effects
        )

        # A work-speed buff is only worth anything on a pal that actually works,
        # so scale that component by how much work the pal can do. Without this
        # the optimiser happily recommends work passives for a combat-only pal.
        if pal is not None and profile.weight_for(Stat.WORK_SPEED) > 0:
            work_component = sum(
                effect.value * profile.weight_for(effect.stat)
                for effect in passive.effects
                if effect.stat is Stat.WORK_SPEED
            )
            if work_component:
                raw += work_component * (self._work_relevance(pal) - 1.0)

        score = raw
        if passive.condition is not None:
            score *= profile.condition_reliability

        return ScoredPassive(
            passive_id=passive.id,
            name=passive.name,
            score=round(score, 4),
            unconditional_score=round(raw, 4),
            condition=passive.condition,
            exclusive_group=passive.exclusive_group,
            innate=pal is not None and passive.id in pal.innate_passives,
        )

    @staticmethod
    def _work_relevance(pal: Pal) -> float:
        """0.0 for a pal with no work suitabilities, approaching 1.0 for a specialist.

        Uses the pal's best suitability level rather than the sum: a level-4
        miner is a great base pal even if it does nothing else.
        """
        if not pal.work:
            return 0.0
        best = max(pal.work.values())
        return min(best / 4.0, 1.0)

    def candidates(
        self, pal: Pal, profile: GoalProfile
    ) -> tuple[list[ScoredPassive], list[str]]:
        """Every passive this pal could carry, scored, plus the ones it cannot."""
        scored: list[ScoredPassive] = []
        excluded: list[str] = []
        for passive in self.index.dataset.passives:
            if not passive.applies_to(pal.id) and not profile.include_restricted:
                excluded.append(passive.id)
                continue
            scored.append(self.score_passive(passive, profile, pal))
        return scored, excluded

    def recommend(
        self, pal_ref: str, goal: str | GoalProfile = "combat"
    ) -> PassiveRecommendation:
        """The optimal passive loadout for `pal_ref` against `goal`."""
        pal = self.index.resolve(pal_ref)
        if pal is None:
            raise KeyError(f"unknown pal: {pal_ref!r}")

        profile = goal if isinstance(goal, GoalProfile) else PROFILES.get(goal)
        if profile is None:
            raise KeyError(
                f"unknown goal: {goal!r}. Known goals: {sorted(PROFILES)}"
            )

        scored, excluded = self.candidates(pal, profile)
        chosen = self.select(scored, profile.max_slots)
        return PassiveRecommendation(
            pal_id=pal.id,
            goal=profile.name,
            passives=chosen,
            total_score=round(sum(p.score for p in chosen), 4),
            considered=len(scored),
            excluded_restricted=tuple(sorted(excluded)),
        )

    @staticmethod
    def select(
        scored: list[ScoredPassive], max_slots: int
    ) -> tuple[ScoredPassive, ...]:
        """Pick the highest-scoring valid loadout.

        Optimal because the score is additive: keeping only the best member of
        each exclusive group can never lose score, and the best `max_slots` of
        those group-winners is then the best overall set.
        """
        if max_slots <= 0:
            return ()

        best_per_group: dict[str, ScoredPassive] = {}
        for candidate in scored:
            if candidate.score <= 0:
                # Never spend a slot on something that does not help; a slot
                # left empty is strictly better than a penalty.
                continue
            # Ungrouped passives compete only with themselves.
            group = candidate.exclusive_group or f"\x00{candidate.passive_id}"
            incumbent = best_per_group.get(group)
            if incumbent is None or (candidate.score, candidate.passive_id) > (
                incumbent.score,
                incumbent.passive_id,
            ):
                best_per_group[group] = candidate

        winners = sorted(
            best_per_group.values(), key=lambda p: (-p.score, p.passive_id)
        )
        return tuple(winners[:max_slots])

    def rank_pals_for_work(self, work: Work, limit: int = 20) -> tuple[tuple[str, int], ...]:
        """Pals sorted by their level in one work suitability."""
        ranked = [
            (pal.id, level)
            for pal in self.index.dataset.pals
            if (level := pal.work.get(work, 0)) > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return tuple(ranked[:limit])
