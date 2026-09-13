"""Tests for the game DataTable adapter.

Unlike the wiki scraper, this adapter's input format is fully known, so these
tests pin real behaviour: the filtering that separates ~750 table rows down to
the pals a player can own, and the two rules that are easy to get wrong --
`IgnoreCombi` and gender-dependent unique pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from palworld_api.ingest.base import IngestError
from palworld_api.ingest.gamefiles import GameFilesSource
from palworld_api.models import Element, Gender, Stat, Work


def _table(rows: dict) -> list[dict]:
    return [{"Type": "DataTable", "Name": "T", "Rows": rows}]


def _monster(zukan: int, rank: int, **overrides) -> dict:
    row = {
        "ZukanIndex": zukan,
        "ZukanIndexSuffix": "",
        "CombiRank": rank,
        "CombiDuplicatePriority": zukan * 100,
        "IgnoreCombi": False,
        "ElementType1": "EPalElementType::Earth",
        "ElementType2": "EPalElementType::None",
        "MaleProbability": 50,
        "Rarity": 5,
        "Hp": 100,
        "MeleeAttack": 100,
        "ShotAttack": 100,
        "Defense": 100,
        "Support": 100,
        "Stamina": 100,
        "RunSpeed": 400,
        "RideSprintSpeed": 500,
        "SlowWalkSpeed": 100,
        "Price": 500,
        "CraftSpeed": 100,
        "WorkSuitability_Mining": 0,
        "WorkSuitability_Handcraft": 0,
        "PassiveSkill1": "None",
        "PassiveSkill2": "None",
        "PassiveSkill3": "None",
        "PassiveSkill4": "None",
    }
    row.update(overrides)
    return row


def _text(rows: dict[str, str]) -> list[dict]:
    return _table(
        {
            key: {"TextData": {"Namespace": "n", "Key": key, "LocalizedString": value}}
            for key, value in rows.items()
        }
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    content = tmp_path / "Pal" / "Content"
    character = content / "Pal" / "DataTable" / "Character"
    passive = content / "Pal" / "DataTable" / "PassiveSkill"
    text = content / "L10N" / "en" / "Pal" / "DataTable" / "Text"
    for directory in (character, passive, text):
        directory.mkdir(parents=True, exist_ok=True)

    monsters = {
        "Anubis": _monster(139, 480, WorkSuitability_Mining=6, WorkSuitability_Handcraft=6),
        "Lamball": _monster(1, 1470, PassiveSkill1="Legend"),
        # Case differs from its localisation key on purpose.
        "WindChimes": _monster(38, 2780),
        # Excluded from the rank formula; breeds only via a unique pair.
        "IceHorse": _monster(110, 150, IgnoreCombi=True),
        # References test/unreleased content in one slot: must be dropped,
        # not fail the dataset's referential-integrity check.
        "CatMage": _monster(79, 2040, PassiveSkill1="Legend", PassiveSkill2="NotInCatalog"),
        "FoxMage": _monster(78, 2080),
        "CatMage_Fire": _monster(79, 1800),
        "FoxMage_Dark": _monster(78, 1900),
        # Noise the filter must drop.
        "SUMMON_Anubis": _monster(139, 9999),
        "Anubis_MAX": _monster(139, 9999),
        "Anubis_BOSS": _monster(139, 480),
        "CutContent": _monster(-1, 9999),
        "QuestProp": _monster(2, 2760),  # real index, but no shipped name
    }
    combos = {
        "1": {
            "ParentTribeA": "EPalTribeID::IceHorse",
            "ParentGenderA": "EPalGenderType::None",
            "ParentTribeB": "EPalTribeID::IceHorse",
            "ParentGenderB": "EPalGenderType::None",
            "ChildCharacterID": "IceHorse",
        },
        "2": {
            "ParentTribeA": "EPalTribeID::CatMage",
            "ParentGenderA": "EPalGenderType::Male",
            "ParentTribeB": "EPalTribeID::FoxMage",
            "ParentGenderB": "EPalGenderType::Female",
            "ChildCharacterID": "FoxMage_Dark",
        },
        "3": {
            "ParentTribeA": "EPalTribeID::CatMage",
            "ParentGenderA": "EPalGenderType::Female",
            "ParentTribeB": "EPalTribeID::FoxMage",
            "ParentGenderB": "EPalGenderType::Male",
            "ChildCharacterID": "CatMage_Fire",
        },
        "4": {
            "ParentTribeA": "EPalTribeID::CutContent",
            "ParentGenderA": "EPalGenderType::None",
            "ParentTribeB": "EPalTribeID::CutContent",
            "ParentGenderB": "EPalGenderType::None",
            "ChildCharacterID": "CutContent",
        },
    }
    passives = {
        "Legend": {
            "Rank": 3,
            "LotteryWeight": 0,
            "EffectType1": "EPalPassiveSkillEffectType::MeleeAttack",
            "EffectValue1": 20.0,
            "EffectType2": "EPalPassiveSkillEffectType::Defense",
            "EffectValue2": 20.0,
            "EffectType3": "EPalPassiveSkillEffectType::no",
            "EffectValue3": 0.0,
        },
        "Noukin": {
            "Rank": 2,
            "LotteryWeight": 100,
            "EffectType1": "EPalPassiveSkillEffectType::ElementBoost_Fire",
            "EffectValue1": 30.0,
            "EffectType2": "EPalPassiveSkillEffectType::no",
            "EffectValue2": 0.0,
            "EffectType3": "EPalPassiveSkillEffectType::no",
            "EffectValue3": 0.0,
        },
        "Serious": {
            "Rank": 2,
            "LotteryWeight": 50,
            "EffectType1": "EPalPassiveSkillEffectType::WorkSuitabilityAddRank_Mining",
            "EffectValue1": 1.0,
            "EffectType2": "EPalPassiveSkillEffectType::LifeSteal",
            "EffectValue2": 5.0,
            "EffectType3": "EPalPassiveSkillEffectType::no",
            "EffectValue3": 0.0,
        },
        "AttackUp_1": {
            "Rank": 1,
            "LotteryWeight": 100,
            "EffectType1": "EPalPassiveSkillEffectType::ShotAttack",
            "EffectValue1": 5.0,
            "EffectType2": "EPalPassiveSkillEffectType::no",
            "EffectValue2": 0.0,
            "EffectType3": "EPalPassiveSkillEffectType::no",
            "EffectValue3": 0.0,
        },
        "AttackUp_2": {
            "Rank": 2,
            "LotteryWeight": 100,
            "EffectType1": "EPalPassiveSkillEffectType::ShotAttack",
            "EffectValue1": 10.0,
            "EffectType2": "EPalPassiveSkillEffectType::no",
            "EffectValue2": 0.0,
            "EffectType3": "EPalPassiveSkillEffectType::no",
            "EffectValue3": 0.0,
        },
        "Unnamed": {
            "Rank": 1,
            "LotteryWeight": 0,
            "EffectType1": "EPalPassiveSkillEffectType::MaxHP",
            "EffectValue1": 10.0,
            "EffectType2": "EPalPassiveSkillEffectType::no",
            "EffectValue2": 0.0,
            "EffectType3": "EPalPassiveSkillEffectType::no",
            "EffectValue3": 0.0,
        },
    }

    (character / "DT_PalMonsterParameter.json").write_text(json.dumps(_table(monsters)))
    (character / "DT_PalCombiUnique.json").write_text(json.dumps(_table(combos)))
    (passive / "DT_PassiveSkill_Main.json").write_text(json.dumps(_table(passives)))
    (text / "DT_PalNameText_Common.json").write_text(
        json.dumps(
            _text(
                {
                    "PAL_NAME_Anubis": "Anubis",
                    "PAL_NAME_Lamball": "Lamball",
                    # Lower-case 'c' where the monster row has 'C'.
                    "PAL_NAME_Windchimes": "Hangyu",
                    "PAL_NAME_IceHorse": "Frostallion",
                    "PAL_NAME_CatMage": "Katress",
                    "PAL_NAME_FoxMage": "Wixen",
                    "PAL_NAME_CatMage_Fire": "Katress Ignis",
                    "PAL_NAME_FoxMage_Dark": "Wixen Noct",
                    "PAL_NAME_SUMMON_Anubis": "Anubis",
                    "PAL_NAME_Anubis_BOSS": "Anubis",
                    "PAL_NAME_CutContent": "en Text",
                }
            )
        )
    )
    (text / "DT_SkillNameText_Common.json").write_text(
        json.dumps(
            _text(
                {
                    "PASSIVE_Legend": "Legend",
                    "PASSIVE_Noukin": "Pyromaniac",
                    "PASSIVE_Serious": "Serious",
                    "PASSIVE_AttackUp_1": "Attack Up Lv. 1",
                    "PASSIVE_AttackUp_2": "Attack Up Lv. 2",
                    "PASSIVE_Unnamed": "en Text",
                }
            )
        )
    )
    return tmp_path


@pytest.fixture
def dataset(root: Path):
    return GameFilesSource(root=root, game_version="test").fetch()


def _ids(dataset) -> set[str]:
    return {p.id for p in dataset.pals}


def test_roster_keeps_only_real_pals(dataset) -> None:
    assert _ids(dataset) == {
        "anubis",
        "lamball",
        "windchimes",
        "icehorse",
        "catmage",
        "foxmage",
        "catmage_fire",
        "foxmage_dark",
    }


def test_summon_and_boss_variants_are_dropped(dataset) -> None:
    ids = _ids(dataset)
    assert not any("summon" in i or i.endswith(("_max", "_boss")) for i in ids)


def test_rows_without_a_shipped_name_are_dropped(dataset) -> None:
    """Quest props and cut content borrow a real pal's paldeck index.

    The only thing separating them from the real pal is that the game ships no
    display name for them, so the name check is what keeps them out.
    """
    assert "questprop" not in _ids(dataset)
    assert "cutcontent" not in _ids(dataset)


def test_localisation_lookup_is_case_insensitive(dataset) -> None:
    # Monster row `WindChimes`, localisation key `PAL_NAME_Windchimes`.
    hangyu = next(p for p in dataset.pals if p.id == "windchimes")
    assert hangyu.name == "Hangyu"


def test_core_fields_are_read(dataset) -> None:
    anubis = next(p for p in dataset.pals if p.id == "anubis")
    assert anubis.combi_rank == 480
    assert anubis.paldeck == "139"
    assert anubis.elements == (Element.GROUND,)
    assert anubis.work == {Work.MINING: 6, Work.HANDIWORK: 6}
    assert anubis.stats[Stat.ATTACK] == 100.0
    assert anubis.stats[Stat.HP] == 100.0
    assert anubis.male_probability == 50
    assert anubis.provenance.confidence.value == "game_files"


def test_base_stats_cover_the_full_combat_sheet(dataset) -> None:
    anubis = next(p for p in dataset.pals if p.id == "anubis")
    assert anubis.base_stats.hp == 100.0
    assert anubis.base_stats.melee_attack == 100.0
    assert anubis.base_stats.ranged_attack == 100.0
    assert anubis.base_stats.defense == 100.0
    assert anubis.base_stats.support == 100.0
    assert anubis.base_stats.stamina == 100.0
    assert anubis.base_stats.run_speed == 400.0
    assert anubis.base_stats.mount_run_speed == 500.0
    assert anubis.base_stats.walk_speed == 100.0
    assert anubis.base_stats.price == 500.0


def test_base_stats_field_missing_from_source_stays_none(root: Path) -> None:
    import json

    monster_path = (
        root / "Pal" / "Content" / "Pal" / "DataTable" / "Character"
        / "DT_PalMonsterParameter.json"
    )
    monsters = json.loads(monster_path.read_text())
    del monsters[0]["Rows"]["Anubis"]["Price"]
    monster_path.write_text(json.dumps(monsters))

    dataset = GameFilesSource(root=root, game_version="test").fetch()
    anubis = next(p for p in dataset.pals if p.id == "anubis")
    assert anubis.base_stats.price is None
    assert anubis.base_stats.hp == 100.0  # unrelated fields are unaffected


def test_ignore_combi_excludes_a_pal_from_the_formula(dataset) -> None:
    """`IgnoreCombi` is what makes the legendaries breed only into themselves."""
    frostallion = next(p for p in dataset.pals if p.id == "icehorse")
    assert frostallion.breedable_as_parent is False
    assert frostallion.breedable_as_child is False


def test_ignore_combi_pals_still_breed_through_their_unique_pair(dataset) -> None:
    from palworld_api.breeding import BreedingEngine
    from palworld_api.dataset import PalIndex

    engine = BreedingEngine(PalIndex(dataset))
    result = engine.breed("icehorse", "icehorse")
    assert result.child == "icehorse"
    assert result.rule == "special"


def test_gendered_pairs_both_survive(dataset) -> None:
    pair = [c for c in dataset.special_combos if set(c.parents) == {"catmage", "foxmage"}]
    assert len(pair) == 2
    assert {c.child for c in pair} == {"catmage_fire", "foxmage_dark"}
    assert all(c.parent_genders is not None for c in pair)


def test_gender_order_follows_the_sorted_parents(dataset) -> None:
    combo = next(
        c
        for c in dataset.special_combos
        if c.parents == ("catmage", "foxmage") and c.child == "catmage_fire"
    )
    # catmage sorts first; the female requirement must travel with it.
    assert combo.parent_genders == (Gender.FEMALE, Gender.MALE)


def test_combos_referencing_dropped_pals_are_skipped(dataset) -> None:
    assert all(
        "cutcontent" not in (c.parents + (c.child,)) for c in dataset.special_combos
    )


def test_unique_combo_children_are_not_wild_obtainable(dataset) -> None:
    katress_ignis = next(p for p in dataset.pals if p.id == "catmage_fire")
    assert katress_ignis.wild_obtainable is False


def test_self_pair_children_stay_wild_obtainable(dataset) -> None:
    """A legendary breeding true is still a pal you caught in the world."""
    frostallion = next(p for p in dataset.pals if p.id == "icehorse")
    assert frostallion.wild_obtainable is True


def test_passives_are_read_with_effects(dataset) -> None:
    legend = next(p for p in dataset.passives if p.name == "Legend")
    assert {(e.stat, e.value) for e in legend.effects} == {
        (Stat.ATTACK, 20.0),
        (Stat.DEFENSE, 20.0),
    }


def test_element_scoped_effects_keep_their_element(dataset) -> None:
    pyro = next(p for p in dataset.passives if p.name == "Pyromaniac")
    assert pyro.effects[0].element is Element.FIRE
    assert pyro.effects[0].stat is Stat.ELEMENT_DAMAGE


def test_work_rank_bonuses_are_separated_from_percentages(dataset) -> None:
    serious = next(p for p in dataset.passives if p.name == "Serious")
    assert serious.work_bonus == {Work.MINING: 1}


def test_unmapped_effects_are_recorded_not_dropped(dataset) -> None:
    serious = next(p for p in dataset.passives if p.name == "Serious")
    assert "LifeSteal" in serious.unmapped_effects


def test_tiered_passives_share_an_exclusive_group(dataset) -> None:
    tiers = [p for p in dataset.passives if p.name.startswith("Attack Up")]
    assert len(tiers) == 2
    assert len({p.exclusive_group for p in tiers}) == 1
    assert tiers[0].exclusive_group == "attack_up"


def test_unnamed_passives_are_skipped(dataset) -> None:
    assert all(p.name != "en Text" for p in dataset.passives)


def test_lottery_weight_is_preserved(dataset) -> None:
    legend = next(p for p in dataset.passives if p.name == "Legend")
    assert legend.lottery_weight == 0


def test_missing_datatable_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="missing DataTable"):
        GameFilesSource(root=tmp_path).fetch()


def test_a_file_that_is_not_a_datatable_is_rejected(root: Path) -> None:
    target = (
        root / "Pal" / "Content" / "Pal" / "DataTable" / "Character"
        / "DT_PalMonsterParameter.json"
    )
    target.write_text(json.dumps({"nope": True}))
    with pytest.raises(IngestError, match="no Rows map"):
        GameFilesSource(root=root).fetch()


def test_innate_passives_are_read_from_passive_skill_slots(dataset) -> None:
    lamball = next(p for p in dataset.pals if p.id == "lamball")
    assert lamball.innate_passives == ("legend",)


def test_a_passive_slot_referencing_test_content_is_dropped(dataset) -> None:
    catmage = next(p for p in dataset.pals if p.id == "catmage")
    # Legend (slot 1) survives; NotInCatalog (slot 2) does not exist in the
    # passive catalog and must be dropped rather than break referential
    # integrity.
    assert catmage.innate_passives == ("legend",)


def test_dropped_innate_passive_is_reported(root: Path) -> None:
    source = GameFilesSource(root=root, game_version="test")
    source.fetch()
    assert any("NotInCatalog" in w for w in source.warnings)


def test_none_and_empty_passive_slots_are_not_innate_passives(dataset) -> None:
    anubis = next(p for p in dataset.pals if p.id == "anubis")
    assert anubis.innate_passives == ()


def test_an_empty_roster_is_an_error(root: Path) -> None:
    target = (
        root / "Pal" / "Content" / "Pal" / "DataTable" / "Character"
        / "DT_PalMonsterParameter.json"
    )
    target.write_text(json.dumps(_table({"X": _monster(-1, 9999)})))
    with pytest.raises(IngestError, match="no pals survived"):
        GameFilesSource(root=root).fetch()
