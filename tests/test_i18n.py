"""Tests for the Spanish translation extraction and matching logic.

These exercise the pure parsing/matching functions against synthetic input,
the same way test_gamefiles.py and test_icons.py test their extractors --
they prove the logic is correct, not that palworld.gg's current layout
matches what this module expects (that can only be checked by actually
running `PalworldGGSource`, which needs a browser and the live site).
"""

from __future__ import annotations

import pytest

from palworld_api.ingest.base import IngestError
from palworld_api.ingest.i18n import (
    effect_signature,
    extract_pal_names,
    match_passive_translations,
    parse_passive_description,
)


class TestExtractPalNames:
    def test_extracts_a_single_record(self) -> None:
        chunk = 'const e={id:"sheepball",key:"SheepBall",slug:"lamball",name:"Lamball",index:1};'
        assert extract_pal_names(chunk) == {"sheepball": "Lamball"}

    def test_extracts_several_records(self) -> None:
        chunk = (
            'const e={id:"sheepball",key:"SheepBall",slug:"lamball",name:"Lamball",x:1},'
            'a={id:"pinkcat",key:"PinkCat",slug:"cattiva",name:"Cattiva",x:2};'
        )
        assert extract_pal_names(chunk) == {"sheepball": "Lamball", "pinkcat": "Cattiva"}

    def test_braces_inside_string_fields_do_not_break_extraction(self) -> None:
        # A description containing literal { } must not confuse the matcher
        # into closing the record early or never.
        chunk = (
            '{id:"sheepball",key:"SheepBall",slug:"lamball",name:"Lamball",'
            'description:`Efecto {EffectValue1} y algo más`,after:"ok"},'
            '{id:"pinkcat",key:"PinkCat",slug:"cattiva",name:"Cattiva",x:1}'
        )
        names = extract_pal_names(chunk)
        assert names == {"sheepball": "Lamball", "pinkcat": "Cattiva"}

    def test_no_records_is_an_empty_map_not_an_error(self) -> None:
        assert extract_pal_names("const e = 42;") == {}

    def test_unbalanced_braces_raise(self) -> None:
        with pytest.raises(IngestError, match="unbalanced"):
            extract_pal_names('{id:"x",key:"X",slug:"x",name:"X"')


class TestParsePassiveDescription:
    def test_single_flat_effect(self) -> None:
        assert parse_passive_description("Ataque +10%") == (("attack", 10.0, None),)

    def test_negative_effect(self) -> None:
        assert parse_passive_description("Defensa -20%") == (("defense", -20.0, None),)

    def test_two_line_effect(self) -> None:
        result = parse_passive_description("Ataque +30%\nVel. de trabajo -50%")
        assert set(result) == {("attack", 30.0, None), ("work_speed", -50.0, None)}

    def test_compound_sentence_with_shared_verb(self) -> None:
        # Legend's real Spanish text: one sentence, three stats, no per-clause
        # repetition of "Aumenta".
        text = (
            "Aumenta el ataque en un 20 %,\n"
            "la defensa en un 20 %\n"
            "y la velocidad de movimiento en un 20 %."
        )
        result = set(parse_passive_description(text))
        assert result == {
            ("attack", 20.0, None),
            ("defense", 20.0, None),
            ("movement_speed", 20.0, None),
        }

    def test_element_damage_with_element(self) -> None:
        result = parse_passive_description(
            "Aumenta el daño de los ataques de fuego en un 30 %"
        )
        assert result == (("element_damage", 30.0, "fire"),)

    def test_two_element_damages(self) -> None:
        text = (
            "Aumenta el daño de los ataques de fuego en un 30 %\n"
            "Aumenta el daño de los ataques de rayo en un 30 %"
        )
        result = set(parse_passive_description(text))
        assert result == {
            ("element_damage", 30.0, "fire"),
            ("element_damage", 30.0, "electric"),
        }

    def test_hunger_loss_sign_flips_on_qualifier(self) -> None:
        # "harder to reduce" = resistance = negative loss;
        # "more easily" = vulnerability = positive loss.
        harder = parse_passive_description("La saciedad se reduce con más dificultad: +50 %")
        easier = parse_passive_description("La saciedad se reduce más fácilmente: +15 %")
        assert harder == (("hunger_loss", -50.0, None),)
        assert easier == (("hunger_loss", 15.0, None),)

    def test_sanity_loss_baja_mas_menos_phrasing(self) -> None:
        more = parse_passive_description("La COR baja un +10.0 % más.")
        less = parse_passive_description("La COR baja un +10.0 % menos.")
        assert more == (("sanity_loss", 10.0, None),)
        assert less == (("sanity_loss", -10.0, None),)

    def test_unrecognised_phrasing_yields_no_effects(self) -> None:
        # Equipment/sphere bonuses and pure-flavour text are out of this
        # project's Stat vocabulary; silently skipped, not guessed at.
        assert parse_passive_description("Número de saltos en montura +1") == ()
        assert parse_passive_description("Inmune a timidez") == ()

    def test_swimming_speed_is_not_confused_with_movement_speed(self) -> None:
        # A pal-only "faster in water" bonus is a different stat than the
        # general movement-speed passive and must not be misparsed as one.
        assert parse_passive_description(
            "Aumenta la velocidad de movimiento en el agua en un 30 %"
        ) == ()

    def test_empty_description_yields_no_effects(self) -> None:
        assert parse_passive_description("") == ()


class TestEffectSignature:
    def test_order_independent(self) -> None:
        a = effect_signature([{"stat": "attack", "value": 20.0}, {"stat": "defense", "value": 10.0}])
        b = effect_signature([{"stat": "defense", "value": 10.0}, {"stat": "attack", "value": 20.0}])
        assert a == b

    def test_rounds_values(self) -> None:
        a = effect_signature([{"stat": "attack", "value": 20.04}])
        b = effect_signature([{"stat": "attack", "value": 20.0}])
        assert a == b


class TestMatchPassiveTranslations:
    def test_unique_signature_resolves(self) -> None:
        passives = [{"id": "legend", "name": "Legend", "effects": [{"stat": "attack", "value": 20.0}]}]
        cards = [{"name": "Leyenda", "effects": (("attack", 20.0, None),)}]
        resolved, unresolved = match_passive_translations(passives, cards)
        assert resolved == {"legend": "Leyenda"}
        assert unresolved == ()

    def test_ambiguous_signature_prefers_the_non_tiered_name(self) -> None:
        passives = [
            {"id": "attack_up_2", "name": "Attack Up Lv. 2", "effects": [{"stat": "attack", "value": 10.0}]},
            {"id": "brave", "name": "Brave", "effects": [{"stat": "attack", "value": 10.0}]},
        ]
        cards = [{"name": "Valentía", "effects": (("attack", 10.0, None),)}]
        resolved, unresolved = match_passive_translations(passives, cards)
        assert resolved == {"brave": "Valentía"}
        assert unresolved == ()

    def test_genuinely_ambiguous_signature_is_reported_not_guessed(self) -> None:
        # Two non-generic English names share one exact numeric signature:
        # no honest way to pick, so it must come back unresolved.
        passives = [
            {"id": "brave", "name": "Brave", "effects": [{"stat": "attack", "value": 10.0}]},
            {"id": "vanguard", "name": "Vanguard", "effects": [{"stat": "attack", "value": 10.0}]},
        ]
        cards = [{"name": "Valentía", "effects": (("attack", 10.0, None),)}]
        resolved, unresolved = match_passive_translations(passives, cards)
        assert resolved == {}
        assert unresolved == ("Valentía",)

    def test_no_matching_signature_is_silently_dropped(self) -> None:
        passives = [{"id": "legend", "name": "Legend", "effects": [{"stat": "attack", "value": 20.0}]}]
        cards = [{"name": "Algo Distinto", "effects": (("defense", 99.0, None),)}]
        resolved, unresolved = match_passive_translations(passives, cards)
        assert resolved == {}
        assert unresolved == ()

    def test_a_card_with_no_effects_is_ignored(self) -> None:
        passives = [{"id": "legend", "name": "Legend", "effects": [{"stat": "attack", "value": 20.0}]}]
        cards = [{"name": "Sin Efecto", "effects": ()}]
        resolved, unresolved = match_passive_translations(passives, cards)
        assert resolved == {}
        assert unresolved == ()

    def test_element_matters_for_matching(self) -> None:
        # Same stat and value, different element: must not cross-match.
        passives = [
            {
                "id": "pyromaniac",
                "name": "Pyromaniac",
                "effects": [{"stat": "element_damage", "value": 10.0, "element": "fire"}],
            }
        ]
        fire_card = [{"name": "Profuego", "effects": (("element_damage", 10.0, "fire"),)}]
        water_card = [{"name": "Proagua", "effects": (("element_damage", 10.0, "water"),)}]
        resolved, _ = match_passive_translations(passives, fire_card)
        assert resolved == {"pyromaniac": "Profuego"}
        resolved, _ = match_passive_translations(passives, water_card)
        assert resolved == {}
