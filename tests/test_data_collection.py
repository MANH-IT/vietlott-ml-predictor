from vietlott_ml.data_collection import merge_draws, parse_live_payload
from vietlott_ml.data_validation import normalize_draw


def live_payload():
    return {
        "schema": "lottery_live_v1",
        "mega645": {
            "r": "mega645",
            "d": "2026-11-04",
            "state": "complete",
            "ky": "",
            "numbers": ["07", "14", "20", "30", "34", "40"],
        },
        "power655": {
            "r": "power655",
            "d": "2026-11-05",
            "state": "complete",
            "ky": "",
            "numbers": ["06", "49", "34", "41", "20", "52"],
            "special": "36",
        },
    }


def test_parse_live_payload():
    draws = parse_live_payload(live_payload(), collected_at="2026-11-05T19:00:00+07:00")
    assert len(draws) == 2
    by_lottery = {draw.lottery: draw for draw in draws}
    assert by_lottery["mega645"].numbers == (7, 14, 20, 30, 34, 40)
    assert by_lottery["power655"].special == 36
    assert by_lottery["mega645"].draw_id == "2026-11-04"


def test_parse_skips_incomplete_state():
    payload = live_payload()
    payload["mega645"]["state"] = "drawing"
    draws = parse_live_payload(payload)
    assert all(draw.lottery != "mega645" for draw in draws)


def test_parse_skips_missing_draw_id():
    payload = live_payload()
    payload["mega645"]["ky"] = ""
    payload["mega645"]["d"] = None
    draws = parse_live_payload(payload)
    assert all(draw.lottery != "mega645" for draw in draws)


def test_merge_prefers_complete_record():
    incomplete = normalize_draw(
        lottery="power655",
        draw_id="",
        date="2026-11-05",
        numbers=[6, 20, 34, 41, 49, 52],
        special=None,
        source="history",
    )
    complete = normalize_draw(
        lottery="power655",
        draw_id="01420",
        date="2026-11-05",
        numbers=[6, 20, 34, 41, 49, 52],
        special=36,
        source="383.im",
    )
    merged = merge_draws([incomplete], [complete])
    assert len(merged) == 1
    assert merged[0].special == 36
    assert merged[0].draw_id == "01420"


def test_merge_deduplicates_and_sorts():
    first = normalize_draw(
        lottery="mega645", draw_id="1", date="2026-01-02", numbers=range(1, 7)
    )
    second = normalize_draw(
        lottery="mega645", draw_id="2", date="2026-01-01", numbers=range(2, 8)
    )
    merged = merge_draws([first], [second])
    assert [draw.date for draw in merged] == ["2026-01-01", "2026-01-02"]
