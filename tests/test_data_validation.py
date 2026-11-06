from vietlott_ml.data_validation import normalize_draw, validate_draw


def mega_record():
    return {
        "lottery": "mega645",
        "draw_id": "01584",
        "date": "2026-11-04",
        "numbers": [7, 14, 20, 30, 34, 40],
        "special": None,
    }


def power_record():
    return {
        "lottery": "power655",
        "draw_id": "01420",
        "date": "2026-11-05",
        "numbers": [6, 20, 34, 41, 49, 52],
        "special": 36,
    }


def test_valid_mega():
    assert validate_draw(mega_record()) == []


def test_valid_power():
    assert validate_draw(power_record()) == []


def test_duplicate_numbers():
    record = mega_record()
    record["numbers"] = [7, 7, 20, 30, 34, 40]
    errors = validate_draw(record)
    assert any("trùng" in error for error in errors)


def test_out_of_range():
    record = mega_record()
    record["numbers"] = [7, 14, 20, 30, 34, 46]
    errors = validate_draw(record)
    assert any("ngoài khoảng" in error for error in errors)


def test_wrong_count():
    record = mega_record()
    record["numbers"] = [7, 14, 20, 30, 34]
    errors = validate_draw(record)
    assert any("đúng 6 số" in error for error in errors)


def test_power_requires_special():
    record = power_record()
    record["special"] = None
    errors = validate_draw(record)
    assert any("đặc biệt" in error for error in errors)


def test_power_special_cannot_duplicate_main():
    record = power_record()
    record["special"] = 20
    errors = validate_draw(record)
    assert any("trùng số chính" in error for error in errors)


def test_mega_rejects_special():
    record = mega_record()
    record["special"] = 12
    assert validate_draw(record)


def test_normalize_strings():
    draw = normalize_draw(
        lottery="mega645",
        draw_id="001",
        date="2026-01-01T00:00:00",
        numbers=["07", "14", "20", "30", "34", "40"],
    )
    assert draw.numbers == (7, 14, 20, 30, 34, 40)
    assert draw.date == "2026-01-01"
