import pytest

from polanon.judge.base import JudgeError
from polanon.judge.mock import MockJudge


async def test_mock_judge_flags_phrases():
    judge = MockJudge({"dyrektor finansowy huty w Stalowej Woli": "PERSON"})
    findings = await judge.review("<OSOBA_1> to dyrektor finansowy huty w Stalowej Woli.")
    assert findings[0].entity_type == "PERSON"


async def test_mock_judge_can_fail():
    with pytest.raises(JudgeError):
        await MockJudge(fail=True).review("x")
