import pytest


@pytest.mark.xfail(reason="no tests were collected, except this to preempt exit code 5")
def test_tests_were_collected(request: str):
    """
    @param request: “Requesting” fixtures
    """
    assert len(request.session.items) > 1