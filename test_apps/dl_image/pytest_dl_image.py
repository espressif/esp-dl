import pytest
from pytest_embedded import Dut
from pytest_embedded_idf import IdfDut
import functools

ORIGINAL_PARSE = IdfDut._parse_test_menu


@functools.wraps(ORIGINAL_PARSE)
def _parse_test_menu_with_longer_timeout(
    self,
    ready_line="Press ENTER to see the list of tests",
    pattern="Here's the test menu, pick your combo:(.+)Enter test for running.",
    trigger="",
):
    self.expect_exact(ready_line, timeout=60)
    res = self.confirm_write(trigger, expect_pattern=pattern, timeout=120)
    return self._parse_unity_menu_from_str(res.group(1).decode("utf8"))


IdfDut._parse_test_menu = _parse_test_menu_with_longer_timeout


@pytest.mark.target("esp32p4")
@pytest.mark.target("esp32s3")
@pytest.mark.env("esp32p4")
@pytest.mark.env("esp32s3")
@pytest.mark.timeout(3000)
def test_dl_image(dut: Dut) -> None:
    dut.run_all_single_board_cases()
