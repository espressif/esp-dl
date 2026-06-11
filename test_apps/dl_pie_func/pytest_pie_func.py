import pytest
from pytest_embedded import Dut


@pytest.mark.target("esp32p4")
@pytest.mark.target("esp32s3")
@pytest.mark.target("esp32")
@pytest.mark.env("esp32p4")
@pytest.mark.env("esp32s3")
@pytest.mark.env("esp32")
def test_pie_func(dut: Dut) -> None:
    dut.run_all_single_board_cases(group="pie")
