import pytest
from pytest_embedded import Dut


@pytest.mark.target("esp32p4")
@pytest.mark.env("esp32p4")
@pytest.mark.parametrize(
    "config",
    [
        "add2d",
        "average_pooling",
    ],
)
def test_model_common(dut: Dut) -> None:
    dut.run_all_single_board_cases(group="dl_model")
    # dut.expect_exact('Press ENTER to see the list of tests.')
    # dut.write('[test_model]')
    # dut.expect_unity_test_output(timeout = 1000)
