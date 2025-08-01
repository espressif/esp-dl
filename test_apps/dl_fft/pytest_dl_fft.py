import pytest
from pytest_embedded import Dut


@pytest.mark.target("esp32p4")
@pytest.mark.target("esp32s3")
@pytest.mark.target("esp32")
@pytest.mark.env("esp32p4")
@pytest.mark.env("esp32s3")
@pytest.mark.env("esp32")
def test_model_common(dut: Dut) -> None:
    dut.expect_exact("Press ENTER to see the list of tests.")
    dut.write("[dl_fft]")
    dut.expect_unity_test_output(timeout=1000)
