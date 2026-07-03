import pytest
from pytest_embedded import Dut


@pytest.mark.target("esp32p4")
@pytest.mark.target("esp32s3")
@pytest.mark.env("esp32p4")
@pytest.mark.env("esp32s3")
@pytest.mark.timeout(300)
def test_mobilenetv2_cls(dut: Dut) -> None:
    # Single-core inference should finish and classify cat.jpg as a tabby cat.
    dut.expect(r"single-core: avg inference latency", timeout=180)
    dut.expect(r"category: tabby", timeout=30)

    # Multi-core inference should also finish with the same top-1 result.
    # If the dual-core path crashes (e.g. heap corruption) this header never
    # appears and the test fails on timeout.
    dut.expect(r"multi-core: avg inference latency", timeout=60)
    dut.expect(r"category: tabby", timeout=30)
