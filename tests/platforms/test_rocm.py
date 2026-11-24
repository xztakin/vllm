# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amdsmi ignores device visibility env vars, so amdsmi answers that depend
on which GPU a process uses must not be taken from an arbitrary handle."""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

# RX 7900 XTX, RX 9070 XT, Radeon AI PRO R9700
MIXED_HOST = ["gfx1100", "gfx1201", "gfx1201"]
MI300X_HOST = ["gfx942"] * 8


@pytest.fixture
def rocm():
    import vllm.platforms.rocm as rocm

    return rocm


@pytest.fixture
def host(rocm, monkeypatch):
    """Fake amdsmi host and torch.cuda for the device this process can see."""
    host: list[str] = []
    monkeypatch.setattr(rocm, "amdsmi_init", lambda: None)
    monkeypatch.setattr(rocm, "amdsmi_shut_down", lambda: None)
    monkeypatch.setattr(
        rocm,
        "amdsmi_get_processor_handles",
        lambda: list(range(len(host))),
    )
    monkeypatch.setattr(
        rocm,
        "amdsmi_get_gpu_asic_info",
        lambda h: {"target_graphics_version": host[h]},
    )
    monkeypatch.setattr(
        rocm.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName="gfx1201:sramecc-:xnack-"),
    )
    return host


def test_gcn_arch_from_amdsmi_on_uniform_host(rocm, host):
    host[:] = MI300X_HOST
    assert rocm._get_gcn_arch() == "gfx942"


def test_gcn_arch_on_mixed_host_uses_visible_device(rocm, host):
    host[:] = MIXED_HOST
    assert "gfx1201" in rocm._get_gcn_arch()


def test_gcn_arch_on_mixed_host_without_visible_device(rocm, host, monkeypatch):
    host[:] = MIXED_HOST

    def no_visible_device(_):
        raise RuntimeError("No HIP GPUs are available")

    monkeypatch.setattr(rocm.torch.cuda, "get_device_properties", no_visible_device)
    assert rocm._get_gcn_arch() == "gfx1100"
