#!/usr/bin/env python3
"""Test that the round-robin load balancer cycles across healthy ports."""

import threading
from unittest.mock import patch

import api_server


def test_round_robin_cycles_through_all_ports():
    """Sequential calls should cycle through healthy ports."""
    fake_ports = [8000, 8001, 8002]

    # Reset counter
    api_server._rr_counter = 0

    with patch.object(api_server, "get_healthy_ports", return_value=fake_ports):
        results = [api_server.next_vllm_port() for _ in range(9)]

    assert results == [8000, 8001, 8002, 8000, 8001, 8002, 8000, 8001, 8002]


def test_round_robin_skips_unhealthy():
    """If a port becomes unhealthy, it should be skipped."""
    api_server._rr_counter = 0

    with patch.object(api_server, "get_healthy_ports", return_value=[8000, 8001, 8002]):
        r1 = api_server.next_vllm_port()  # 8000
        r2 = api_server.next_vllm_port()  # 8001

    # Now port 8001 goes down
    with patch.object(api_server, "get_healthy_ports", return_value=[8000, 8002]):
        r3 = api_server.next_vllm_port()  # counter=2 → 2%2=0 → 8000
        r4 = api_server.next_vllm_port()  # counter=3 → 3%2=1 → 8002

    assert [r1, r2, r3, r4] == [8000, 8001, 8000, 8002]


def test_round_robin_returns_none_when_no_healthy():
    """Should return None when no backends are healthy."""
    with patch.object(api_server, "get_healthy_ports", return_value=[]):
        assert api_server.next_vllm_port() is None


def test_round_robin_concurrent():
    """Concurrent calls should distribute evenly across ports."""
    fake_ports = [8000, 8001, 8002]
    api_server._rr_counter = 0
    results = []
    lock = threading.Lock()

    def call_n(n):
        for _ in range(n):
            port = api_server.next_vllm_port()
            with lock:
                results.append(port)

    with patch.object(api_server, "get_healthy_ports", return_value=fake_ports):
        threads = [threading.Thread(target=call_n, args=(30,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # 120 total calls across 3 ports — each port should get exactly 40
    from collections import Counter
    counts = Counter(results)
    assert len(results) == 120
    assert counts[8000] == 40
    assert counts[8001] == 40
    assert counts[8002] == 40


if __name__ == "__main__":
    test_round_robin_cycles_through_all_ports()
    print("PASS: cycles through all ports")

    test_round_robin_skips_unhealthy()
    print("PASS: skips unhealthy ports")

    test_round_robin_returns_none_when_no_healthy()
    print("PASS: returns None when no healthy")

    test_round_robin_concurrent()
    print("PASS: concurrent distribution is even")

    print("\nAll round-robin tests passed!")
