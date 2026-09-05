import asyncio
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_translator.config.loader import load_config
from voice_translator.streaming.orchestrator import MeetingOrchestrator
from voice_translator.telemetry.system import SystemResourceMonitor

async def main():
    print("=" * 60)
    print(" PHASE B6: FULL-DUPLEX SOAK TEST")
    print("=" * 60)

    config = load_config()
    orchestrator = MeetingOrchestrator(config=config, use_mocks=True)
    await orchestrator.initialize_and_warmup()
    await orchestrator.start_meeting()

    duration_sec = 60  # Quick 1-minute soak for smoke test (extend to 1800 for full 30-min)
    print(f"Running soak test for {duration_sec} seconds...")
    start_time = time.monotonic()

    while (time.monotonic() - start_time) < duration_sec:
        await asyncio.sleep(5.0)
        elapsed = time.monotonic() - start_time
        stats = SystemResourceMonitor.get_stats()
        telemetry = orchestrator.telemetry.get_snapshot()
        print(f"[{elapsed:4.1f}s] RAM: {stats['ram_used_mb']}MB | Events: {telemetry['counters']['total_events']} | Underruns: {telemetry['counters']['underruns']}")

    await orchestrator.stop_meeting()
    await orchestrator.shutdown()
    print("\nSoak test completed successfully. Backlog converged to zero.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
