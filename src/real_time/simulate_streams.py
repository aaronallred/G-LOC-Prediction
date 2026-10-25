"""Start LSL streams from a recorded session, for demoing the on-site real-time mode.

Usage: python -m src.real_time.simulate_streams --session "Extra spin data/<session folder>" [--speed 1.0]
"""

import argparse

from src.Data_Pipeline.real_time_data_pipeline import SingleSubjectLSLStreamer


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded session over LSL.")
    parser.add_argument("--session", required=True, help="Session folder containing the stream CSVs and export_meta.json")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0 = real time)")
    args = parser.parse_args()

    streamer = SingleSubjectLSLStreamer(args.session, playback_speed=args.speed, source_id_prefix="DEMO_")
    try:
        streamer.start()
        print(f"Streaming {streamer.stream_names} at {args.speed}x. Ctrl-C to stop.", flush=True)
        streamer.finished_event.wait()
        print("Replay finished", flush=True)
    finally:
        streamer.close()


if __name__ == "__main__":
    main()
