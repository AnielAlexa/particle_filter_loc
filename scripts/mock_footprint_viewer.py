#!/usr/bin/env python3
"""Mock footprint viewer: replays MCAP, shows drone view vs satellite footprint side-by-side.

Uses RTK ground truth for position/heading — no particle filter involved.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import json
import numpy as np
import yaml

# Add package to path
PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from particle_filter_loc.geo_utils import ENUFrame
from particle_filter_loc.footprint_reconstruction import SatelliteFootprintReconstructor


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_viewer(config_path: str, show: bool = True, save_frames: bool = False,
               max_frames: int = 0):
    cfg = load_config(config_path)
    enu = ENUFrame(cfg["enu_origin"]["lat"], cfg["enu_origin"]["lon"])

    # Replay config
    rcfg = cfg["replay"]
    mcap_path = rcfg["mcap_path"]
    camera_topic = rcfg.get("camera_topic", "/camera/image_mono")
    rtk_topic = rcfg.get("rtk_topic", "/m300/rtk/fix")
    yaw_topic = rcfg.get("yaw_topic", "/m300/rtk/yaw")
    altimeter_topic = rcfg.get("altimeter_topic", "/altimeter/range")
    start_offset_s = rcfg.get("start_offset_s", 0.0)
    camera_subsample = rcfg.get("camera_subsample", 1)

    # Camera intrinsics (original resolution)
    mcfg = cfg["matchers"]
    # Original camera: 1280x720, fx=1129, fy=1130
    # Config has scaled intrinsics for matcher resolution; use original for footprint
    orig_fx = cfg.get("camera_original", {}).get("fx", 1129.0)
    orig_fy = cfg.get("camera_original", {}).get("fy", 1130.0)
    orig_w = cfg.get("camera_original", {}).get("w", 1280)
    orig_h = cfg.get("camera_original", {}).get("h", 720)
    heading_offset_deg = cfg.get("camera_original", {}).get("heading_offset_deg", 0.0)

    # Load satellite metadata and reconstructor
    script_dir = Path(mcfg["script_dir"])
    meta_path = script_dir / mcfg["gps_metadata_path"]
    patches_dir = script_dir / mcfg["patches_dir"]

    print(f"Loading GPS metadata from {meta_path}")
    with open(str(meta_path)) as f:
        gps_metadata = json.load(f)

    # Apply GPS offset if configured
    lat_off = mcfg.get("gps_offset_lat", 0.0)
    lon_off = mcfg.get("gps_offset_lon", 0.0)
    if lat_off != 0.0 or lon_off != 0.0:
        for entry in gps_metadata.values():
            entry["lat"] += lat_off
            entry["lon"] += lon_off
            b = entry["bounds"]
            b["min_lat"] += lat_off; b["max_lat"] += lat_off
            b["min_lon"] += lon_off; b["max_lon"] += lon_off
        print(f"GPS offset applied: +{lat_off*111320:.1f}m N, +{lon_off*111320*0.707:.1f}m E")

    reconstructor = SatelliteFootprintReconstructor(gps_metadata, patches_dir, enu)
    print(f"Reconstructor ready: {len(gps_metadata)} tiles from {patches_dir}")

    # Output directory for saved frames
    save_dir = None
    if save_frames:
        save_dir = PKG_DIR / "results" / "footprint_viewer"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to {save_dir}")

    # State
    altitude_buf = deque(maxlen=10)
    current_altitude_m = 0.0
    current_lat, current_lon = None, None
    current_yaw_deg = 0.0
    camera_frame_idx = 0
    paused = False

    # Panel size
    panel_h, panel_w = 480, 640

    print(f"Opening MCAP: {mcap_path}")

    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    bag_file = open(mcap_path, "rb")
    reader = make_reader(bag_file, decoder_factories=[DecoderFactory()])
    topics = [camera_topic, rtk_topic, yaw_topic, altimeter_topic]

    t_start = None
    print("Starting replay... (SPACE=pause, Q=quit)")

    for schema, channel, message, decoded_msg in reader.iter_decoded_messages(topics=topics):
        topic = channel.topic
        ts_ns = message.log_time

        if t_start is None:
            t_start = ts_ns

        elapsed_s = (ts_ns - t_start) * 1e-9
        if elapsed_s < start_offset_s:
            continue

        # --- Altimeter ---
        if topic == altimeter_topic:
            alt = float(decoded_msg.range)
            altitude_buf.append(alt)
            current_altitude_m = float(np.median(altitude_buf)) if altitude_buf else alt
            continue

        # --- Yaw ---
        if topic == yaw_topic:
            current_yaw_deg = float(decoded_msg.data) * 10.0
            continue

        # --- RTK ---
        if topic == rtk_topic:
            current_lat = decoded_msg.latitude
            current_lon = decoded_msg.longitude
            continue

        # --- Camera ---
        if topic == camera_topic:
            if current_lat is None or current_altitude_m < 10.0:
                continue

            camera_frame_idx += 1
            if camera_frame_idx % camera_subsample != 0:
                continue

            if max_frames > 0 and camera_frame_idx > max_frames:
                print(f"Reached max_frames={max_frames}, stopping.")
                break

            # Decode image
            h = decoded_msg.height
            w = decoded_msg.width
            if decoded_msg.encoding == "mono8":
                frame_gray = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(h, w)
                frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            elif decoded_msg.encoding in ("bgr8", "rgb8"):
                frame_bgr = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(h, w, 3)
                if decoded_msg.encoding == "rgb8":
                    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            else:
                continue

            t0 = time.monotonic()

            # Reconstruct satellite footprint
            result = reconstructor.reconstruct(
                current_lat, current_lon,
                current_altitude_m, current_yaw_deg + heading_offset_deg,
                orig_fx, orig_fy, orig_w, orig_h,
                output_size=(panel_h, panel_w),
            )

            dt_ms = (time.monotonic() - t0) * 1000

            # Resize drone frame to panel size
            drone_panel = cv2.resize(frame_bgr, (panel_w, panel_h))

            if result is not None:
                sat_panel = result.satellite_crop
                fp_w = result.footprint_w_m
                fp_h = result.footprint_h_m
                n_tiles = len(result.source_tiles)
            else:
                sat_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(sat_panel, "No tiles", (panel_w // 2 - 60, panel_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                fp_w, fp_h, n_tiles = 0, 0, 0

            # Build side-by-side display
            canvas = np.hstack([drone_panel, sat_panel])

            # Labels on panels
            cv2.putText(canvas, "DRONE VIEW", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, "SATELLITE", (panel_w + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Status bar
            bar_h = 40
            bar = np.zeros((bar_h, panel_w * 2, 3), dtype=np.uint8)
            status = (
                f"alt={current_altitude_m:.0f}m  "
                f"hdg={current_yaw_deg:.0f}deg  "
                f"pos={current_lat:.5f}N {current_lon:.5f}E  "
                f"footprint={fp_w:.0f}x{fp_h:.0f}m  "
                f"tiles={n_tiles}  "
                f"t={dt_ms:.0f}ms  "
                f"frame={camera_frame_idx}"
            )
            cv2.putText(bar, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            canvas = np.vstack([canvas, bar])

            if show:
                cv2.imshow("Footprint Viewer", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested.")
                    break
                elif key == ord(' '):
                    paused = not paused
                    if paused:
                        print("Paused. Press SPACE to resume.")
                        while True:
                            key2 = cv2.waitKey(100) & 0xFF
                            if key2 == ord(' '):
                                paused = False
                                print("Resumed.")
                                break
                            elif key2 == ord('q'):
                                paused = False
                                print("Quit requested.")
                                bag_file.close()
                                cv2.destroyAllWindows()
                                return

            if save_frames and save_dir is not None:
                cv2.imwrite(str(save_dir / f"frame_{camera_frame_idx:06d}.jpg"), canvas)

            if camera_frame_idx % 20 == 0:
                print(f"  [{elapsed_s:.1f}s] frame={camera_frame_idx:4d}  "
                      f"alt={current_altitude_m:.1f}m  hdg={current_yaw_deg:.0f}  "
                      f"footprint={fp_w:.0f}x{fp_h:.0f}m  tiles={n_tiles}  "
                      f"reconstruct={dt_ms:.1f}ms")

    bag_file.close()
    if show:
        cv2.destroyAllWindows()
    print(f"Done. Processed {camera_frame_idx} camera frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Footprint Viewer (RTK ground truth)")
    parser.add_argument("--config", default=str(PKG_DIR / "config" / "pf_config.yaml"),
                        help="Path to pf_config.yaml")
    parser.add_argument("--show", action="store_true",
                        help="Show live OpenCV window")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save frames as JPEGs")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N camera frames (0=unlimited)")
    args = parser.parse_args()
    run_viewer(args.config, show=args.show, save_frames=args.save_frames,
               max_frames=args.max_frames)
