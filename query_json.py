#!/usr/bin/env python3
"""
query_json.py

Helper script to query and explore the global_identity_catalogue.json

Usage:
    python query_json.py                    # Show summary
    python query_json.py --id 0             # Show specific global_id
    python query_json.py --list             # List all global IDs
    python query_json.py --stats            # Show detailed statistics
"""

import json
import argparse

def load_catalogue(filename="global_identity_catalogue.json"):
    with open(filename, 'r') as f:
        return json.load(f)

def show_summary(data):
    """Display summary information"""
    summary = data.get("summary", {})
    print("\n" + "=" * 60)
    print("📊 GLOBAL IDENTITY CATALOGUE SUMMARY")
    print("=" * 60)
    print(f"Total Global IDs:  {summary.get('total_global_ids', 'N/A')}")
    print(f"Total Appearances: {summary.get('total_appearances', 'N/A')}")
    params = summary.get('clustering_parameters', {})
    print(f"\nClustering Parameters:")
    print(f"  EPS:         {params.get('eps', 'N/A')}")
    print(f"  MIN_SAMPLES: {params.get('min_samples', 'N/A')}")
    print("=" * 60 + "\n")

def list_all_ids(data):
    """List all global IDs with appearance counts"""
    identities = data.get("identities", {})
    print("\n" + "=" * 60)
    print("📋 ALL GLOBAL IDs")
    print("=" * 60)
    print(f"{'ID':<15} {'Appearances':<15} {'Clips'}")
    print("-" * 60)
    
    for gid_key in sorted(identities.keys(), key=lambda x: int(x.split('_')[-1])):
        gid = int(gid_key.split('_')[-1])
        appearances = identities[gid_key]["appearances"]
        clips = sorted(set(a["clip_id"] for a in appearances))
        print(f"{gid:<15} {len(appearances):<15} {clips}")
    print("=" * 60 + "\n")

def show_id_details(data, target_id):
    """Show detailed information for a specific global ID"""
    identities = data.get("identities", {})
    gid_key = f"global_id_{target_id}"
    
    if gid_key not in identities:
        print(f"\n❌ Global ID {target_id} not found in catalogue.")
        print(f"Available IDs: {sorted([int(k.split('_')[-1]) for k in identities.keys()])}\n")
        return
    
    person = identities[gid_key]
    appearances = person["appearances"]
    
    print("\n" + "=" * 60)
    print(f"👤 GLOBAL ID {target_id}")
    print("=" * 60)
    print(f"Total Appearances: {len(appearances)}")
    
    # Group by clip
    clips = {}
    for app in appearances:
        clip_id = app["clip_id"]
        if clip_id not in clips:
            clips[clip_id] = []
        clips[clip_id].append(app)
    
    print(f"Appears in {len(clips)} clip(s): {sorted(clips.keys())}")
    print("\n" + "-" * 60)
    
    for clip_id in sorted(clips.keys()):
        print(f"\n📹 Clip {clip_id}:")
        for i, app in enumerate(clips[clip_id], 1):
            start, end = app["frame_range"]
            t_start, t_end = app["time_span"]
            duration = t_end - t_start
            print(f"  {i}. Frames {start:4d}-{end:4d}  |  Time: {t_start:6.2f}s - {t_end:6.2f}s  ({duration:.2f}s)")
    
    print("=" * 60 + "\n")

def show_statistics(data):
    """Show detailed statistics"""
    identities = data.get("identities", {})
    summary = data.get("summary", {})
    
    print("\n" + "=" * 60)
    print("📈 DETAILED STATISTICS")
    print("=" * 60)
    
    # Collect stats
    appearance_counts = []
    clip_coverage = []
    total_duration_per_id = []
    
    for gid_key in identities:
        appearances = identities[gid_key]["appearances"]
        appearance_counts.append(len(appearances))
        clips = set(a["clip_id"] for a in appearances)
        clip_coverage.append(len(clips))
        total_duration = sum(a["time_span"][1] - a["time_span"][0] for a in appearances)
        total_duration_per_id.append(total_duration)
    
    print(f"\nGlobal IDs: {len(identities)}")
    print(f"Total Appearances: {sum(appearance_counts)}")
    
    print(f"\nAppearances per ID:")
    print(f"  Min:     {min(appearance_counts)}")
    print(f"  Max:     {max(appearance_counts)}")
    print(f"  Average: {sum(appearance_counts) / len(appearance_counts):.1f}")
    
    print(f"\nClip Coverage per ID:")
    print(f"  Min:     {min(clip_coverage)} clip(s)")
    print(f"  Max:     {max(clip_coverage)} clip(s)")
    print(f"  Average: {sum(clip_coverage) / len(clip_coverage):.1f} clip(s)")
    
    print(f"\nTotal Screen Time per ID:")
    print(f"  Min:     {min(total_duration_per_id):.2f}s")
    print(f"  Max:     {max(total_duration_per_id):.2f}s")
    print(f"  Average: {sum(total_duration_per_id) / len(total_duration_per_id):.2f}s")
    
    # Show top IDs by appearances
    print(f"\n🏆 Top IDs by Appearances:")
    sorted_ids = sorted(identities.items(), key=lambda x: len(x[1]["appearances"]), reverse=True)
    for i, (gid_key, person) in enumerate(sorted_ids[:5], 1):
        gid = int(gid_key.split('_')[-1])
        count = len(person["appearances"])
        duration = sum(a["time_span"][1] - a["time_span"][0] for a in person["appearances"])
        print(f"  {i}. ID {gid:2d}: {count:2d} appearances ({duration:.1f}s screen time)")
    
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Query global identity catalogue")
    parser.add_argument("--id", type=int, help="Show details for specific global ID")
    parser.add_argument("--list", action="store_true", help="List all global IDs")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    parser.add_argument("--file", default="global_identity_catalogue.json", help="JSON file to read")
    args = parser.parse_args()
    
    try:
        data = load_catalogue(args.file)
    except FileNotFoundError:
        print(f"\n❌ Error: File '{args.file}' not found.")
        print("Run clustering first: python cluster_embeddings.py --eps 0.35 --min_samples 2\n")
        return
    except json.JSONDecodeError:
        print(f"\n❌ Error: '{args.file}' is not a valid JSON file.\n")
        return
    
    if args.id is not None:
        show_id_details(data, args.id)
    elif args.list:
        list_all_ids(data)
    elif args.stats:
        show_statistics(data)
    else:
        show_summary(data)

if __name__ == "__main__":
    main()

