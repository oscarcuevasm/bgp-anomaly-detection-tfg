# ==============================================================================
# Copyright (c) UNIVERSIDAD AUTÓNOMA DE MADRID
# Francisco Tomás y Valiente, no 1
# Madrid, 28049
# Spain
#
# Óscar Cuevas Martínez
# Evaluating the Performance of BGP Different Anomaly Detection Methods
# All Rights Reserved
# ==============================================================================

"""
Automatically reads the RIPE directory and downloads all available files
for the specified date, regardless of whether the interval is 5 or 15 minutes.
"""

import os
import gzip
import shutil
import subprocess
import csv
import re
from urllib.request import urlopen
from pathlib import Path

# --- PATH CONFIGURATION ---
RIPE_DIR = "./CodeRed_RRC04_Raw"
OUTPUT_DIR = os.path.join(RIPE_DIR, "mrt_files")
TEMP_DIR = os.path.join(RIPE_DIR, "temp_mrt")

# --- EXACT DATES TO RETRIEVE (full day) ---
# (Date_String, Output_File)
TARGETS = [
    ("20010712", os.path.join(RIPE_DIR, "codered_baseline_12jul_raw.csv")),
    ("20010719", os.path.join(RIPE_DIR, "codered_anomaly_19jul_raw.csv"))
]

fieldnames = ['MRT_Type', 'Time', 'Entry_Type', 'Peer_IP', 'Peer_AS',
              'Prefix', 'AS_Path', 'Origin', 'Next_Hop', 'Local_Pref',
              'MED', 'Community', 'Atomic_Aggregate', 'Aggregator', 'Label']

def create_directories():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

def get_available_files_from_ripe(date_str):
    """
    Connects to the RIPE directory listing and retrieves the real filenames
    for a specific date, avoiding the need to guess the minute intervals.
    """
    year_month = f"{date_str[:4]}.{date_str[4:6]}"
    url = f"https://data.ris.ripe.net/rrc04/{year_month}/"
    print(f" -> Scanning web directory: {url}")
    
    try:
        html = urlopen(url).read().decode('utf-8')
        # Search for files matching the pattern: updates.YYYYMMDD.HHMM.gz
        pattern = rf'href="(updates\.{date_str}\.\d{{4}}\.gz)"'
        files = re.findall(pattern, html)
        # Remove duplicates and sort chronologically
        return sorted(list(set(files)))
    except Exception as e:
        print(f"Error accessing RIPE: {e}")
        return []

def download_file(url, local_path):
    try:
        with urlopen(url) as response:
            with open(local_path, 'wb') as out_file:
                out_file.write(response.read())
        return True
    except Exception:
        return False

def decompress_gz(gz_file, output_file):
    try:
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        print(f"✗ Error al descomprimir {gz_file}: {e}")
        return False

# --- BGP EXTRACTION LOGIC ---
def parse_bgpdump_line(line):
    line = line.strip()
    if not line: return None
    parts = line.split('|')
    if len(parts) < 6: return None

    try:
        msg_type = parts[0]
        if msg_type != 'BGP4MP': return None

        timestamp = int(parts[1])
        # Robust timestamp handling
        from datetime import datetime
        dt = datetime.utcfromtimestamp(timestamp)
        date_time = dt.strftime('%Y-%m-%d %H:%M:%S')

        update_type = parts[2]
        peer_ip = parts[3]
        peer_as = parts[4]
        prefix = parts[5]

        if update_type == 'W':
            return {
                'MRT_Type': 'BGP4MP', 'Time': date_time, 'Entry_Type': 'W',
                'Peer_IP': peer_ip, 'Peer_AS': peer_as, 'Prefix': prefix,
                'AS_Path': '', 'Origin': '', 'Next_Hop': '', 'Local_Pref': '',
                'MED': '', 'Community': '', 'Atomic_Aggregate': '', 'Aggregator': '',
                'Label': 'normal'
            }

        as_path = parts[6] if len(parts) > 6 else ''
        origin = parts[7] if len(parts) > 7 else ''
        next_hop = parts[8] if len(parts) > 8 else ''
        local_pref = parts[9] if len(parts) > 9 else ''
        med = parts[10] if len(parts) > 10 else ''
        community = parts[11] if len(parts) > 11 else ''
        atomic_agg = parts[12] if len(parts) > 12 else ''
        aggregator = parts[13] if len(parts) > 13 else ''

        return {
            'MRT_Type': 'BGP4MP', 'Time': date_time, 'Entry_Type': 'A',
            'Peer_IP': peer_ip, 'Peer_AS': peer_as, 'Prefix': prefix,
            'AS_Path': as_path, 'Origin': origin, 'Next_Hop': next_hop,
            'Local_Pref': local_pref, 'MED': med, 'Community': community,
            'Atomic_Aggregate': atomic_agg, 'Aggregator': aggregator,
            'Label': 'normal'
        }
    except (ValueError, IndexError):
        return None

def parse_mrt_file_with_bgpdump(mrt_file):
    records = []
    try:
        result = subprocess.run(['bgpdump', '-m', mrt_file], capture_output=True, text=True, check=False)
        if result.returncode != 0: return []
        lines = result.stdout.strip().split('\n')
        for line in lines:
            record = parse_bgpdump_line(line)
            if record: records.append(record)
    except Exception:
        pass
    return records

def collect_and_process_updates():
    create_directories()
    
    for date_str, csv_output in TARGETS:
        print("\n" + "=" * 70)
        print(f"PROCESANDO DÍA: {date_str} -> Guardando en {os.path.basename(csv_output)}")
        print("=" * 70)

        # 1. Retrieve the real list of files from the server
        filenames = get_available_files_from_ripe(date_str)
        if not filenames:
            print(f"No files found for date {date_str}.")
            continue
            
        print(f" -> Found {len(filenames)} valid files on the server.")

        # 2. Download and process files
        year_month = f"{date_str[:4]}.{date_str[4:6]}"
        base_url = f"https://data.ris.ripe.net/rrc04/{year_month}"
        
        # Write CSV headers
        with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            writer.writeheader()

        total_records = 0
        total_announcements = 0
        total_withdrawals = 0

        for i, filename in enumerate(filenames, 1):
            url = f"{base_url}/{filename}"
            local_path = os.path.join(OUTPUT_DIR, filename)

            # Download file
            if not os.path.exists(local_path):
                print(f"[{i}/{len(filenames)}] Downloading {filename}...")
                if not download_file(url, local_path):
                    continue
            else:
                print(f"[{i}/{len(filenames)}] Processing {filename} (already downloaded)...", end=" ")

            mrt_file = os.path.join(TEMP_DIR, filename.replace('.gz', ''))
            
            if not os.path.exists(local_path): continue

            # Descomprimir
            if not decompress_gz(local_path, mrt_file):
                print("Failed decompress")
                continue

            # Parsear bgpdump
            records = parse_mrt_file_with_bgpdump(mrt_file)
            
            # Guardar en CSV
            if records:
                with open(csv_output, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
                    writer.writerows(records)
                    
                total_records += len(records)
                total_announcements += sum(1 for r in records if r['Entry_Type'] == 'A')
                total_withdrawals += sum(1 for r in records if r['Entry_Type'] == 'W')
                print(f"✓ {len(records)} registros")
            else:
                print("✓ 0 registros")

            try:
                os.remove(mrt_file)
            except:
                pass

        print(f"\nSummary for {os.path.basename(csv_output)}:")
        print(f"✓ Total packets: {total_records:,} (A: {total_announcements:,} | W: {total_withdrawals:,})")

    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass

if __name__ == "__main__":
    collect_and_process_updates()