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

import pandas as pd
import numpy as np
import datetime
from collections import defaultdict, Counter
import os

INPUT_FILE = "./Nimda_RRC04_Raw/nimda_baseline_14sept_raw.csv"
OUTPUT_FILE = "./Nimda_RRC04_Raw/nimda_baseline_14sept_features_1s.csv"
WINDOW_SIZE = '1s'  # 1-second aggregation window
LABEL_STRATEGY = 'majority'  

# ==============================================================================
# ORIGINAL FEATURE EXTRACTION LOGIC (UNCHANGED)
# ==============================================================================
def calculate_edit_distance(as_path1, as_path2):
    """ Calculate edit distance between two AS paths """
    if not as_path1 or not as_path2: return 0
    
    if isinstance(as_path1, int): as_path1 = [as_path1]
    if isinstance(as_path2, int): as_path2 = [as_path2]
    
    if isinstance(as_path1, str):
        as_path1 = as_path1.replace('{', '').replace('}', '')
        as_path1 = [int(as_num) for as_num in as_path1.split() if as_num.isdigit()]
    if isinstance(as_path2, str):
        as_path2 = as_path2.replace('{', '').replace('}', '')
        as_path2 = [int(as_num) for as_num in as_path2.split() if as_num.isdigit()]
    
    if not as_path1 or not as_path2: return 0
    m, n = len(as_path1), len(as_path2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if as_path1[i-1] == as_path2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def extract_features(df_window):
    """ Extract BGP features from a dataframe within a specific time window """
    features = {}

    announcements = df_window[df_window['Subtype'] == 'ANNOUNCE']
    withdrawal_types = ['WITHDRAW', 'WITHDRAW_MP_UNREACH_NLRI_AFI2']
    withdrawals = df_window[df_window['Subtype'].isin(withdrawal_types)]
    
    features['announcements'] = len(announcements)
    features['withdrawals'] = len(withdrawals)
    features['nlri_ann'] = features['announcements']
    
    if not announcements.empty:
        dup_cols = ['Peer_IP', 'Peer_ASN', 'Prefix', 'AS_Path', 'Origin', 'Next_Hop', 'MED', 'Local_Pref', 'Communities']
        dup_cols = [col for col in dup_cols if col in announcements.columns]
        announcement_counts = announcements.groupby(dup_cols).size()
        duplicates = sum(count - 1 for count in announcement_counts if count > 1)
        features['dups'] = duplicates
    else:
        features['dups'] = 0
    
    if not announcements.empty and 'Origin' in announcements.columns:
        origin_counts = announcements['Origin'].value_counts()
        features['origin_0'] = origin_counts.get('IGP', 0)  
        features['origin_2'] = origin_counts.get('INCOMPLETE', 0)  
        if not announcements.empty:
            unique_prefix_origins = announcements.groupby('Prefix')['Origin'].nunique()
            features['origin_changes'] = (unique_prefix_origins > 1).sum()
        else:
            features['origin_changes'] = 0
    else:
        features['origin_0'] = 0
        features['origin_2'] = 0
        features['origin_changes'] = 0
    
    if not announcements.empty and 'AS_Path' in announcements.columns:
        valid_as_paths = announcements[announcements['AS_Path'].notna() & (announcements['AS_Path'] != '')]
        if not valid_as_paths.empty:
            as_path_lengths = valid_as_paths['AS_Path'].apply(
                lambda path: len([p for p in path.split() if p.isdigit()]) if isinstance(path, str) else 0
            )
            features['as_path_max'] = as_path_lengths.max() if not as_path_lengths.empty else 0
            unique_paths_per_prefix = valid_as_paths.groupby('Prefix')['AS_Path'].nunique()
            features['unique_as_path_max'] = unique_paths_per_prefix.max() if not unique_paths_per_prefix.empty else 0
            
            edit_distances = []
            edit_distance_dict = defaultdict(list)
            
            for prefix, group in valid_as_paths.groupby('Prefix'):
                if len(group) >= 2:
                    sorted_group = group.sort_values('Timestamp')
                    prev_path = None
                    for _, row in sorted_group.iterrows():
                        current_path = row['AS_Path']
                        if prev_path is not None:
                            dist = calculate_edit_distance(prev_path, current_path)
                            edit_distances.append(dist)
                            edit_distance_dict[prefix].append(dist)
                        prev_path = current_path
            
            if edit_distances:
                features['edit_distance_avg'] = np.mean(edit_distances)
                features['edit_distance_max'] = max(edit_distances)
                edit_dist_counter = Counter(edit_distances)
                for i in range(7): features[f'edit_distance_dict_{i}'] = edit_dist_counter.get(i, 0)
                
                unique_edit_dists = {}
                for prefix, dists in edit_distance_dict.items():
                    unique_dists = set(dists)
                    for dist in unique_dists:
                        if dist in unique_edit_dists: unique_edit_dists[dist] += 1
                        else: unique_edit_dists[dist] = 1
                for i in range(2): features[f'edit_distance_unique_dict_{i}'] = unique_edit_dists.get(i, 0)
            else:
                features['edit_distance_avg'] = 0
                features['edit_distance_max'] = 0
                for i in range(7): features[f'edit_distance_dict_{i}'] = 0
                for i in range(2): features[f'edit_distance_unique_dict_{i}'] = 0
        else:
            features['as_path_max'] = 0
            features['unique_as_path_max'] = 0
            features['edit_distance_avg'] = 0
            features['edit_distance_max'] = 0
            for i in range(7): features[f'edit_distance_dict_{i}'] = 0
            for i in range(2): features[f'edit_distance_unique_dict_{i}'] = 0
    else:
        features['as_path_max'] = 0
        features['unique_as_path_max'] = 0
        features['edit_distance_avg'] = 0
        features['edit_distance_max'] = 0
        for i in range(7): features[f'edit_distance_dict_{i}'] = 0
        for i in range(2): features[f'edit_distance_unique_dict_{i}'] = 0
    
    if not announcements.empty:
        prefix_peer_groups = announcements.groupby(['Prefix', 'Peer_IP'])
        imp_wd_prefixes = 0
        imp_wd_spath_prefixes = 0
        imp_wd_dpath_prefixes = 0
        for (prefix, peer), group in prefix_peer_groups:
            if len(group) > 1:
                imp_wd_prefixes += 1
                if 'AS_Path' in group.columns:
                    unique_paths = group['AS_Path'].nunique()
                    if unique_paths == 1: imp_wd_spath_prefixes += 1
                    else: imp_wd_dpath_prefixes += 1
        features['imp_wd'] = imp_wd_prefixes
        features['imp_wd_spath'] = imp_wd_spath_prefixes
        features['imp_wd_dpath'] = imp_wd_dpath_prefixes
    else:
        features['imp_wd'] = 0
        features['imp_wd_spath'] = 0
        features['imp_wd_dpath'] = 0

    if not announcements.empty and 'AS_Path' in announcements.columns:
        all_asns = []
        for as_path in announcements['AS_Path']:
            if pd.isnull(as_path) or as_path == '': continue
            as_path_str = str(as_path)
            if as_path_str.isdigit():
                all_asns.append(as_path_str)
                continue
            as_path_str = as_path_str.replace('{', '').replace('}', '')
            path_asns = [asn for asn in as_path_str.split() if asn.isdigit()]
            all_asns.extend(path_asns)
        asn_counts = Counter(all_asns)
        rare_threshold = 3
        rare_asns = [asn for asn, count in asn_counts.items() if count < rare_threshold]
        features['number_rare_ases'] = len(rare_asns)
        features['rare_ases_avg'] = len(rare_asns) / len(all_asns) if all_asns else 0
    else:
        features['number_rare_ases'] = 0
        features['rare_ases_avg'] = 0
        
    if not df_window.empty:
        if not withdrawals.empty and not announcements.empty:
            withdrawn_prefixes = set(withdrawals['Prefix'].dropna())
            announced_prefixes = set(announcements['Prefix'].dropna())
            flapped_prefixes = withdrawn_prefixes.intersection(announced_prefixes)
            features['flaps'] = len(flapped_prefixes)
        else:
            features['flaps'] = 0
        
        very_specific_prefixes = 0
        if 'Prefix' in df_window.columns:
            very_specific_prefixes = sum(1 for prefix in df_window['Prefix'].dropna() 
                                       if isinstance(prefix, str) and prefix.endswith('/32'))
        wd_ann_ratio = features['withdrawals'] / features['announcements'] if features['announcements'] > 0 else 0
        features['nadas'] = very_specific_prefixes + (wd_ann_ratio > 0.5) * 10
    else:
        features['flaps'] = 0
        features['nadas'] = 0
    
    if 'Label' in df_window.columns:
        labels = df_window['Label'].value_counts()
        if not labels.empty:
            if LABEL_STRATEGY == 'majority': features['label'] = labels.idxmax()
            elif LABEL_STRATEGY == 'conservative':
                if any(label != 'normal' for label in labels.index):
                    abnormal_labels = [label for label in labels.index if label != 'normal']
                    features['label'] = abnormal_labels[0]
                else: features['label'] = 'normal'
            elif LABEL_STRATEGY == 'weighted':
                total = labels.sum()
                abnormal_weight = sum(count for label, count in labels.items() if label != 'normal') / total
                if abnormal_weight > 0.4:
                    abnormal_labels = [label for label in labels.index if label != 'normal']
                    features['label'] = abnormal_labels[0] if abnormal_labels else 'normal'
                else: features['label'] = 'normal'
        else: features['label'] = 'unknown'
    else: features['label'] = 'unknown'
    
    return features

# ==============================================================================
# PROCESAMIENTO PRINCIPAL ADAPTADO
# ==============================================================================
def process_bgp_data():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: RAW file not found at {INPUT_FILE}")
        return None

    print(f"Reading RAW file: {INPUT_FILE}")
    print("This may take a while depending on file size...")
    
    # Read CSV ensuring data types do not cause issues
    df = pd.read_csv(INPUT_FILE, dtype=str)
    
    # --- COLUMN NAME MAPPING ---
    # Rename columns from our extractor format to the expected feature extraction format
    print("Adapting data format...")
    df = df.rename(columns={
        'Time': 'Timestamp',
        'Entry_Type': 'Subtype',
        'Peer_AS': 'Peer_ASN',
        'Community': 'Communities'
    })
    
    # Map message types: A -> ANNOUNCE, W -> WITHDRAW
    df['Subtype'] = df['Subtype'].map({'A': 'ANNOUNCE', 'W': 'WITHDRAW'}).fillna(df['Subtype'])


    # Convertir Timestamp a formato datetime real para Pandas
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()
    print(f"Time range: {start_time} to {end_time}")
    
    df.set_index('Timestamp', inplace=True)
    all_features = []
    grouped = df.groupby(pd.Grouper(freq=WINDOW_SIZE))
    
    window_count = 0
    total_windows = len(grouped)
    
    print(f"Starting feature extraction in {WINDOW_SIZE} windows...")
    
    for window_start, window_df in grouped:
        if not window_df.empty:
            window_df = window_df.reset_index()
            features = extract_features(window_df)
            
            if features:
                window_end = window_start + pd.Timedelta(WINDOW_SIZE)
                features['window_start'] = window_start
                features['window_end'] = window_end
                all_features.append(features)
                window_count += 1
                
                if window_count % 100 == 0:
                    print(f"Processed {window_count}/{total_windows} windows (with activity)...")
    
    print(f"Total windows processed successfully: {window_count}")
    
    if all_features:
        features_df = pd.DataFrame(all_features)
        
        required_features = [
            'dups', 'edit_distance_avg', 'edit_distance_dict_0', 'edit_distance_dict_1',
            'nlri_ann', 'origin_0', 'origin_2', 'imp_wd', 'rare_ases_avg', 'imp_wd_spath',
            'unique_as_path_max', 'edit_distance_dict_2', 'edit_distance_dict_4', 
            'edit_distance_dict_6', 'edit_distance_max', 'edit_distance_unique_dict_0',
            'edit_distance_unique_dict_1', 'announcements', 'origin_changes', 'flaps',
            'nadas', 'number_rare_ases', 'withdrawals', 'as_path_max', 'imp_wd_dpath'
        ]
        
        missing_features = [feature for feature in required_features if feature not in features_df.columns]
        if missing_features:
            for feature in missing_features:
                features_df[feature] = 0
        
        # Save final result
        features_df.to_csv(OUTPUT_FILE, index=False)
        print(f"SUCCESS! Features saved to: {OUTPUT_FILE}")
        return features_df
    else:
        print("No features extracted. Please check the input data.")
        return None

if __name__ == "__main__":
    process_bgp_data()