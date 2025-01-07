"""Script to merge binary classification predictions"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple


###------ Merge Function -------###
def merge_binary_predictions(csv_paths: List[str], 
                           binary_groups: List[List[str]], 
                           output_path: str) -> None:
    """Merge binary classification predictions into final predictions
    
    Args:
        csv_paths (List[str]): List of paths to binary prediction CSV files
        binary_groups (List[List[str]]): List of binary classification groups
            Each inner list should contain [class_0, class_1] where
            class_0 is the target class and class_1 can be multiple classes joined by space
        output_path (str): Path to save merged predictions
    """
    ### Read first CSV as base predictions
    df = pd.read_csv(csv_paths[0])
    print(f"\nBase predictions from {csv_paths[0]}")
    print(df['label'].value_counts())
    
    ### Get initial classes from first binary group
    base_other_classes = binary_groups[0][1].split()
    base_label = '+'.join(base_other_classes)
    
    ### Process each subsequent binary classification
    for i in range(1, len(csv_paths)):
        print(f"\nProcessing {csv_paths[i]}")
        current_df = pd.read_csv(csv_paths[i])
        
        ### Create replacement mask using base label
        replace_mask = df['label'] == base_label
        
        ### Show current distribution
        print("Current distribution in target rows:")
        print(current_df.loc[replace_mask, 'label'].value_counts())
        
        ### Replace predictions
        df.loc[replace_mask, 'label'] = current_df.loc[replace_mask, 'label']
        
        ### Show updated distribution
        print("\nUpdated overall distribution:")
        print(df['label'].value_counts())
        
        ### Update base label for next iteration if needed
        if i < len(csv_paths) - 1:
            next_other_classes = binary_groups[i][1].split()
            base_label = '+'.join(next_other_classes)
    
    ### Clean up any remaining 'bothcells' in labels
    bothcells_mask = df['label'].str.contains('bothcells')
    if bothcells_mask.any():
        print("\nCleaning up 'bothcells' from final labels...")
        print("Before cleanup:")
        print(df['label'].value_counts())
        
        ### Remove 'bothcells+' and '+bothcells' patterns
        df['label'] = df['label'].replace({
            'bothcells\+': '',  ### Remove 'bothcells+'
            '\+bothcells': ''   ### Remove '+bothcells'
        }, regex=True)
        
        print("\nAfter cleanup:")
        print(df['label'].value_counts())
    
    ### Save merged predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nMerged predictions saved to {output_path}")


###------ Main Function -------###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_paths', nargs='+', required=True,
                        help='Paths to binary prediction CSV files (in order)')
    parser.add_argument('--class_0', nargs='+', required=True,
                        help='Target classes for each CSV (in order)')
    parser.add_argument('--class_1', nargs='+', required=True,
                        help='Other classes for each CSV (space-separated, in order)')
    parser.add_argument('--output_path', type=str, default='submission/merged_predictions.csv',
                        help='Path to save merged predictions')
    args = parser.parse_args()
    
    ### Validate input lengths
    if not (len(args.csv_paths) == len(args.class_0) == len(args.class_1)):
        raise ValueError(f"Number of inputs must match: {len(args.csv_paths)=} != {len(args.class_0)=} != {len(args.class_1)=}")
    
    ### Create binary groups from inputs
    binary_groups = list(zip(args.class_0, args.class_1))
    
    ### Print merge plan
    print("\nMerge Plan:")
    for i, (csv_path, group) in enumerate(zip(args.csv_paths, binary_groups)):
        print(f"\nStage {i+1}:")
        print(f"CSV: {csv_path}")
        print(f"Group: {group[0]} vs {group[1]}")
    
    ### Merge predictions
    merge_binary_predictions(args.csv_paths, binary_groups, args.output_path)


if __name__ == '__main__':
    main() 