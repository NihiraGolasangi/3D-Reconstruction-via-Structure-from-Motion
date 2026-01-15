import numpy as np


def parse_matching_file(filepath, current_img_idx):
    """
    Parse a matching file.
    
    Args:
        filepath: Path to matchingX.txt
        current_img_idx: Index of current image (e.g., 1 for matching1.txt)
        
    Returns:
        matches_dict: Dictionary mapping (current_img, other_img) -> correspondences
    """
    matches_dict = {}
    
    with open(filepath, 'r') as f:
        # Read header
        first_line = f.readline()
        n_features = int(first_line.split(':')[1].strip())
        
        print(f"Reading matching{current_img_idx}.txt: {n_features} features")
        
        # Read each feature line
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            # Parse the line
            n_matches = int(parts[0])
            r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
            u_current, v_current = float(parts[4]), float(parts[5])
            
            # Parse matches with other images
            idx = 6
            while idx + 2 < len(parts):
                other_img_idx = int(parts[idx])
                u_other = float(parts[idx + 1])
                v_other = float(parts[idx + 2])
                
                # Create key (always current < other since file is forward-looking)
                key = (current_img_idx, other_img_idx)
                
                # Initialize if first correspondence for this pair
                if key not in matches_dict:
                    matches_dict[key] = {
                        'pts1': [],
                        'pts2': [],
                        'colors': []
                    }
                
                # Add correspondence
                matches_dict[key]['pts1'].append([u_current, v_current])
                matches_dict[key]['pts2'].append([u_other, v_other])
                matches_dict[key]['colors'].append([r, g, b])
                
                idx += 3
    
    # Convert lists to numpy arrays
    for key in matches_dict:
        matches_dict[key]['pts1'] = np.array(matches_dict[key]['pts1'])
        matches_dict[key]['pts2'] = np.array(matches_dict[key]['pts2'])
        matches_dict[key]['colors'] = np.array(matches_dict[key]['colors'])
    
    return matches_dict




def load_all_matches(data_dir, n_images=6):
    """
    Load all matching files and combine into one dictionary.
    
    Args:
        data_dir: Directory containing matching files
        n_images: Number of images (default: 6)
        
    Returns:
        all_matches: Dictionary mapping (i, j) -> correspondences
    """
    all_matches = {}
    
    # Parse each matching file
    for i in range(1, n_images):  # matching1.txt to matching5.txt
        filepath = f"{data_dir}/matching{i}.txt"
        
        try:
            matches = parse_matching_file(filepath, current_img_idx=i)
            
            # Merge into all_matches
            for key, value in matches.items():
                all_matches[key] = value
                print(f"  Pair {key}: {len(value['pts1'])} correspondences")
        
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
    
    print(f"\nTotal image pairs: {len(all_matches)}")
    
    return all_matches


def load_calibration(filepath):
    """
    Load camera intrinsic matrix K from calibration file.
    
    The file format is:
    K = [fx  0   cx;
         0   fy  cy;
         0   0   1]
    
    Args:
        filepath: Path to calibration.txt
        
    Returns:
        K: 3x3 camera intrinsic matrix
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove "K = " prefix and brackets
    content = content.replace('K = ', '')
    content = content.replace('[', '')
    content = content.replace(']', '')
    content = content.replace(';', '')
    
    # Split into values and convert to floats
    values = content.split()
    values = [float(v) for v in values]
    
    # Reshape into 3x3 matrix
    K = np.array(values).reshape(3, 3)
    
    print(f"Camera intrinsics K:")
    print(K)
    print(f"Focal lengths: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    
    return K

def load_sfm_data(data_dir):
    """
    Load all data needed for SfM.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        matches: Dictionary of correspondences
        K: Camera intrinsic matrix
        n_images: Number of images
    """
    print("=" * 50)
    print("Loading SfM Data")
    print("=" * 50)
    
    # Load camera calibration
    print("\n1. Loading camera calibration...")
    K = load_calibration(f"{data_dir}/calibration.txt")
    
    # Load all matches
    print("\n2. Loading feature matches...")
    matches = load_all_matches(data_dir, n_images=6)
    
    print("\n" + "=" * 50)
    print("Data loading complete!")
    print("=" * 50)
    
    return matches, K, 6