from data_loading_utils import load_sfm_data
from src.Wrapper import StructureFromMotion



# Test loading
data_dir = "Data"
matches, K, n_images = load_sfm_data(data_dir)

StructureFromMotion(matches, K, n_images)