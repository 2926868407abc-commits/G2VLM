import json
import os
from pathlib import Path

from .vlm_dataset import SftJSONLIterableDataset
from .recon_dataset import SftJSONLIterableReconDataset
from .interleave_datasets.recon_then_und_dataset import ReconthenUndIterableDataset

INTERNDATA_N1_REPLICA_ROOT = Path(
	os.environ.get(
		"G2VLM_INTERNDATA_N1_REPLICA_ROOT",
		"/mnt/data/wangqq/G2VLM/data/g2vlm_interndata_n1/replica_d435i",
	)
)


def _interndata_n1_replica_info():
	info = {
		'data_dir': str(INTERNDATA_N1_REPLICA_ROOT / 'parquets'),
		'num_files': 1,
		'num_total_samples': 0,
		"parquet_info_path": str(INTERNDATA_N1_REPLICA_ROOT / 'parquet_info.json'),
	}
	snippet_path = INTERNDATA_N1_REPLICA_ROOT / "dataset_info_snippet.json"
	if snippet_path.exists():
		try:
			with snippet_path.open("r", encoding="utf-8") as f:
				info.update(json.load(f))
		except Exception as exc:
			print(f"Warning: failed to read {snippet_path}: {exc}")
	return info


DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,
    'recon_then_und': ReconthenUndIterableDataset,
    'recon': SftJSONLIterableReconDataset, 
}

DATASET_INFO = {
	'vlm_sft':{
        'llava_ov': {
			'data_dir': 'your_data_path/g2vlm_example/vlm/images',
			'jsonl_path': 'your_data_path/g2vlm_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },

	'recon_then_und':{
		'spatial_mix': {
			'data_dir': 'your_data_path/g2vlm_example/joint_trainng/images',
			'num_files': 10,
			'num_total_samples': 1000,
			"parquet_info_path": 'your_data_path/g2vlm_example/joint_trainng/parquet_info', # information of the parquet files
		},
		'intern_n1_replica_d435i': _interndata_n1_replica_info(),
	},
    
    'recon': {
		'scannet': {
			'data_dir': 'your_data_path/g2vlm_example/recon/images',
			'jsonl_path': 'your_data_path/g2vlm_example/recon/scannet.jsonl',
			'num_total_samples': 2000
		},
    },
}
