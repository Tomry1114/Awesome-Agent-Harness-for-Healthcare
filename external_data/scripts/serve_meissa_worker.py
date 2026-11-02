"""Launch the pinned upstream worker with local logs and a deployment identity endpoint."""
import argparse
import base64
import hashlib
import io
import logging
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_biomed_masks(masks, prompt_count):
    """Preserve the upstream N,H,W predictions as one H,W mask per prompt.

    The Meissa overlay worker otherwise wraps the whole ndarray in one list,
    confusing the prompt axis with a channel axis for multi-prompt requests.
    """
    import numpy as np
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2 and prompt_count == 1:
            masks = masks[None]
        if masks.ndim != 3 or masks.shape[0] != prompt_count:
            raise ValueError('BioMedParse mask count/shape does not match prompts')
        masks = list(masks)
    if not isinstance(masks, (list, tuple)) or len(masks) != prompt_count:
        raise ValueError('BioMedParse must return one mask per prompt')
    if any(np.asarray(mask).ndim != 2 or not np.isfinite(mask).all() for mask in masks):
        raise ValueError('BioMedParse returned an invalid mask')
    return list(masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', choices=['sam2', 'biomedparse'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--biomedparse-config', type=Path, help='Optional config with a local PRETRAINED_TOKENIZER directory')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    checkpoint_sha = digest(checkpoint)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('HF_HOME', str(ROOT / 'external_data/runtime/hf'))
    os.environ.setdefault('HF_DATASETS_OFFLINE', '0')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')
    os.environ.setdefault('NLTK_DATA', str(ROOT / 'external_data/runtime/nltk'))
    upstream = ROOT / 'references/Meissa/environments/interleaved_thinking_images'
    sys.path.insert(0, str(upstream))
    # Keep the read-only upstream checkout free of worker-created logs.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    import tool_server.utils.server_utils as server_utils
    server_utils.build_logger = lambda name, *args, **kwargs: logging.getLogger(Path(name).stem)
    # Both released workers call these helpers, but the pinned utility module
    # only defines load_image_from_base64. Supply codecs without editing upstream.
    from PIL import Image
    def decode_image(encoded):
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            return image.convert('RGB')
    def encode_image(image):
        stream = io.BytesIO()
        image.convert('RGB').save(stream, format='PNG')
        return base64.b64encode(stream.getvalue()).decode('ascii')
    server_utils.base64_to_pil = decode_image
    server_utils.pil_to_base64 = encode_image
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU validation requires CUDA; refusing a silent CPU fallback')
    common = {'controller_addr': '', 'no_register': True, 'host': args.host,
              'port': args.port, 'limit_model_concurrency': 1}
    if args.tool == 'sam2':
        from tool_server.tool_workers.online_workers.SAMAroundPoint_worker import SAM2ToolWorker
        worker = SAM2ToolWorker(sam2_checkpoint=str(checkpoint),
                               sam2_model_cfg='configs/sam2/sam2_hiera_l.yaml', **common)
        model_repo = ROOT / 'references/sam2'
    else:
        model_repo = ROOT / 'references/BiomedParse'
        os.environ['BIOMEDPARSE_ROOT'] = str(model_repo)
        sys.path.insert(0, str(model_repo))
        from tool_server.tool_workers.online_workers import BioMedParseTextSeg_worker as biomed_worker
        original_infer = biomed_worker.interactive_infer_image
        def infer_masks(model, image, prompts):
            return split_biomed_masks(original_infer(model, image, prompts), len(prompts))
        biomed_worker.interactive_infer_image = infer_masks
        BioMedParseToolWorker = biomed_worker.BioMedParseToolWorker
        biomed_config = (args.biomedparse_config or model_repo / 'configs/biomedparse_inference.yaml').resolve(strict=True)
        worker = BioMedParseToolWorker(biomedparse_ckpt=str(checkpoint),
            biomedparse_cfg=str(biomed_config),
            biomedparse_repo_root=str(model_repo), **common)
    identity = {'tool': args.tool, 'checkpoint_sha256': checkpoint_sha,
                'compatibility_shims': ['missing upstream base64_to_pil/pil_to_base64; RGB PNG transport',
                                        'upstream logs redirected; SAM2 config package path'],
                'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'slurm_job': os.environ.get('SLURM_JOB_ID'),
                'source_commit': subprocess.check_output(['git', '-C', str(ROOT / 'references/Meissa'), 'rev-parse', 'HEAD'], text=True).strip(),
                'model_code_commit': subprocess.check_output(['git', '-C', str(model_repo), 'rev-parse', 'HEAD'], text=True).strip()}
    if args.tool == 'biomedparse':
        identity['config_sha256'] = digest(biomed_config)
        identity['parameter_devices'] = sorted({str(p.device) for p in worker.model.parameters()})
        identity['parameter_count'] = sum(p.numel() for p in worker.model.parameters())
        identity['compatibility_shims'] = identity['compatibility_shims'][:1] + [
            'upstream logs redirected', 'N,H,W predictions split into one H,W mask per prompt']
    @worker.app.get('/health')
    async def health():
        return dict(identity, cuda_allocated_bytes=torch.cuda.memory_allocated(), ready=True)
    logging.info('Loaded real worker: %s', identity)
    worker.run()


if __name__ == '__main__':
    main()
