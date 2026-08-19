# MCI-Diff

Official implementation of **Diffusion with a Linguistic Compass: Steering the
Generation of Clinically Plausible Future sMRI Representations for Early MCI
Conversion Prediction** (CVPR 2026). Work in progress. This repository contains the current implementation of MCI-Diff. We are still checking the training pipeline, experiment settings, and documentation, so some parts may be updated before the final release.

[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Diffusion_with_a_Linguistic_Compass_Steering_the_Generation_of_Clinically_CVPR_2026_paper.html) |
[arXiv](https://arxiv.org/abs/2506.05428)

MCI-Diff generates longitudinal sMRI representations at 6, 12, 18, 24 and 36
months from a baseline representation. The generated sequence is then used for
pMCI/sMCI classification.

```text
sMRI -> feature extractor -> progressive diffusion -> future candidates
     -> candidate selection -> longitudinal classifier
```

## Environment

```bash
conda env create -f environment.yml
conda activate mci-diff
python -m pip install -e .
```

The image preprocessing and feature extraction follow FSL, ANTs and HFCN. Use
their public implementations or another compatible sMRI encoder. FreeSurfer and
the language-model scorer are not included here; use an available implementation
when running the linguistic-guidance part.

## Data

The code uses one row per scan:

```csv
subject_id,dataset,month,image_path,label,feature_path,sex,age,mmse
ADNI_0001,ADNI1,0,data/mni/ADNI_0001_m00.nii.gz,pMCI,data/features/ADNI_0001_m00.npy,M,72.4,27
ADNI_0001,ADNI1,6,data/mni/ADNI_0001_m06.nii.gz,pMCI,data/features/ADNI_0001_m06.npy,M,72.9,26
```

`month` is one of `0, 6, 12, 18, 24, 36`. Missing visits are represented by
missing rows. Each subject needs the baseline visit. Feature files can be `.npy`
or `.npz`, with one vector of length `data.feature_dim` for each scan.

An example is provided in `examples/manifest.example.csv`.

If the sMRI features have already been extracted, only `feature_path` is used.
A wrapped encoder can also be called from the command line:

```bash
mci-diff extract \
  --config configs/default.yaml \
  --encoder /path/to/hfcn_wrapper.ts
```

A Python encoder can be passed as `--factory module.path:function`.

## Training

Prepare the data split:

```bash
mci-diff validate --config configs/default.yaml
mci-diff split --config configs/default.yaml
```

Run the stages separately:

```bash
mci-diff train-diffusion  --config configs/default.yaml --fold 0
mci-diff generate         --config configs/default.yaml --fold 0
mci-diff train-classifier --config configs/default.yaml --fold 0
mci-diff evaluate         --config configs/default.yaml --fold 0
```

Or run one fold end to end:

```bash
mci-diff run --config configs/default.yaml --folds 0
```

For the combined ADNI five-fold experiment:

```bash
mci-diff run --config configs/default.yaml --folds 0 1 2 3 4 \
  --set split.protocol=adni_5fold \
  --set split.file=data/splits/adni_5fold.json \
  --set experiment.output_dir=outputs/adni_5fold
```

The default protocol trains on ADNI-1, evaluates on ADNI-2 and uses AIBL as the
external set. pMCI is the positive class.

## Candidate scoring

The sampler accepts an external scorer through `sampling.scorer_factory`. The
value is written as `module.path:function`; the factory returns an object with a
`score_candidates` method. It should return one scalar score for every generated
candidate. If no scorer is supplied, candidate 0 is used.

## Main settings

```yaml
diffusion:
  num_steps: 40
  denoiser_layers: 6

progressive:
  max_difficulty: 4

sampling:
  num_candidates: 20
```

The visit grid is `[0, 6, 12, 18, 24, 36]`. Training starts from complete
sequences and alternates interpolation and extrapolation. At each difficulty,
the newly completed sequences are added back to the training pool.

`configs/paper_experiments.yaml` contains the interpolation/extrapolation
ablations and the sensitivity grids:

```bash
mci-diff matrix --experiments configs/paper_experiments.yaml
```

Add `--execute` to run the generated experiment list.

## Repository layout

```text
configs/                    experiment settings
examples/                   manifest example
src/mci_diff/data.py        manifest, sequences and data splits
src/mci_diff/models.py      diffusion model and classifier
src/mci_diff/adapters.py    wrappers for external sMRI encoders
src/mci_diff/engine.py      training, sampling and evaluation
src/mci_diff/cli.py         command line entry point
tests/                      formula and pipeline checks
```

## Citation

```bibtex
@inproceedings{tang2026linguistic,
  title     = {Diffusion with a Linguistic Compass: Steering the Generation of Clinically Plausible Future sMRI Representations for Early MCI Conversion Prediction},
  author    = {Tang, Zhihao and Li, Chaozhuo and Zhang, Litian and Zhang, Xi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {42691--42700},
  year      = {2026}
}
```

## License

MIT. External programs, datasets and model weights follow their own licenses.
