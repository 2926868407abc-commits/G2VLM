# G2VLM Reconstruction Evaluation

Our reconstruction evaluation code is based on the PI3 evaluation code. Please follow the instructions (link) to prepare the required data.
To run the reconstruction evaluation for G2VLM, the following modifications are required:

1. In `mv_recon/eval.py`,`relpose/eval_angle.py`,`monodepth/infer.py`,modify the path added to sys so that it points to the codebase root directory.
    ```python
    import sys
    sys.path.append("/path/to/G2VLM")
    ```

2. Specify the model checkpoint path under `configs/model/default.yaml`.
3. To avoid import conflicts, save the required data under `datas` instead of `data`.



## 1. Monocular Depth Estimation

```python
python monodepth/infer.py
python monodepth/eval.py
```

## 2. Relative Camera Pose Estimation (Angular Metrics)


```python
python relpose/eval_angle.py
```

## 3. Multi-view Reconstruction (Point Map Estimation)

```python
python mv_recon/eval.py
```