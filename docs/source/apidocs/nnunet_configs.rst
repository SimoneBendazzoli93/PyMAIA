nnDetection config
========================================================


.. jsonschema:: configs/nnUNet_config_template.json
.. code-block:: json

    {
          "Seed": 12345,
          "label_suffix": "-seg.nii.gz",
          "Modalities": {
            "-t1c.nii.gz": "MRI",
            "-t1n.nii.gz": "MRI",
            "-t2f.nii.gz": "MRI",
            "-t2w.nii.gz": "MRI"
          },
          "label_dict": {
            "background": 0,
            "NCR": 1,
            "ED": 2,
            "ET": 3
          },
          "n_folds": 5,
          "Experiment Name": "BraTS_nnUNet_3D_fullres",
          "FileExtension": ".nii.gz",
    }
