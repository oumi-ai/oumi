# HallOumi

## 🛠 Setup

```bash
pip install oumi
```

## ⚙️ Training

Example of Oumi fine-tuning:

```bash
# Train HallOumi-8B locally
oumi train -c oumi://configs/projects/halloumi/8b_train.yaml

# Launch a job to train HallOumi-8B on GCP
# Setup instructions: https://oumi.ai/docs/en/latest/user_guides/launch/launch.html
oumi launch up -c oumi://configs/projects/halloumi/gcp_job.yaml --cluster halloumi-8b-sft
```

## 🚀 Inference

Try out our web demo here:

https://oumi.ai/halloumi-demo

To run inference yourself, please see our demo on GitHub:

https://github.com/oumi-ai/halloumi-demo

## ❗️ License

This model is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## 📖 Citation

If you use **HallOumi** in your research, please cite:

```
@misc{oumi2025HallOumi,
      title={HallOumi - a state-of-the-art claim verification model},
      author={Jeremiah Greer and Panos Achlioptas and Konstantinos Aisopos and Michael Schuler and Matthew Persons and Oussama Elachqar and Emmanouil Koukoumidis},
      year={2025},
      url={https://oumi.ai/halloumi},
}
```
