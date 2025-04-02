# HallOumi

## 🛠 Setup
```bash
git clone https://github.com/oumi-ai/oumi.git
cd oumi/configs/projects/halloumi

pip install oumi
```

## ⚙️ Training
Example of Oumi fine-tuning:
```bash
# train HallOumi-8B
oumi train -c 8b_train.yaml

# train HallOumi-8B with GCP
oumi launch up -c configs/projects/halloumi/gcp_job.yaml --cluster halloumi-8b-sft
```

## 🚀 Inference
For more information on running inference, please see our demo on GitHub:

https://github.com/oumi-ai/halloumi-demo

To try out without installation, see our web demo:

https://oumi.ai/halloumi-demo

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
