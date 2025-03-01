# DS-Font

PyTorch Code for "Few-shot Font Generation by Learning Style Difference and Similarity" (Accepted by TCSVT)

## Requirements

* Python 3
* torch>=0.4.1
* torchvision>=0.2.1
* dominate>=2.3.1
* visdom>=0.1.8.3

## Getting Started


- Download the [dataset](https://drive.google.com/file/d/1XJppxR00pyk5xG-64Ia_BF12XSxeZgfa/view?usp=sharing  "https://drive.google.com/file/d/1XJppxR00pyk5xG-64Ia_BF12XSxeZgfa/view?usp=sharing").
- Unzip it to ./datasets/
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the model
```bash
bash ./train.sh
```
- Test
```bash
bash ./test.sh
```


## Acknowledgements

* This code builds heavily on **[Few-shot Font Style Transfer between Different Languages](https://github.com/ligoudaner377/font_translator_gan)**. Thanks for open-sourcing!