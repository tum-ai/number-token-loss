<div align="center">
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://badge.fury.io/py/ntloss.svg)](https://badge.fury.io/py/ntloss)
[![Downloads](https://static.pepy.tech/badge/ntloss)](https://pepy.tech/project/ntloss)

# ntloss

*Introducing `ntloss` -- a lightweight python package to calculate a regression-like loss on a token head of a LLM. Use `ntloss` to achieve better performance on math tasks without computational overhead or complex reasoning 🚀*

</div>


## Usage

Simply install NTL into your existing project
```sh
uv add ntloss
pip install ntloss # if you are oldschool
```

Use like this:
```py
from ntloss import NTLoss as NTL
ntl = NTL(tokenizer=tokenizer)
loss = ntl(logits, labels)
```

NOTE: `ntloss` is currently in alpha phase and pre-release. Feedback & PRs are very welcome.


## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zausinger2025regress,
  title   = {Regress, Don't Guess – A Regression-like Loss on Number Tokens for Language Models},
  author  = {Jonas Zausinger and Lars Pennig and Anamarija Kozina and Sean Sdahl
             and Julian Sikora and Adrian Dendorfer and Timofey Kuznetsov
             and Mohamad Hagog and Nina Wiedemann and Kacper Chlodny
             and Vincent Limbach and Anna Ketteler and Thorben Prein
             and Vishwa Mohan Singh and Michael Danziger and Jannis Born},
  booktitle = {Proc. of the 42nd International Conference on Machine Learning (ICML)},
  year    = {2025},
  url     = {https://tum-ai.github.io/number-token-loss/}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## 🙏 Acknowledgments
This work was supported by TUM.ai, Technical University of Munich, and IBM Research Europe.
-->
--- 

<div align="center">

**[🌐 Project Website](https://tum-ai.github.io/number-token-loss/) | [📄 Paper](https://arxiv.org/abs/2411.02083) | [🎮 Demo](https://huggingface.co/spaces/jannisborn/NumberTokenLoss) | [💻 PyPI](https://pypi.org/project/ntloss/) | [💻 Integration Example](scripts/loss_integration.ipynb)**

</div>
