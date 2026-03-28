# AiZynthFinder

[![License](https://img.shields.io/github/license/MolecularAI/aizynthfinder)](https://github.com/MolecularAI/aizynthfinder/blob/master/LICENSE)
[![Tests](https://github.com/MolecularAI/aizynthfinder/workflows/tests/badge.svg)](https://github.com/MolecularAI/aizynthfinder/actions?workflow=tests)
[![codecov](https://codecov.io/gh/MolecularAI/aizynthfinder/branch/master/graph/badge.svg)](https://codecov.io/gh/MolecularAI/aizynthfinder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![version](https://img.shields.io/github/v/release/MolecularAI/aizynthfinder)](https://github.com/MolecularAI/aizynthfinder/releases)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MolecularAI/aizynthfinder/blob/master/contrib/notebook.ipynb)

AiZynthFinder 是一个用于逆合成规划的工具。默认算法基于蒙特卡洛树搜索，会递归地将目标分子拆解为可采购的前体。树搜索过程由策略网络引导，该网络利用在已知反应模板库上训练得到的神经网络来推荐可能的前体。整个方案具有很强的可定制性，因为工具支持多种搜索算法和扩展策略。

介绍视频见：[https://youtu.be/r9Dsxm-mcgA](https://youtu.be/r9Dsxm-mcgA)

## 前置条件

开始之前，请确认满足以下要求：

* 支持 Linux、Windows 和 macOS，只要这些平台能够满足项目依赖即可。

* 已安装 [anaconda](https://www.anaconda.com/) 或 [miniconda](https://docs.conda.io/en/latest/miniconda.html)，并使用 Python 3.10 - 3.12。

该工具最初在 Linux 平台上开发，但已经在 Windows 10 和 macOS Catalina 上完成测试。

## 安装

### 面向终端用户

首次使用时，请在终端或 Anaconda Prompt 中执行以下命令：

    conda create "python>=3.10,<3.13" -n aizynth-env

安装时，先激活环境，再通过 PyPI 安装软件包：

    conda activate aizynth-env
    python -m pip install aizynthfinder[all]

如果你只想安装一个更轻量的版本，不包含全部功能，也可以执行：

    python -m pip install aizynthfinder

### 面向开发者

首先使用 Git 克隆仓库。

然后在仓库根目录执行以下命令：

    conda env create -f env-dev.yml
    conda activate aizynth-dev
    poetry install --all-extras

此时，`aizynthfinder` 包会以可编辑模式安装。


## 使用方式

安装完成后，会同时提供 `aizynthcli` 和 `aizynthapp` 两个工具，作为算法的命令行与图形界面入口：

    aizynthcli --config config_local.yml --smiles smiles.txt
    aizynthapp --config config_local.yml


更多信息请查阅文档：[here](https://molecularai.github.io/aizynthfinder/)

要使用该工具，你需要准备：

    1. 库存文件
    2. 训练好的扩展策略网络
    3. 训练好的过滤策略网络（可选）

这些文件可以从 [figshare](https://figshare.com/articles/AiZynthFinder_a_fast_robust_and_flexible_open-source_software_for_retrosynthetic_planning/12334577) 和 [这里](https://figshare.com/articles/dataset/A_quick_policy_to_filter_reactions_based_on_feasibility_in_AI-guided_retrosynthetic_planning/13280507) 下载，也可以通过以下命令自动获取：

```
download_public_data my_folder
```

其中，`my_folder` 是你希望下载到的目录。
命令执行后会生成一个 `config.yml` 文件，你可以将它用于 `aizynthcli` 或 `aizynthapp`。

## 开发

### 测试

测试使用 `pytest`，并会通过 `poetry` 一并安装。

运行测试：

    pytest -v

CI 服务器使用的完整测试命令也提供了对应的 `invoke` 入口：

    invoke full-tests

### 文档生成

项目文档通过 Sphinx 生成，内容来自手写教程和代码中的文档字符串。

生成 HTML 文档：

    invoke build-docs

## 参与贡献

欢迎通过 issue 或 pull request 的形式参与贡献。

如果你有问题或想报告 bug，请提交 issue。


如果你想为项目提交代码，请按以下步骤操作：

1. Fork 本仓库。
2. 创建分支：`git checkout -b <branch_name>`。
3. 完成修改并提交：`git commit -m '<commit_message>'`
4. 推送到远端分支：`git push`
5. 创建 pull request。

请使用 `black` 进行代码格式化，并遵循 `pep8` 风格指南。


## 贡献者

* [@SGenheden](https://www.github.com/SGenheden)
* [@lakshidaa](https://github.com/Lakshidaa)
* [@helenlai](https://github.com/helenlai)
* [@EBjerrum](https://www.github.com/EBjerrum)
* [@A-Thakkar](https://www.github.com/A-Thakkar)
* [@benteb](https://www.github.com/benteb)

贡献者可用于支持问题的时间有限，但仍然欢迎你提交 issue（见上文）。

## 许可证

本软件基于 MIT 许可证发布（见 `LICENSE` 文件），可免费使用，并按“现状”提供。

## 参考文献

1. Thakkar A, Kogej T, Reymond J-L, et al (2019) Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain. Chem Sci. https://doi.org/10.1039/C9SC04944D
2. Genheden S, Thakkar A, Chadimova V, et al (2020) AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.12465371.v1
