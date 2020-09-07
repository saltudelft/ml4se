# Machine Learning for Software Engineering
This repository contains a curated list of papers, datasets, and tools that are devoted to research on Machine Learning for Software Engineering. The papers are organized into popular research areas so that researchers can find recent papers and state-of-the-art approaches easily.

Please feel free to send a pull request to add papers and relevant content that are not listed here.

## Content
- [Papers](#papers)
    - [Type Inference](#type-inference)
    - [Code Completion](#code-completion)
    - [Code Generation](#code-generation)
    - [Code Summarization](#code-summarization)
    - [Code Embeddings](#code-embeddings)
    - [Code Changes](#code-changes)
    - [Bug/Vulnerability Detection](#bug/vulnerability-detection)
    - [Source Code Modeling](#source-code-modeling)
    - [Program Repair](#program-repair)
    - [Program Translation](#program-translation)
    - [Code Duplication](#code-duplication)
    - [Surveys](#surveys)
- [PhD Theses](#phd-theses)
- [Talks](#talks)
- [Datasets](#datasets)
- [Tools](#tools)
- [Research Groups](#research-groups)

# Papers

## Type Inference

- **OptTyper: Probabilistic Type Inference by Optimising Logical and Natural Constraints** (2020), arxiv 2020, Pandi, Irene Vlassi, et al. [[pdf]](https://arxiv.org/pdf/2004.00348)
- **Typilus: Neural Type Hints** (2020), PLDI 2020, Allamanis, Miltiadis, et al. [[pdf]](https://arxiv.org/pdf/2004.10657)
- **LambdaNet: Probabilistic Type Inference using Graph Neural Networks** (2020), arxiv 2020, Wei, Jiayi, et al. [[pdf]](https://arxiv.org/pdf/2005.02161)
- **TypeWriter: Neural Type Prediction with Search-based Validation** (2019), arxiv 2019, Pradel, Michael, et al. [[pdf]](https://arxiv.org/pdf/1912.03768)
- **NL2Type: Inferring JavaScript Function Types from Natural Language Information** (2019), ICSE 2019, Malik, Rabee S., et al. [[pdf]](http://software-lab.org/publications/icse2019_NL2Type.pdf)
- **Deep Learning Type Inference** (2018), ESEC/FSE 2018, Hellendoorn, Vincent J., et al. [[pdf]](http://vhellendoorn.github.io/PDF/fse2018-j2t.pdf)
- **Python Probabilistic Type Inference with Natural Language Support** (2016), FSE 2016, Xu, Zhaogui, et al.
- **Predicting Program Properties from “Big Code”** (2015) ACM SIGPLAN 2015, Raychev, Veselin, et al. [[pdf]](https://files.sri.inf.ethz.ch/website/papers/jsnice15.pdf)

## Code Completion

- **Code Prediction by Feeding Trees to Transformers** (2020), arxiv 2020, Kim, Seohyun, et al. [[pdf]](https://arxiv.org/pdf/2003.13848)
- **Fast and Memory-Efficient Neural Code Completion** (2020), arxiv 2020, Svyatkovskoy, Alexey, et al. [[pdf]](https://arxiv.org/pdf/2004.13651)
- **Pythia: AI-assisted Code Completion System** (2019), KDD '19, Svyatkovskiy, Alexey, et al. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3292500.3330699)
- **Code Completion with Neural Attention and Pointer Networks** (2018), arxiv 2018, Li, Jian, et al. [[pdf]](https://arxiv.org/pdf/1711.09573)

## Code Generation

- **Code Prediction by Feeding Trees to Transformers** (2020), arxiv 2020, Kim, Seohyun, et al. [[pdf]](https://arxiv.org/pdf/2003.13848)
- **TreeGen: A Tree-Based Transformer Architecture for Code Generation** (2019), arxiv 2019, Zhu, Qihao, et al. [[pdf]](https://arxiv.org/abs/1911.09983)
- **A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation** (2017), arxiv 2017, Barone, Antonio V. M., et al. [[pdf]](https://arxiv.org/pdf/1707.02275)

## Code Summarization

- **Source Code Summarization Using Attention-Based Keyword Memory Networks** (2020), IEEE BigComp 2020, Choi, YunSeok, et al.
- **A Transformer-based Approach for Source Code Summarization** (2020), arxiv 2020, Ahmad, Wasi Uddin, et al. [[pdf]](https://arxiv.org/pdf/2005.00653)
- **A Convolutional Attention Network for Extreme Summarization of Source Code** (2016), ICML 2016, Allamanis, Miltiadis, et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/allamanis16.pdf)

## Code Embeddings

- **Contrastive Code Representation Learning** (2020), arxiv 2020, Jain, Paras, et al. [[pdf]](https://arxiv.org/pdf/2007.04973.pdf)
- **Codebert: A Pre-trained Model for Programming and Natural Languages** (2020), arxiv 2020, Feng, Zhangyin, et al. [[pdf]](https://arxiv.org/pdf/2002.08155)
- **SCELMo: Source Code Embeddings from Language Models** (2020), arxiv 2020, Karampatsis, Rafael-Michael, et al. [[pdf]](https://arxiv.org/pdf/2004.13214)
- **code2vec: Learning Distributed Representations of Code** (2019), ACM POPL 2019, Alon, Uri, et al. [[pdf]](http://www.cs.technion.ac.il/~mbs/publications/code2vec-popl19.pdf)
- **Pre-trained Contextual Embedding of Source Code** (2019), arxiv 2019, Kanade, Aditya, et al. [[pdf]](https://arxiv.org/pdf/2001.00059)
- **COSET: A Benchmark for Evaluating Neural Program Embeddings** (2019), arxiv 2019, Wang, Ke, et al. [[pdf]](https://arxiv.org/pdf/1905.11445)
- **A Literature Study of Embeddings on Source Code** (2019), arxiv 2019, Chen, Zimin, et al. [[pdf]](https://arxiv.org/pdf/1904.03061)
- **code2seq: Generating Sequences from Structured Representations of Code** (2018), arxiv 2018, Alon, Uri, et al. [[pdf]](https://arxiv.org/pdf/1808.01400)
- **Neural Code Comprehension: A Learnable Representation of Code Semantics** (2018), NIPS 2018, Ben-Nun, Tal, et al. [[pdf]](http://papers.nips.cc/paper/7617-neural-code-comprehension-a-learnable-representation-of-code-semantics.pdf)

## Code Changes

- **CODIT: Code Editing with Tree-Based Neural Models** (2020), TSE 2020, Chakraborty, Saikat, et al.
- **On learning meaningful code changes via neural machine translation** (2019), ICSE 2019, Tufano, Michele, et al.
- **Commit2Vec: Learning Distributed Representations of Code Changes** (2019), arxiv 2019, Lozoya, Rocío Cabrera, et al. [[pdf]](https://arxiv.org/pdf/1911.07605)

## Bug/Vulnerability Detection

- **Deep Learning based Software Defect Prediction** (2020), Neurocomputing, Qiao, Lei, et al.
- **Software Vulnerability Discovery via Learning Multi-domain Knowledge Bases** (2019), IEEE TDSC, Lin, Guanjun, et al.
- **Neural Bug Finding: A Study of Opportunities and Challenges** (2019), arxiv 2019, Habib, Andrew, et al. [[pdf]](https://arxiv.org/pdf/1906.00307)
- **Automated Vulnerability Detection in Source Code Using Deep Representation Learning** (2018), ICMLA 2018, Russell, Rebecca, et al.
- **DeepBugs: A Learning Approach to Name-based Bug Detection** (2018), ACM PL 2018, Pradel, Michael, et al. [[pdf]](http://software-lab.org/publications/DeepBugs_arXiv_1805.11683.pdf)
- **Automatically Learning Semantic Features for Defect Prediction** (2016), ICSE 2016, Wang, Song, et al.

## Source Code Modeling

- **Maybe Deep Neural Networks are the Best Choice for Modeling Source Code** (2019), arxiv 2019, Karampatsis, Rafael-Michael, et al. [[pdf]](https://arxiv.org/pdf/1903.05734)
- **Are Deep Neural Networks the Best Choice for Modeling Source Code?** (2017), FSE 2017, Hellendoorn, Vincent J., et al. [[pdf]](https://vhellendoorn.github.io/PDF/fse2017.pdf)

## Program Repair

- **Neural Program Repair by Jointly Learning to Localize and Repair** (2019), arxiv 2019, Vasic, Marko, et al. [[pdf]](https://arxiv.org/pdf/1904.01720)

## Program Translation
- **Unsupervised Translation of Programming Languages** (2020), arxiv 2020, Lachaux, Marie-Anne et al. [[pdf]](https://arxiv.org/abs/2006.03511)

## Code Duplication

- **The Adverse Effects of Code Duplication in Machine Learning Models of Code** (2019), Onward! 2019, Allamanis, Miltiadis, et al. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3359591.3359735)

## Surveys
- **Synergy between Machine/Deep Learning and Software Engineering: How Far Are We?** (2020), arxiv 2020, Wang, Simin, et al. [[pdf]](https://arxiv.org/pdf/2008.05515.pdf)
- **Software Engineering Meets Deep Learning: A Literature Review** (2020), arxiv 2020, Ferreira, Fabio, et al. [[pdf]](https://arxiv.org/pdf/1909.11436.pdf)
- **Software Vulnerability Detection Using Deep Neural Networks: A Survey** (2020), Proceedings of the IEEE, Lin, Guanjun, et al.
- **Deep Learning for Source Code Modeling and Generation: Models, Applications and Challenges** (2020), arxiv 2020, Le, Triet HM, et al. [[pdf]](https://arxiv.org/pdf/2002.05442)
- **A Survey of Machine Learning for Big Code and Naturalness** (2018), ACM Computing Surveys, Allamanis, Miltiadis, et al. [[pdf]](https://miltos.allamanis.com/publicationfiles/allamanis2018survey/allamanis2018survey.pdf)

# PhD Theses

- **Learning Code Transformations via Neural Machine Translation** (2019), Michele Tufano [[pdf]](https://scholarworks.wm.edu/cgi/viewcontent.cgi?article=6811&context=etd)
- **Improving the Usability of Static Analysis Tools Using Machine Learning** (2019), Ugur Koc [[pdf]](https://drum.lib.umd.edu/bitstream/handle/1903/25464/Koc_umd_0117E_20465.pdf?sequence=2&isAllowed=y)
- **Learning Natural Coding Conventions** (2016), Miltiadis Allamanis [[pdf]](https://miltos.allamanis.com/publicationfiles/allamanis2017dissertation/allamanis2017dissertation.pdf)

# Talks
- **Machine Learning for Software Engineering: AMA**, MSR 2020 [[video]](https://youtu.be/cphPhsehw2M)
- **Understanding Source Code with Deep Learning**, FOSDEM 2019 [[video]](http://bofh.nikhef.nl/events/FOSDEM/2019/H.2213/ml_on_code_understanding.webm)

# Datasets

- [ManySStuBs4J](https://datashare.is.ed.ac.uk/handle/10283/3424) (2019)
- [150k Python Dataset](https://eth-sri.github.io/py150) (2016)
- [150k Javascript Dataset](https://eth-sri.github.io/js150) (2016)
- [GitHub Java Corpus](http://groups.inf.ed.ac.uk/cup/javaGithub/) (2013)

# Tools

- [SentencePiece](https://github.com/google/sentencepiece) - Unsupervised text tokenizer for Neural Network-based text generation
- [Near-duplicate Source Code Detector](https://github.com/microsoft/near-duplicate-code-detector)
- [Utilities by the DPU team of Microsoft](https://github.com/microsoft/dpu-utils)
- [A set of tools to work with Big Code](https://github.com/danhper/bigcode-tools) - Fetching GitHub repos, tokenizers, embeddings and etc
- [LibCST](https://github.com/Instagram/LibCST) - A concrete syntax tree parser library for Python 
- [cloc](https://github.com/AlDanial/cloc) - Counts blank lines, comment lines, and physical lines of source code in many programming languages.

# Research Groups

- [Software Engineering Research Group (SERG)](https://se.ewi.tudelft.nl/), Delft University of Technology
- [Secure, Reliable, and Intelligent Systems Lab (SRI)](https://www.sri.inf.ethz.ch/), ETH Zurich
- [Software Lab (SOLA)](https://software-lab.org/index.html), University of Stuttgart
- [Machine Learning for the Analysis of Source Code Text (MAST)](http://mast-group.github.io/), Edinburgh University
- [Deep Program Understanding](https://www.microsoft.com/en-us/research/project/program/), Microsoft Research