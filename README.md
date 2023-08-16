# Machine Learning for Software Engineering
![GitHub last commit](https://img.shields.io/github/last-commit/saltudelft/ml4se)
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mir-am/0d2ee0d88d518e1a916aef1f8c77a666/raw/ml4se-paper-no.json)

This repository contains a curated list of papers, PhD theses, datasets, and tools that are devoted to research on Machine Learning for Software Engineering. The papers are organized into popular research areas so that researchers can find recent papers and state-of-the-art approaches easily.

Please feel free to send a pull request to add papers and relevant content that are not listed here.

> Note: to quickly access this page, use [ml4se.dev](https://ml4se.dev/)

## Content
- [Papers](#papers)
    - [Type Inference](#type-inference)
    - [Code Completion](#code-completion)
    - [Code Generation](#code-generation)
    - [Code Summarization](#code-summarization)
    - [Code Embeddings/Representation](#code-embeddingsrepresentation)
    - [Code Changes/Editing](#code-changesediting)
    - [Code Comments](#code-comments)
    - [Bug/Vulnerability Detection](#bugvulnerability-detection)
    - [Source Code Modeling](#source-code-modeling)
    - [Program Repair](#program-repair)
    - [Program Translation](#program-translation)
    - [Program Analysis](#program-analysis)
    - [Software Testing](#software-testing)
    - [Code Clone Detection](#code-clone-detection)
    - [Code Language Models](#code-language-models)
    - [Code Review](#code-review)
    - [Empirical Studies](#empirical-studies)
    - [Surveys](#surveys)
    - [Misc](#misc)
- [PhD Theses](#phd-theses)
- [Talks](#talks)
- [Datasets](#datasets)
- [Tools](#tools)
- [Research Groups](#research-groups)
- [Venues](#venues)

# Papers

## Type Inference

- **Domain Knowledge Matters: Improving Prompts with Fix Templates for Repairing Python Type Errors** (2023), ICSE'24, Peng, Yun, et al. [[pdf]](https://arxiv.org/pdf/2306.01394)
- **Generative Type Inference for Python** (2023), ASE'23, Peng, Yun, et al. [[pdf]](https://arxiv.org/pdf/2307.09163)
- **Type Prediction With Program Decomposition and Fill-in-the-Type Training** (2023), arxiv, Cassano, Federico, et al. [[pdf]](https://arxiv.org/pdf/2305.17145)
- **TypeT5: Seq2seq Type Inference using Static Analysis** (2023), ICLR'23, Wei, Jiayi, et al. [[pdf]](https://arxiv.org/pdf/2303.09564)
- **Do Machine Learning Models Produce TypeScript Types that Type Check?** (2023), arxiv, Yee, M., and Arjun G. [[pdf]](https://arxiv.org/pdf/2302.12163)
- **Cross-Domain Evaluation of a Deep Learning-Based Type Inference System** (2022), arxiv, Gruner, Bernd, et al. [[pdf]](https://arxiv.org/pdf/2208.09189) [[code]](https://gitlab.com/dlr-dw/type-inference)
- **Learning To Predict User-Defined Types** (2022), TSE'22, Jesse, Keven, et al. [[pdf]](https://www.cs.ucdavis.edu/~devanbu/DiverseTyper_TSE.pdf)
- **Recovering Container Class Types in C++ Binaries** (2022), CGO'22, Wang, Xudong, et al.
- **Finding the Dwarf: Recovering Precise Types from WebAssembly Binaries** (2022), PLDI'22, Lehmann, Daniel and Pradel, Michael [[pdf]](https://dlehmann.eu/publications/WasmTypePrediction-pldi2022.pdf)
- **Type4Py: Practical Deep Similarity Learning-Based Type Inference for Python** (2022), ICSE'22, Mir, Amir, et al. [[pdf]](https://arxiv.org/pdf/2101.04470.pdf)[[code]](https://github.com/saltudelft/type4py)
- **Static Inference Meets Deep Learning: A Hybrid Type Inference Approach for Python** (2022), ICSE'22, Peng, Yun, et al. [[pdf]](https://arxiv.org/pdf/2105.03595)
- **Type Inference as Optimization** (2021), NeurIPS'21 AIPLANS,  Pandi, Irene Vlassi, et al. [[pdf]](https://openreview.net/pdf?id=yHYZaQ0Zvml)
- **SimTyper: Sound Type Inference for Ruby using Type Equality Prediction** (2021), OOPSLA'21, Kazerounian, Milod, et al.
- **Learning type annotation: is big data enough?** (2021), FSE 2021, Jesse, Kevin, et al. [[pdf]](https://www.cs.ucdavis.edu/~devanbu/typebert_esec_fse_.pdf)[[code]](https://github.com/typebert/typebert)
- **Cross-Lingual Adaptation for Type Inference** (2021), arxiv 2021, Li, Zhiming, et al. [[pdf]](https://arxiv.org/pdf/2107.00157)
- **PYInfer: Deep Learning Semantic Type Inference for Python Variables** (2021), arxiv 2021, Cui, Siwei, et al. [[pdf]](https://arxiv.org/pdf/2106.14316)
- **Advanced Graph-Based Deep Learning for Probabilistic Type Inference** (2020), arxiv 2020, Ye, Fangke, et al. [[pdf]](https://arxiv.org/pdf/2009.05949.pdf)
- **Typilus: Neural Type Hints** (2020), PLDI 2020, Allamanis, Miltiadis, et al. [[pdf]](https://arxiv.org/pdf/2004.10657)[[code]](https://github.com/typilus/typilus)
- **LambdaNet: Probabilistic Type Inference using Graph Neural Networks** (2020), arxiv 2020, Wei, Jiayi, et al. [[pdf]](https://arxiv.org/pdf/2005.02161)
- **TypeWriter: Neural Type Prediction with Search-based Validation** (2019), arxiv 2019, Pradel, Michael, et al. [[pdf]](https://arxiv.org/pdf/1912.03768)
- **NL2Type: Inferring JavaScript Function Types from Natural Language Information** (2019), ICSE 2019, Malik, Rabee S., et al. [[pdf]](http://software-lab.org/publications/icse2019_NL2Type.pdf)[[code]](https://github.com/sola-da/NL2Type)
- **Deep Learning Type Inference** (2018), ESEC/FSE 2018, Hellendoorn, Vincent J., et al. [[pdf]](http://vhellendoorn.github.io/PDF/fse2018-j2t.pdf)[[code]](https://github.com/DeepTyper/DeepTyper)
- **Python Probabilistic Type Inference with Natural Language Support** (2016), FSE 2016, Xu, Zhaogui, et al.
- **Predicting Program Properties from “Big Code”** (2015) ACM SIGPLAN 2015, Raychev, Veselin, et al. [[pdf]](https://files.sri.inf.ethz.ch/website/papers/jsnice15.pdf)

## Code Completion

- **RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems** (2023), arxiv, Liu, T., et al. [[pdf]](https://arxiv.org/pdf/2306.03091)
- **A Static Evaluation of Code Completion by Large Language Models** (2023), arxiv, Ding, Hantian, et al. [[pdf]](https://arxiv.org/pdf/2306.03203)
- **Large Language Models of Code Fail at Completing Code with Potential Bugs** (2023), arxiv, Dinh, Tuan, et al. [[pdf]](https://arxiv.org/pdf/2306.03438)
- **RepoFusion: Training Code Models to Understand Your Repository** (2023), arxiv, Shrivastava, Disha, et al., [[pdf]](https://arxiv.org/pdf/2306.10998)
- **LongCoder: A Long-Range Pre-trained Language Model for Code Completion** (2023), ICML'23, Guo, Daya, et al. [[pdf]](https://arxiv.org/pdf/2306.14893)
- **R-U-SURE? Uncertainty-Aware Code Suggestions By Maximizing Utility Across Random User Intents** (2023), arxiv, Johnson, Daniel D, et al. [[pdf]](https://arxiv.org/pdf/2303.00732) 
- **Optimized Tokenization Process for Open-Vocabulary Code Completion: An Empirical Study** (2023), EASE'23, Hussain, Yasir, et al.
- **Enriching Source Code with Contextual Data for Code Completion Models: An Empirical Study** (2023), MSR'23, van Dam, Tim, et al. [[pdf]](https://arxiv.org/pdf/2304.12269)
- **RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation** (2023), arxiv, Zhang, Fengji, et al. [[pdf]](https://arxiv.org/pdf/2303.12570)
- **COCOMIC: ✿✿✿✿ Code ✿✿✿✿ Completion By Jointly Modeling In-file and ✿✿Cross-file Context** (2022), Ding, Yangruibo, et al. [[pdf]](https://arxiv.org/pdf/2212.10007)
- **Boosting source code suggestion with self-supervised Transformer Gated Highway** (2022), JSS, Hussain, Yasir, et al.
- **Syntax-Aware On-the-Fly Code Completion** (2022), arxiv, Takerngsaksiri, W., et al. [[pdf]](https://arxiv.org/pdf/2211.04673)
- **Learning to Prevent Profitless Neural Code Completion** (2022), arxiv, Sun, Z., et al. [[pdf]](https://arxiv.org/pdf/2209.05948)
- **All You Need Is Logs: Improving Code Completion by Learning from Anonymous IDE Usage Logs** (2022), arxiv, Bibaev, Vitaliy, et al. [[pdf]](https://arxiv.org/pdf/2205.10692.pdf)
- **CodeFill: Multi-token Code Completion by Jointly Learning from Structure and Naming Sequences** (2022), ICSE'22, Izadi, Maliheh, et al. [[pdf]](https://arxiv.org/pdf/2202.06689.pdf) [[code]](https://github.com/saltudelft/codefill)
- **Code Completion by Modeling Flattened Abstract Syntax Trees as Graphs** (2021), AAAI'21, Wang, Yanlin, et al. [[pdf]](https://www.aaai.org/AAAI21Papers/AAAI-1654.WangY.pdf)
- **Code Prediction by Feeding Trees to Transformers** (2021), ICSE'21, Kim, Seohyun, et al. [[pdf]](https://arxiv.org/pdf/2003.13848)
- **Fast and Memory-Efficient Neural Code Completion** (2020), arxiv 2020, Svyatkovskoy, Alexey, et al. [[pdf]](https://arxiv.org/pdf/2004.13651)
- **Pythia: AI-assisted Code Completion System** (2019), KDD'19, Svyatkovskiy, Alexey, et al. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3292500.3330699)
- **Code Completion with Neural Attention and Pointer Networks** (2018), arxiv 2018, Li, Jian, et al. [[pdf]](https://arxiv.org/pdf/1711.09573)

## Code Generation

- **Is Model Attention Aligned with Human Attention?: An Empirical Study on LLMs for Code Generation** (2023), arxiv, Kou, Bonan, et al. [[pdf]](https://arxiv.org/pdf/2306.01220)
- **Demystifying GPT Self-Repair for Code Generation** (2023), arxiv, Olausson, Theo X., et al. [[pdf]](https://arxiv.org/pdf/2306.09896)
- **Exploring Continual Learning for Code Generation Models** (2023), arxiv, Yadav, Prateek, et al. [[pdf]](https://arxiv.org/pdf/2307.02435)
- **CodePrompt: Task-Agnostic Prefix Tuning for Program and Language Generation** (2023), ACL'23, Choi, Y., & Lee, J. H. [[pdf]](https://aclanthology.org/2023.findings-acl.325.pdf)
- **Aligning Offline Metrics and Human Judgments of Value for Code Generation Models** (2023), ACL'23, Dibia, Victor, et al. [[pdf]](https://aclanthology.org/2023.findings-acl.540.pdf)
- **RLTF: Reinforcement Learning from Unit Test Feedback** (2023), arxiv, Liu, Jiate, et al. [[pdf]](https://arxiv.org/pdf/2307.04349)
- **A Lightweight Framework for High-Quality Code Generation** (2023), arxiv, Siddiq, M. L., et al. [[pdf]](https://arxiv.org/pdf/2307.08220)
- **Large Language Models for Code: Security Hardening and Adversarial Testing** (2023), ICML'23 workshop, He, J., & Vechev, M. [[pdf]](https://openreview.net/pdf?id=Km1XyJJVpS)
- **Reinforcement Learning for Syntax-Guided Synthesis** (2023), arxiv, Parsert, J., and E. Polgreen [[pdf]](https://arxiv.org/pdf/2307.09564)
- **Refining ChatGPT-Generated Code: Characterizing and Mitigating Code Quality Issues**, arxiv, Liu, Yue, et al. [[pdf]](https://arxiv.org/pdf/2307.12596)
- **ExeDec: Execution Decomposition for Compositional Generalization in Neural Program Synthesis**, arxiv, Shi, Kensen, et al. [[pdf]](https://arxiv.org/pdf/2307.13883)
- **Private-Library-Oriented Code Generation with Large Language Models** (2023), arxiv, Zan, Daoguang, et al. [[pdf]](https://arxiv.org/pdf/2307.15370)
- **LLM is Like a Box of Chocolates: the Non-determinism of ChatGPT in Code Generation** (2023), arxiv, Ouyang, Shuyin, et al. [[pdf]](https://arxiv.org/pdf/2303.00732)
- **No Need to Lift a Finger Anymore? Assessing the Quality of Code Generation by ChatGPT** (2023), arxiv, Liu, Zhijie, et al. [[pdf]](https://arxiv.org/pdf/2308.04838)
- **Think Outside the Code: Brainstorming Boosts Large Language Models in Code Generation** (2023), arxiv, Li, Xin-Ye, et al. [[pdf]](https://arxiv.org/pdf/2305.10679) 
- **Neural Machine Translation for Code Generation** (2023), arxiv, KC, Dharma, and Clayton T. M. [[pdf]](https://arxiv.org/pdf/2305.13504)
- **CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X** (2023), arxiv, Zheng, Qinkai, et al. [[pdf]](https://arxiv.org/pdf/2303.17568)
- **Towards Enhancing In-Context Learning for Code Generation** (2023), arxiv, Li, Jia, et al. [[pdf]](https://arxiv.org/pdf/2303.17780)
- **Knowledge Transfer for Pseudo-code Generation from Low Resource Programming Language** (2023), arxiv, Sontakke, Ankita, et al. [[pdf]](https://arxiv.org/pdf/2303.09062)
- **MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation** (2023), TSE, Paul, Rishov, et al.  
- **Self-collaboration Code Generation via ChatGPT** (2023), arxiv, Dong, Yihong, et al. [[pdf]](https://arxiv.org/pdf/2304.07590)
- **Greener yet Powerful: Taming Large Code Generation Models with Quantization** (2023), arxiv, Wei, Xiaokai, et al. [[pdf]](https://arxiv.org/pdf/2303.05378.pdf)
- **A Syntax-Guided Multi-Task Learning Approach for Turducken-Style Code Generation** (2023), arxiv, Yang, Guang, et al. [[pdf]](https://arxiv.org/pdf/2303.05061)
- **WikiCoder: Learning to Write Knowledge-Powered Code** (2023), arxiv, Matricon, Théo, et al. [[pdf]](https://arxiv.org/pdf/2303.08574)
- **Self-planning Code Generation with Large Language Model** (2023), arxiv, Jiang, Xue, et al. [[pdf]](https://arxiv.org/pdf/2303.06689)
- **Systematically Finding Security Vulnerabilities in Black-Box Code Generation Models.** (2023), arxiv, Hajipour, Hossein, et al. [[pdf]](https://arxiv.org/pdf/2302.04012)
- **Exploring Data Augmentation for Code Generation Tasks** (2023), arxiv, C., Pinzhen, and G. Lampouras [[pdf]](https://arxiv.org/pdf/2302.03499)
- **Controlling Large Language Models to Generate Secure and Vulnerable Code** (2023), arxiv, He, J., and M. Vechev [[pdf]](https://arxiv.org/pdf/2302.05319)
- **SKCODER: A Sketch-based Approach for Automatic Code Generation** (2023), arxiv, Li, Jia, et al. [[pdf]](https://arxiv.org/pdf/2302.06144)
- **LEVER: Learning to Verify Language-to-Code Generation with Execution** (2023), arxiv, Ni, Ansong, et al. [[pdf]](https://arxiv.org/pdf/2302.08468)
- **CodeScore: Evaluating Code Generation by Learning Code Execution** (2023), arxiv, Dong, Yihong, et al. [[pdf]](https://arxiv.org/pdf/2301.09043)
- **Program Generation from Diverse Video Demonstrations** (2023), arxiv, Manchin, Anthony, et al. [[pdf]](https://arxiv.org/pdf/2302.00178) 
- **Execution-based Code Generation using Deep Reinforcement Learning** (2023), arxiv, Shojaee, Parshin, et al. [[pdf]](https://arxiv.org/pdf/2301.13816)
- **SantaCoder: don't reach for the stars!** (2023), arxiv, Allal, Loubna Ben, et al. [[pdf]](https://arxiv.org/pdf/2301.03988.pdf)
- **Exploring and Evaluating Personalized Models for Code Generation**, FSE'22, Zlotchevski, Andrei, et al.
- **Natural Language to Code Generation in Interactive Data Science Notebooks** (2022), arxiv, Yin, Pengcheng, et al. [[pdf]](https://arxiv.org/pdf/2212.09248)
- **Asking Clarification Questions for Code Generation in General-Purpose Programming Language** (2022), arxiv, Li, Haau-Sing, et al. [[pdf]](https://arxiv.org/pdf/2212.09885)
- **ExploitGen: Template-augmented exploit code generation based on CodeBERT** (2022), JSS journal, Yang, Guang, et al.
- **Explicit Knowledge Transfer for Weakly-Supervised Code Generation** (2022), arxiv, Azerbayev, Zhangir, et al. [[pdf]](https://arxiv.org/pdf/2211.16740)
- **Program Generation from Diverse Video Demonstrations** (2022), Manchin123, Anthony, et al. [[pdf]](https://bmvc2022.mpi-inf.mpg.de/1039.pdf)
- **Coder Reviewer Reranking for Code Generation** (2022), arxiv, Zhang, Tianyi, et al. [[pdf]](https://arxiv.org/pdf/2211.16490)
- **Execution-based Evaluation for Data Science Code Generation Models** (2022), arxiv, Huang, Junjie, et al. [[pdf]](https://arxiv.org/pdf/2211.09374)
- **Multi-lingual Evaluation of Code Generation Models** (2022), arxiv, Athiwaratkun, Ben, et al. [[pdf]](https://arxiv.org/pdf/2210.14868)[[code]](https://github.com/amazon-science/mbxp-exec-eval)
- **DocCoder: Generating Code by Retrieving and Reading Docs** (2022), arxiv, Zhou, Shuyan, et al. [[pdf]](https://arxiv.org/pdf/2207.05987)
- **Compilable Neural Code Generation with Compiler Feedback** (2022), ACL'22, Wang, Xin, et al. [[pdf]](https://aclanthology.org/2022.findings-acl.2.pdf)
- **T5QL: Taming language models for SQL generation** (2022), arxiv, Arcadinho, S., et al. [[pdf]](https://arxiv.org/pdf/2209.10254)
- **Incorporating Domain Knowledge through Task Augmentation for Front-End JavaScript Code Generation** (2022), arxiv, Shen, Sijie, et al. [[pdf]](https://arxiv.org/pdf/2208.10091)
- **Language Models Can Teach Themselves to Program Better** (2022), arxiv, Haluptzok, Patrick, et al. [[pdf]](https://arxiv.org/pdf/2207.14502)
- **DocCoder: Generating Code by Retrieving and Reading Docs** (2022), arxiv, Zhou, Shuyan, et al. [[pdf]](https://arxiv.org/pdf/2207.05987)
- **CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning** (2022), arxiv, Le, Hung, et al. [[pdf]](https://arxiv.org/pdf/2207.01780)
- **Repository-Level Prompt Generation for Large Language Models of Code** (2022), arxiv, Shrivastava, Disha, et al. [[pdf]](https://arxiv.org/pdf/2206.12839)
- **CERT: Continual Pre-Training on Sketches for Library-Oriented Code Generation** (2022), arxiv, Zan, Daoguang, et al. [[pdf]](https://arxiv.org/pdf/2206.06888)
- **NatGen: Generative pre-training by “Naturalizing” source code** (2022), FSE'22, Chakraborty, Saikat, et al. [[pdf]](https://arxiv.org/pdf/2206.07585)
- **StructCoder: Structure-Aware Transformer for Code Generation** (2022), arxiv, Tipirneni, Sindhu, et al. [[pdf]](https://arxiv.org/pdf/2206.05239)
- **Compilable Neural Code Generation with Compiler Feedback** (2022), arxiv 2022, Wang, Xin, et al. [[pdf]](https://arxiv.org/pdf/2203.05132.pdf)
- **Predictive Synthesis of API-Centric Code** (2022), arxiv 2022, Nam, Daye, et al. [[pdf]](https://arxiv.org/pdf/2201.03758.pdf)
- **Code Prediction by Feeding Trees to Transformers** (2020), arxiv 2020, Kim, Seohyun, et al. [[pdf]](https://arxiv.org/pdf/2003.13848)
- **TreeGen: A Tree-Based Transformer Architecture for Code Generation** (2019), arxiv 2019, Zhu, Qihao, et al. [[pdf]](https://arxiv.org/abs/1911.09983)
- **A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation** (2017), arxiv 2017, Barone, Antonio V. M., et al. [[pdf]](https://arxiv.org/pdf/1707.02275)

## Code Summarization

- **An data augmentation method for source code summarization** (2023), Journal of Neurocomputing, Song, Zixuan, et al.
- **Multilingual Adapter-based Knowledge Aggregation on Code Summarization for Low-Resource Languages** (2023), arxiv, Saberi, Iman et al. [[pdf]](https://arxiv.org/pdf/2307.07854)
- **Statement-based Memory for Neural Source Code Summarization** (2023), arxiv, Bansal, Aakash, et al. [[pdf]](https://arxiv.org/pdf/2307.11709)
- **Tram: A Token-level Retrieval-augmented Mechanism for Source Code Summarization** (2023), arxiv, Ye, Tong, et al. [[pdf]](https://arxiv.org/pdf/2305.11074)
- **Automatic Code Summarization via ChatGPT: How Far Are We?** (2023), arxiv, Sun, Weisong, et al. 
- **Function Call Graph Context Encoding for Neural Source Code Summarization** (2023), TSE, Bansal, Aakash, et al.
- **Label Smoothing Improves Neural Source Code Summarization** (2023), arxiv, Haque, Sakib, et al. [[pdf]](https://arxiv.org/pdf/2303.16178)
- **Demystifying What Code Summarization Models Learned** (2023), arxiv, Wang, Yu, and Ke Wang. [[pdf]](https://arxiv.org/pdf/2303.02333)
- **CoSS: Leveraging Statement Semantics for Code Summarization** (2023), TSE, Shi, Chaochen, et al.
- **An Extensive Study of the Structure Features in Transformer-based Code Semantic Summarization** (2023), RG, Yang, Kang, et al. [[pdf]](https://www.researchgate.net/profile/Shangwen-Wang/publication/369273643_An_Extensive_Study_of_the_Structure_Features_in_Transformer-based_Code_Semantic_Summarization/links/641313ce92cfd54f84060e2f/An-Extensive-Study-of-the-Structure-Features-in-Transformer-based-Code-Semantic-Summarization.pdf)
- **Interpretation-based Code Summarization** (2023), arxiv, Geng, Mingyang, et al. [[pdf]](https://www.researchgate.net/profile/Shangwen-Wang/publication/368755660_Interpretation-based_Code_Summarization/links/63f842890d98a97717b27fb8/Interpretation-based-Code-Summarization.pdf)
- **Towards Retrieval-Based Neural Code Summarization: A Meta-Learning Approach** (2023), TSE, Zhou, Ziyi, et al.
- **CLG-Trans: Contrastive Learning for Code Summarization via Graph Attention-based Transformer** (2023), SCP journal, Zeng, Jianwei, et al.
- **ClassSum: a deep learning model for class-level code summarization** (2022), Springer NCA, Li, Mingchen, et al. [[code]](https://github.com/classsum/ClassSum)
- **Boosting Code Summarization by Embedding Code Structures** (2022), COLING'22, Son, Jikyoeng, et al. [[pdf]](https://aclanthology.org/2022.coling-1.521.pdf)
- **Low-Resources Project-Specific Code Summarization** (2022), ASE'22, Xie, Rui, et al. [[pdf]](https://arxiv.org/pdf/2210.11843)
- **Few-shot training LLMs for project-specific code-summarization** (2022), arxiv, A., Toufique, and P. Devanbu. [[pdf]](https://arxiv.org/pdf/2207.04237)
- **Are We Building on the Rock? On the Importance of Data Preprocessing for Code Summarization** (2022), FSE'22, Shi, Lin, et al. [[pdf]](https://arxiv.org/pdf/2207.05579)
- **Learning code summarization from a small and local dataset** (2022), arxiv, Ahmed, Toufique, and Devanbu, P. [[pdf]](https://arxiv.org/pdf/2206.00804)
- **Modeling Hierarchical Syntax Structure with Triplet Position for Source Code Summarization** (2022), ACL'22, Guo, Juncai, et al. [[pdf]](https://aclanthology.org/2022.acl-long.37.pdf)
- **AST-Trans: Code Summarization with Efficient Tree-Structured Attention** (2022), ICSE'22, Tang, Ze, et al. [[pdf]](http://lichuanyi.info/files/papers/2022-Ze%20Tang-AST-Trans%20ICSE2022.pdf)
- **GypSum: Learning Hybrid Representations for Code Summarization** (2022), ICPC'22, Wang, Yu, et al. [[pdf]](https://arxiv.org/pdf/2204.12916)
- **M2TS: Multi-Scale Multi-Modal Approach Based on Transformer for Source Code Summarization** (2022), ICPC'22, Gao, Yuexiu and Lyu, Chen [[pdf]](https://arxiv.org/pdf/2203.09707)
- **Project-Level Encoding for Neural Source Code Summarization of Subroutines** (2021), ICPC'21, Bansal, Aakash, et al. [[pdf]](https://arxiv.org/pdf/2103.11599)
- **Code Structure Guided Transformer for Source Code Summarization** (2021), arxiv 2021, Gao, Shuzheng, et al. [[pdf]](https://arxiv.org/pdf/2104.09340)
- **Source Code Summarization Using Attention-Based Keyword Memory Networks** (2020), IEEE BigComp 2020, Choi, YunSeok, et al.
- **A Transformer-based Approach for Source Code Summarization** (2020), arxiv 2020, Ahmad, Wasi Uddin, et al. [[pdf]](https://arxiv.org/pdf/2005.00653)
- **Learning to Represent Programs with Graphs** (2018), ICLR'18, Allamanis, Miltiadis, et al. [[pdf]](https://arxiv.org/pdf/1711.00740)
- **A Convolutional Attention Network for Extreme Summarization of Source Code** (2016), ICML 2016, Allamanis, Miltiadis, et al. [[pdf]](http://www.jmlr.org/proceedings/papers/v48/allamanis16.pdf)

## Code Embeddings/Representation

- **TransCoder: Towards Unified Transferable Code Representation Learning Inspired by Human Skills** (2023), arxiv, Sun, Qiushi, et al. [[pdf]](https://arxiv.org/pdf/2306.07285)
- **CodeGrid: A Grid Representation of Code** (2023), ISSTA'23, Kaboré, Abdoul Kader, et al.
- **Symmetry-Preserving Program Representations for Learning Code Semantics** (2023), arxiv, Pei, Kexin, et al. [[pdf]](https://arxiv.org/pdf/2308.03312)
- **PERFOGRAPH: A Numerical Aware Program Graph Representation for Performance Optimization and Program Analysis** (2023), arxiv, TehraniJamsaz, Ali, et al. [[pdf]](https://arxiv.org/pdf/2306.00210)
- **xASTNN: Improved Code Representations for Industrial Practice** (2023), arxiv, Xu, Zhiwei, et al. [[pdf]](https://arxiv.org/pdf/2303.07104)
- **Toward Interpretable Graph Tensor Convolution Neural Network for Code Semantics Embedding** (2023), TOSEM, Yang, Jia, et al. 
- **Practical Binary Code Similarity Detection with BERT-based Transferable Similarity Learning** (2022), ACSAC'22, Ahn, Sunwoo, et al.
- **CLAWSAT: Towards Both Robust and Accurate Code Models** (2022), arxiv, Jia, Jinghan, et al. [[pdf]](https://arxiv.org/pdf/2211.11711)
- **sem2vec: Semantics-Aware Assembly Tracelet Embedding** (2022), TSE, Wang, Huaijin, et al.
- **COMBO: Pre-Training Representations of Binary Code Using Contrastive Learning** (2022), arxiv, Zhang, Yifan, et al. [[pdf]](https://arxiv.org/pdf/2210.05102.pdf)
- **Soft-Labeled Contrastive Pre-training for Function-level Code Representation** (2022), arxiv, Li, Xiaonan, et al. [[pdf]](https://arxiv.org/pdf/2210.09597) 
- **A Tree-structured Transformer for Program Representation Learning** (2022), arxiv, Wang, Wenhan, et al. [[pdf]](https://arxiv.org/pdf/2208.08643)
- **What does Transformer learn about source code?** (2022), arxiv, Zhang, Kechi, et al. [[pdf]](https://arxiv.org/pdf/2207.08466)
- **Diet Code is Healthy: Simplifying Programs for Pre-Trained Models of Code** (2022), arxiv, Zhang, Zhaowei, et al. [[pdf]](https://arxiv.org/pdf/2206.14390)
- **MetaTPTrans: A Meta Learning Approach for Multilingual Code Representation Learning** (2022), arxiv, Pian, Weiguo, et al. [[pdf]](https://arxiv.org/pdf/2206.06460)
- **Towards Learning (Dis)-Similarity of Source Code from Program Contrasts** (2022), ACL'22, Ding, Yangruibo, et al. [[pdf]](https://aclanthology.org/2022.acl-long.436/)
- **Towards Learning Generalizable Code Embeddings using Task-agnostic Graph Convolutional Networks** (2022), TOSEM, Ding, Zishuo, et al.
- **Learning to Represent Programs with Code Hierarchies** (2022), arxiv, Nguyen, Minh and Nghi DQ Bui, [[pdf]](https://arxiv.org/pdf/2205.15479)
- **CV4Code: Sourcecode Understanding via Visual Code Representations** (2022), arxiv, Shi, Ruibo, et al. [[pdf]](https://arxiv.org/pdf/2205.08585)
- **Hyperbolic Representations of Source Code** (2022), AAAI'22, Khan, Raiyan, et al. [[pdf]](https://assets.amazon.science/55/d9/58097f0d41b886269b30e5c68522/hyperbolic-representations-of-source-code.pdf)
- **Unified Abstract Syntax Tree Representation Learning for Cross-Language Program Classification** (2022), ICPC'22, Wang, Kesu, et al. [[pdf]](https://arxiv.org/pdf/2205.00424)
- **Hierarchical Semantic-Aware Neural Code Representation** (2022), JSS'22, Jiang, Yuan, et al.
- **CODE-MVP: Learning to Represent Source Code from Multiple Views with Contrastive Pre-Training** (2022), arxiv 2022, Wang, Xin, et al. [[pdf]](https://arxiv.org/pdf/2205.02029)
- **Hierarchical Heterogeneous Graph Attention Network for Syntax-Aware Summarization** (2022), AAAI'22, Song, Z., and King, I., [[pdf]](https://www.aaai.org/AAAI22Papers/AAAI-6812.SongZ.pdf)
- **Unleashing the Power of Compiler Intermediate Representation to Enhance Neural Program Embeddings** (2022), ICSE'22, Li, Zongjie, et al. [[pdf]](https://arxiv.org/pdf/2204.09191.pdf)
- **XCode: Towards Cross-Language Code Representation with Large-Scale Pre-Training** (2022), TOSEM'22, Lin, Zehao, et al.
- **Fold2Vec: Towards a Statement Based Representation of Code for Code Comprehension** (2022), TOSEM'22, Bertolotti, Francesco and Cazzola, Walter
- **HELoC: Hierarchical Contrastive Learning of Source Code Representation** (2022), ICPC'22, Wang, Xiao, et al. [[pdf]](https://arxiv.org/pdf/2203.14285)
- **Multi-View Graph Representation for Programming Language Processing: An Investigation into Algorithm Detection** (2022), AAAI'22, Long, Tin et al. [[pdf]](https://www.aaai.org/AAAI22Papers/AAAI-928.LongT.pdf)
- **UniXcoder: Unified Cross-Modal Pre-training for Code Representation** (2022), arxiv 2022, Guo, Daya, et al. [[pdf]](https://arxiv.org/pdf/2203.03850)
- **SPT-Code: Sequence-to-Sequence Pre-Training for Learning Source Code Representations** (2022), ICSE'22, Niu, Changan, et al. [[pdf]](https://arxiv.org/pdf/2201.01549.pdf)
- **GraphCode2Vec: Generic Code Embedding via Lexical and Program Dependence Analyses** (2022), MSR'22, Ma, Wei, et al.
- **OSCAR: How could Neural Networks understand Programs?** (2021), ICML'21, Peng, Dinglan, et al. [[pdf]](http://proceedings.mlr.press/v139/peng21b/peng21b.pdf)
- **PROGRAML: A Graph-based Program Representation for Data Flow Analysis and Compiler Optimizations** (2021), ICML'21, Cummins, Chris, et al. [[pdf]](http://proceedings.mlr.press/v139/cummins21a/cummins21a.pdf)
- **CoTexT: Multi-task Learning with Code-Text Transformer** (2021), arxiv, Phan, Long, et al. [[pdf]](https://arxiv.org/pdf/2105.08645)
- **TreeCaps: Tree-Based Capsule Networks for Source Code Processing** (2021), AAAI'21, Bui, Nghi DQ, et al. [[pdf]](https://www.aaai.org/AAAI21Papers/AAAI-9746.BuiNDQ.pdf) [[code]](https://github.com/bdqnghi/treecaps)
- **Language-Agnostic Representation Learning of Source Code from Structure and Context** (2021), ICLR'21, Zügner, Daniel, et al. [[pdf]](https://arxiv.org/pdf/2103.11318)
- **IR2Vec: LLVM IR Based Scalable Program Embeddings** (2020), TACO journal, VenkataKeerthy, S., et al. 
- **Compiler-Based Graph Representations for Deep Learning Models of Code** (2020), CC'20, Brauckmann, Alexander, et al.
- **Learning and Evaluating Contextual Embedding of Source Code** (2020), ICML 2020, Kanade, Aditya, et al. [[pdf]](http://proceedings.mlr.press/v119/kanade20a/kanade20a.pdf)
- **Learning Semantic Program Embeddings with Graph Interval Neural Network** (2020), OOPSLA'20, Wang, Yu, et al.
- **Contrastive Code Representation Learning** (2020), arxiv 2020, Jain, Paras, et al. [[pdf]](https://arxiv.org/pdf/2007.04973.pdf)
- **SCELMo: Source Code Embeddings from Language Models** (2020), arxiv 2020, Karampatsis, Rafael-Michael, et al. [[pdf]](https://arxiv.org/pdf/2004.13214)
- **code2vec: Learning Distributed Representations of Code** (2019), ACM POPL 2019, Alon, Uri, et al. [[pdf]](http://www.cs.technion.ac.il/~mbs/publications/code2vec-popl19.pdf)
- **COSET: A Benchmark for Evaluating Neural Program Embeddings** (2019), arxiv 2019, Wang, Ke, et al. [[pdf]](https://arxiv.org/pdf/1905.11445)
- **A Literature Study of Embeddings on Source Code** (2019), arxiv 2019, Chen, Zimin, et al. [[pdf]](https://arxiv.org/pdf/1904.03061)
- **code2seq: Generating Sequences from Structured Representations of Code** (2018), arxiv 2018, Alon, Uri, et al. [[pdf]](https://arxiv.org/pdf/1808.01400)
- **Neural Code Comprehension: A Learnable Representation of Code Semantics** (2018), NIPS 2018, Ben-Nun, Tal, et al. [[pdf]](http://papers.nips.cc/paper/7617-neural-code-comprehension-a-learnable-representation-of-code-semantics.pdf)
- **Convolutional Neural Networks over Tree Structures for Programming Language Processing** (2016), AAAI'16, Mou, Lili, et al. [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11775/11735)

## Code Changes/Editing

- **Automated Code Editing with Search-Generate-Modify** (2023), arxiv, Liu, Changshu, et al. [[pdf]](https://arxiv.org/pdf/2306.06490)
- **Multilingual Code Co-Evolution Using Large Language Models** (2023), arxiv, Zhang, Jiyang, et al. [[pdf]](https://arxiv.org/pdf/2307.14991)
- **Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing** (2023), arxiv, Wei, Jiayi, et al. [[pdf]](https://arxiv.org/pdf/2305.18584)
- **CCT5: A Code-Change-Oriented Pre-Trained Model** (2023), arxiv, Lin, Bo, et al. [[pdf]](https://arxiv.org/pdf/2305.10785)
- **GrACE: Generation using Associated Code Edits** (2023), arxiv, Gupta, Priyanshu, et al. [[pdf]](https://arxiv.org/pdf/2305.14129)
- **Slice-Based Code Change Representation Learning** (2023), arxiv, Zhang, Fengyi, et al. [[pdf]](https://chenbihuan.github.io/paper/saner23-zhang-ccs2vec.pdf)
- **Towards Generating Functionally Correct Code Edits from Natural Language Issue Descriptions** (2023), arxiv, Fakhoury, Sarah, et al. [[pdf]](https://arxiv.org/pdf/2304.03816)
- **CCRep: Learning Code Change Representations via Pre-Trained Code Model and Query Back** (2023), arxiv, Liu, Zhongxin, et al. [[pdf]](https://arxiv.org/pdf/2302.03924)
- **CoditT5: Pretraining for Source Code and Natural Language Editing** (2022), ASE 2022, Jiyang, Zhang, et al. [[pdf]](https://arxiv.org/abs/2208.05446)
- **Commit2Vec: Learning Distributed Representations of Code Changes** (2021), SN Computer Science, Lozoya, Rocío Cabrera, et al.
- **CODIT: Code Editing with Tree-Based Neural Models** (2020), TSE 2020, Chakraborty, Saikat, et al.
- **On learning meaningful code changes via neural machine translation** (2019), ICSE 2019, Tufano, Michele, et al.

## Code Comments

- **Large Language Models are Few-Shot Summarizers: Multi-Intent Comment Generation via In-Context Learning** (2023), ICSE'24, Geng, Mingyang, et al. [[pdf]](https://www.researchgate.net/profile/Shangwen-Wang/publication/370228019_Large_Language_Models_are_Few-Shot_Summarizers_Multi-Intent_Comment_Generation_via_In-Context_Learning/links/648aeb559605ba270e45bf26/Large-Language-Models-are-Few-Shot-Summarizers-Multi-Intent-Comment-Generation-via-In-Context-Learning.pdf)
- **Snippet Comment Generation Based on Code Context Expansion** (2023), arxiv, GUO, HANYANG, et al.
- **An Empirical Study on Using Large Language Models for Multi-Intent Comment Generation** (2023), arxiv, Geng, Mingyang, et al. [[pdf]](https://arxiv.org/pdf/2304.11384)
- **An Intra-Class Relation Guided Approach for Code Comment Generation** (2023), EACL'23, Wang, Zhenni, et al. [[pdf]](https://aclanthology.org/2023.findings-eacl.97.pdf)
- **APIContext2Com: Code Comment Generation by Incorporating Pre-Defined API Documentation** (2023), arxiv, Shahbazi, R., and Fard F. [[pdf]](https://arxiv.org/pdf/2303.01645)
- **Developer-Intent Driven Code Comment Generation** (2023), arxiv, Mu, Fangwen, et al. [[pdf]](https://arxiv.org/pdf/2302.07055)
- **ALSI-Transformer: Transformer-Based Code Comment Generation With Aligned Lexical and Syntactic Information** (2023), IEEE Access, Park, Youngmi, et al.

## Bug/Vulnerability Detection

- **Transformer-based Vulnerability Detection in Code at EditTime: Zero-shot, Few-shot, or Fine-tuning?** (2023), arxiv, Chan, Aaron, et al. [[pdf]](https://arxiv.org/pdf/2306.01754)
- **LIVABLE: Exploring Long-Tailed Classification of Software Vulnerability Types** (2023), arxiv, Wen, Xin-Cheng, et al. [[pdf]](https://arxiv.org/pdf/2306.06935)
- **Learning to Quantize Vulnerability Patterns and Match to Locate Statement-Level Vulnerabilities** (2023), arxiv, Fu, Michael, et al. [[pdf]](https://arxiv.org/pdf/2306.06109)
- **CPVD: Cross Project Vulnerability Detection Based on Graph Attention Network and Domain Adaptation** (2023), TSE, Zhang, Chunyong, et al. 
- **FLAG: Finding Line Anomalies (in code) with Generative AI** (2023), arxiv, Ahmad, Baleegh, et al. [[pdf]](https://arxiv.org/pdf/2306.12643)
- **A Novel Approach to Identify Security Controls in Source Code** (2023), arxiv, Okutan, Ahmet, et al. [[pdf]](https://arxiv.org/pdf/2307.05605)
- **Limits of Machine Learning for Automatic Vulnerability Detection** (2023), arxiv, Risse, N., & Böhme, M. [[pdf]](https://arxiv.org/pdf/2306.17193)
- **Detecting Condition-Related Bugs with Control Flow Graph Neural Network** (2023), ISTTA'23, Zhang, Jian, et al.
- **A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification** (2023), arxiv, Charalambous, Yiannis, et al. [[pdf]](https://arxiv.org/pdf/2305.14752)
- **An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph** (2023), arxiv, Islam, Nafis Tanveer, et al. [[pdf]](https://arxiv.org/pdf/2304.11072)
- **Large Language Models and Simple, Stupid Bugs** (2023), arxiv, Jesse, Kevin, et al. [[pdf]](https://arxiv.org/pdf/2303.11455)
- **Vulnerability Detection with Graph Simplification and Enhanced Graph Representation Learning** (2023), arxiv, Wen, Xin-Cheng, et al. [[pdf]](https://arxiv.org/pdf/2302.04675)
- **DeepVD: Toward Class-Separation Features for Neural Network Vulnerability Detection** (2023), arxiv, Wang, Wenbo, et al. [[pdf]](https://aashishyadavally.github.io/files/C4.pdf)
- **CSGVD: A deep learning approach combining sequence and graph embedding for source code vulnerability detection** (2023), JSS journal, Tang, Wei, et al.
- **Fixing Hardware Security Bugs with Large Language Models** (2023), arxiv, Ahmad, Baleegh, et al. [[pdf]](https://arxiv.org/pdf/2302.01215)
- **VulEye: A Novel Graph Neural Network Vulnerability Detection Approach for PHP Application** (2023), Applied Sciences journal, Lin, Chun, et al. [[pdf]](https://www.mdpi.com/2076-3417/13/2/825/pdf)
- **VDGraph2Vec: Vulnerability Detection in Assembly Code using Message Passing Neural Networks** (2022), ICMLA'22, Diwan, Ashita, et al. [[pdf]](https://dmas.lab.mcgill.ca/fung/pub/DLF22icmla.pdf)
- **VulChecker: Graph-based Vulnerability Localization in Source Code** (2022), Usenix, Mirsky, Yisroel, et al. [[pdf]](https://www.usenix.org/system/files/sec23summer_449-mirsky-prepub.pdf)
- **DeepVulSeeker: A Novel Vulnerability Identification Framework via Code Graph Structure and Pre-training Mechanism** (2022), arxiv, Wang, Jin, et al. [[pdf]](https://arxiv.org/pdf/2211.13097)
- **Compact Abstract Graphs for Detecting Code Vulnerability with GNN Models** (2022), ACSAC'22, Luo, Yu, et al.
- **An Empirical Study of Deep Learning Models for Vulnerability Detection** (2022), arxiv, Steenhoek, Benjamin, et al. [[pdf]](https://arxiv.org/pdf/2212.08109)
- **Variable-Based Fault Localization via Enhanced Decision Tree** (2022), arxiv, Jiang, Jiajun, et al. [[pdf]](https://arxiv.org/pdf/2211.11526)
- **SPVF: security property assisted vulnerability fixing via attention-based models** (2022), EMSE, Zhou, Zhou, et al. 
- **Modeling function-level interactions for file-level bug localization** (2022), EMSE, Liang, H., et al.
- **Practical Automated Detection of Malicious npm Packages** (2022), ICSE'22, Sejfia, A., and M. Schäfer [[pdf]](https://arxiv.org/pdf/2202.13953)
- **Machine Learning for Source Code Vulnerability Detection: What Works and What Isn't There Yet** (2022), IEEE Security & Privacy, Marjanov, Tina, et al.
- **Path-sensitive code embedding via contrastive learning for software vulnerability detection** (2022), ISSTA'22, Cheng, Xiao, et al.
- **VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection** (2022), arxiv 2022, Hanif, H. and Maffeis, S. [[pdf]](https://arxiv.org/pdf/2205.12424)
- **Katana: Dual Slicing-Based Context for Learning Bug Fixes** (2022), arxiv 2022, Sintaha, Mifta, et al. [[pdf]](https://arxiv.org/pdf/2205.00180)
- **LineVul: A Transformer-based Line-Level Vulnerability Prediction** (2022), MSR'22, Fu, M., & Tantithamthavorn, C. [[pdf]](https://www.researchgate.net/profile/Chakkrit-Tantithamthavorn/publication/359402890_LineVul_A_Transformer-based_Line-Level_Vulnerability_Prediction/links/623ee3d48068956f3c4cbede/LineVul-A-Transformer-based-Line-Level-Vulnerability-Prediction.pdf)[[code]](https://github.com/awsm-research/LineVul)
- **Transformer-Based Language Models for Software Vulnerability Detection: Performance, Model's Security and Platforms** (2022), arxiv 2022, Thapa, Chandra, et al. [[pdf]](https://arxiv.org/pdf/2204.03214.pdf)
- **LineVD: Statement-level Vulnerability Detection using Graph Neural Networks** (2022), MSR'22, Hin, David, et al. [[pdf]](https://arxiv.org/pdf/2203.05181)
- **Nalin: Learning from Runtime Behavior to Find Name-Value Inconsistencies in Jupyter Notebooks** (2022), ICSE'22, Patra, Jibesh, et al. [[pdf]](https://arxiv.org/pdf/2112.06186.pdf)
- **Hoppity: Learning graph transformations to detect and fix bugs in programs** (2020), ICLR 2020, Dinella, Elizabeth, et al. [[pdf]](https://openreview.net/pdf/9d37b18aba351f4294aa84e69ea330d1fa51c471.pdf)
- **Deep Learning based Software Defect Prediction** (2020), Neurocomputing, Qiao, Lei, et al.
- **Software Vulnerability Discovery via Learning Multi-domain Knowledge Bases** (2019), IEEE TDSC, Lin, Guanjun, et al.
- **Neural Bug Finding: A Study of Opportunities and Challenges** (2019), arxiv 2019, Habib, Andrew, et al. [[pdf]](https://arxiv.org/pdf/1906.00307)
- **Automated Vulnerability Detection in Source Code Using Deep Representation Learning** (2018), ICMLA 2018, Russell, Rebecca, et al.
- **DeepBugs: A Learning Approach to Name-based Bug Detection** (2018), ACM PL 2018, Pradel, Michael, et al. [[pdf]](http://software-lab.org/publications/DeepBugs_arXiv_1805.11683.pdf)
- **Automatically Learning Semantic Features for Defect Prediction** (2016), ICSE 2016, Wang, Song, et al.

## Source Code Modeling

- **CONCORD: Clone-aware Contrastive Learning for Source Code** (2023), ISSTA'23, Ding, Yangruibo, et al. [[pdf]](https://arxiv.org/pdf/2306.03234)
- **TRACED: Execution-aware Pre-training for Source Code** (2023), ICSE'24, Ding, Yangruibo, et al. [[pdf]](https://arxiv.org/pdf/2306.07487)
- **ContraBERT: Enhancing Code Pre-trained Models via Contrastive Learning** (2023), arxiv, Liu, Shangqing, et al. [[pdf]](https://arxiv.org/pdf/2301.09072)
- **ERNIE-Code: Beyond English-Centric Cross-lingual Pretraining for Programming Languages** (2022), arxiv, Chai, Yekun, et al. [[pdf]](https://arxiv.org/pdf/2212.06742)
- **Do Bugs Lead to Unnaturalness of Source Code?** (2022), FSE'22, Jiang, Yanjie, et al.
- **On the Naturalness of Bytecode Instructions** (2022), ASE'22, Choi, Y., and J. Nam. [[pdf]](https://isel.handong.edu/papers/ase22-140.pdf)
- **CodeBERT-nt: code naturalness via CodeBERT** (2022), arxiv, Khanfir, Ahmed, et al. [[pdf]](https://arxiv.org/pdf/2208.06042)
- **CommitBART: A Large Pre-trained Model for GitHub Commits** (2022), arxiv, Liu, S., et al, [[pdf]](https://arxiv.org/pdf/2208.08100) 
- **Towards Learning (Dis)-Similarity of Source Code from Program Contrasts** (2022), ACL'22, Ding, Yangruibo, et al. [[pdf]](https://aclanthology.org/2022.acl-long.436.pdf)
- **Multilingual training for Software Engineering** (2022), ICSE'22, Ahmed, Toufique, et al. [[pdf]](https://arxiv.org/pdf/2112.02043)
- **Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code** (2020), ICSE'20, Karampatsis, Rafael-Michael, et al.
- **Maybe Deep Neural Networks are the Best Choice for Modeling Source Code** (2019), arxiv 2019, Karampatsis, Rafael-Michael, et al. [[pdf]](https://arxiv.org/pdf/1903.05734)
- **Are Deep Neural Networks the Best Choice for Modeling Source Code?** (2017), FSE 2017, Hellendoorn, Vincent J., et al. [[pdf]](https://vhellendoorn.github.io/PDF/fse2017.pdf)

## Program Repair

- **Code Similarity and Location-Awareness Automatic Program Repair** (2023), Applied Sciences, Cao, Heling, et al. 
- **The Future Can’t Help Fix The Past: Assessing Program Repair In The Wild** (2023), RG, Kabadi, Vinay, et al. [[pdf]](https://www.researchgate.net/profile/Xuan-Bach-D-Le/publication/372788577_The_Future_Can't_Help_Fix_The_Past_Assessing_Program_Repair_In_The_Wild/links/64c8d8ff862f8d2999875f1e/The-Future-Cant-Help-Fix-The-Past-Assessing-Program-Repair-In-The-Wild.pdf)
- **Revisiting the Plastic Surgery Hypothesis via Large Language Models** (2023), arxiv, Xia, Chunqiu Steven et al. [[pdf]](https://arxiv.org/pdf/2303.10494)
- **A Survey on Automated Program Repair Techniques** (2023), arxiv, Huang, Kai, et al. [[pdf]](https://arxiv.org/pdf/2303.18184)
- **Keep the Conversation Going: Fixing 162 out of 337 bugs for $0.42 each using ChatGPT** (2023), arxiv, Xia, C. S., and Lingming Z. [[pdf]](https://arxiv.org/pdf/2304.00385)
- **MUFIN: Improving Neural Repair Models with Back-Translation** (2023), arxiv, Silva, André, et al. [[pdf]](https://arxiv.org/pdf/2304.02301)
- **Explainable Automated Debugging via Large Language Model-driven Scientific Debugging** (2023), arxiv, Kang, Sungmin, et al. [[pdf]](https://arxiv.org/pdf/2304.02195)
- **A study on Prompt Design, Advantages and Limitations of ChatGPT for Deep Learning Program Repair** (2023), arxiv, Cao, Jialun, et al. [[pdf]](https://arxiv.org/pdf/2304.08191)
- **ITER: Iterative Neural Repair for Multi-Location Patches** (2023), arxiv, Ye, He, and Martin M. [[pdf]](https://arxiv.org/pdf/2304.12015)
- **TraceFixer: Execution Trace-Guided Program Repair** (2023), arxiv, Bouzenia, Islem, et al. [[pdf]](https://arxiv.org/pdf/2304.12743)
- **PatchZero: Zero-Shot Automatic Patch Correctness Assessment** (2023), arxiv, Zhou, Xin, et al. [[pdf]](https://arxiv.org/pdf/2303.00202)
- **Rete: Learning Namespace Representation for Program Repair** (2023), ICSE'23, Parasaram, Nikhil et al. [[pdf]](https://mechtaev.com/files/icse23.pdf)
- **InferFix: End-to-End Program Repair with LLMs over Retrieval-Augmented Prompts** (2023), arxiv, Jin, Matthew, et al. [[pdf]](https://arxiv.org/pdf/2303.07263)
- **Automated Program Repair in the Era of Large Pre-trained Language Models** (2023), arxiv, Xia, C. S. et al. [[pdf]](https://lingming.cs.illinois.edu/publications/icse2023a.pdf)
- **KNOD: Domain Knowledge Distilled Tree Decoder for Automated Program Repair** (2023), ICSE'23, Jiang, Nan, et al. [[pdf]](https://arxiv.org/pdf/2302.01857)
- **Impact of Code Language Models on Automated Program Repair** (2023), ICSE'23, Jiang, Nan, et al. [[pdf]](https://arxiv.org/pdf/2302.05020)
- **Embedding Context as Code Dependencies for Neural Program Repair** (2023), ICST'23, Nashid, Noor, et al. [[pdf]](https://people.ece.ubc.ca/amesbah/resources/papers/icst23.pdf)
- **Tare: Type-Aware Neural Program Repair** (2023), arxiv, Zhu, Qihao, et al. [[pdf]](https://xiongyingfei.github.io/papers/ICSE23a.pdf)
- **Boosting Automated Patch Correctness Prediction via Pre-trained Language Mode** (2023), arxiv, Zhang, Quanjun, et al. [[pdf]](https://arxiv.org/pdf/2301.12453)
- **Conversational Automated Program Repair** (2023), arxiv, Xia, Chunqiu Steven et al. [[pdf]](https://arxiv.org/pdf/2301.13246) 
- **An Analysis of the Automatic Bug Fixing Performance of ChatGPT** (2023), arxiv, Sobania, Dominik, et al. [[pdf]](https://arxiv.org/pdf/2301.08653)
- **Improving Automated Program Repair with Domain Adaptation** (2023), arxiv, Zirak, A., and Hemati, H. [[pdf]](https://arxiv.org/pdf/2212.11414)
- **A Survey of Learning-based Automated Program Repair** (2023), arxiv, Zhang, Quanjun, et al.  [[pdf]](https://arxiv.org/pdf/2301.03270.pdf)
- **TransplantFix:  Graph Differencing-based Code Transplantation for Automated Program Repair** (2023), ASE'22, Yang, Deheng, et al. [[pdf]](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=8734&context=sis_research)
- **Program Repair: Survey** (2022), arxiv, Gao, Xiang, et al. [[pdf]](https://arxiv.org/pdf/2211.12787.pdf)
- **SelfAPR: Self-supervised Program Repair with Test Execution Diagnostics** (2022), ASE'22, He et al. [[pdf]](http://arxiv.org/pdf/2203.12755)
- **Neural Program Repair using Execution-based Backpropagation** (2022), ICSE'22, He et al. [[pdf]](https://arxiv.org/abs/2105.04123)
- **Practical Program Repair in the Era of Large Pre-trained Language Models** (2022), arxiv, Xia, C. S. et al. [[pdf]](https://arxiv.org/pdf/2210.14179)
- **SYNSHINE: improved fixing of Syntax Errors** (2022), IEEE TSE, Ahmed, T. et al.
- **TransRepair: Context-aware Program Repair for Compilation Errors** (2022), ASE'22, Li, Xueyang, et al. [[pdf]](https://arxiv.org/pdf/2210.03986)
- **Repairing Bugs in Python Assignments Using Large Language Models** (2022), arxiv, Zhang, Jialu, et al. [[pdf]](https://arxiv.org/pdf/2209.14876.pdf)
- **Repair Is Nearly Generation: Multilingual Program Repair with LLMs** (2022), arxiv, Joshi, Harshit, et al. [[pdf]](https://arxiv.org/pdf/2208.11640)
- **VulRepair: A T5-Based Automated Software Vulnerability Repair** (2022), FSE'22, Fu, Michael, et al. [[pdf]](https://www.researchgate.net/profile/Chakkrit-Tantithamthavorn/publication/362092639_VulRepair_A_T5-Based_Automated_Software_Vulnerability_Repair/links/62d67c1ef976fb7443cecc35/VulRepair-A-T5-Based-Automated-Software-Vulnerability-Repair.pdf)
- **Less Training, More Repairing Please: Revisiting Automated Program Repair via Zero-shot Learning** (2022), FSE'22, Xia, Chunqiu Steven, and Lingming Z. [[pdf]](https://arxiv.org/pdf/2207.08281)
- **Can we learn from developer mistakes? Learning to localize and repair real bugs from real bug fixes** (2022), arxiv, Richter, Cedric, and Heike W. [[pdf]](https://arxiv.org/pdf/2207.00301)
- **AdaptivePaste: Code Adaptation through Learning Semantics-aware Variable Usage Representations** (2022), arxiv 2022, Liu, Xiaoyu, et al. [[pdf]](https://arxiv.org/pdf/2205.11023)
- **DEAR: A Novel Deep Learning-based Approach for Automated Program Repair** (2022), ICSE'22, Li, Yi, et al. [[pdf]](https://arxiv.org/pdf/2205.01859)
- **TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer** (2021), ICML'21, Berabi, Berkay, et al. [[pdf]](http://proceedings.mlr.press/v139/berabi21a/berabi21a.pdf)
- **Neural Transfer Learning for Repairing Security Vulnerabilities in C Code** (2021), Chen, Zimin, et al. [[pdf]](https://arxiv.org/pdf/2104.08308)
- **Generating Bug-Fixes Using Pretrained Transformers** (2021), arxiv 2021, Drain, Dawn, et al. [[pdf]](https://arxiv.org/pdf/2104.07896)
- **Global Relational Models of Source Code** (2020), ICLR'20, Hellendoorn, Vincent J., et al. [[pdf]](https://openreview.net/pdf?id=B1lnbRNtwr)
- **Neural Program Repair by Jointly Learning to Localize and Repair** (2019), arxiv 2019, Vasic, Marko, et al. [[pdf]](https://arxiv.org/pdf/1904.01720)

## Program Translation

- **Attention, Compilation, and Solver-based Symbolic Analysis are All You Need** (2023), arxiv, Jana, Prithwish, et al. [[pdf]](https://arxiv.org/pdf/2306.06755)
- **Understanding the Effectiveness of Large Language Models in Code Translation** (2023), arxiv, Pan, Rangeet, et al. [[pdf]](https://arxiv.org/pdf/2308.03109)
- **On ML-Based Program Translation: Perils and Promises** (2023), arxiv, Malyala, Aniketh, et al. [[pdf]](https://arxiv.org/pdf/2302.10812)
- **Boosting Neural Networks to Decompile Optimized Binaries** (2022), ACSAC'22, Cao, Ying, et al.
- **The Effectiveness of Transformer Models for Analyzing Low-Level Programs** (2022), MIT Primes, Zifan Guo [[pdf]](https://math.mit.edu/research/highschool/primes/materials/2021/GuoCarl.pdf)
- **Code Translation with Compiler Representations** (2022), arxiv, Szafraniec, Marc, et al. [[pdf]](https://arxiv.org/pdf/2207.03578)
- **BabelTower: Learning to Auto-parallelized Program Translation** (2022), ICML'22, Wen, Yuanbo, et al. [[pdf]](https://proceedings.mlr.press/v162/wen22b/wen22b.pdf)
- **Multilingual Code Snippets Training for Program Translation** (2022), AAAI'22, Zhu, Ming, et al. [[pdf]](https://people.cs.vt.edu/~reddy/papers/AAAI22.pdf)
- **Semantics-Recovering Decompilation through Neural Machine Translation** (2021), arxiv 2021, Liang, Ruigang, et al. [[pdf]](https://arxiv.org/pdf/2112.15491.pdf)
- **Unsupervised Translation of Programming Languages** (2020), arxiv 2020, Lachaux, Marie-Anne et al. [[pdf]](https://arxiv.org/abs/2006.03511)

## Program Analysis

- **The Hitchhiker’s Guide to Program Analysis: A Journey with Large Language Models** (2023), arxiv, Li, Haonan, et al. [[pdf]](https://arxiv.org/pdf/2308.00245)
- **AutoPruner: Transformer-Based Call Graph Pruning** (2022), FSE'22, Le-Cong, Thanh, et al. [[pdf]](https://arxiv.org/pdf/2209.03230)[[code]](https://github.com/soarsmu/AutoPruner/)
- **Striking a Balance: Pruning False-Positives from Static Call Graphs** (2022), ICSE'22, Utture, Akshay, et al. [[pdf]](http://compilers.cs.ucla.edu/papers/balancing-callgraphs.pdf)[[code]](https://zenodo.org/record/6057691)

## Software Testing

- **Automatic Unit Test Generation for Deep Learning Frameworks based on API Knowledge** (2023), arxiv, Narayanan, A., et al. [[pdf]](https://arxiv.org/pdf/2307.00404)
- **Black-Box Prediction of Flaky Test Fix Categories Using Language Models** (2023), arxiv, Fatima, S., et al. [[pdf]](https://arxiv.org/pdf/2307.00012)
- **Large Language Models Are Zero-Shot Fuzzers: Fuzzing Deep-Learning Libraries via Large Language Models** (2023), ISSTA'23, Deng, Yinlin, et al. [[pdf]](https://arxiv.org/abs/2212.14834)
- **Understanding Large Language Model Based Fuzz Driver Generation** (2023), arxiv, Zhang, Cen, et al. [[pdf]](https://arxiv.org/pdf/2307.12469)
- **Universal Fuzzing via Large Language Models** (2023), arxiv, Xia, Chunqiu Steven, et al. [[pdf]](https://arxiv.org/pdf/2308.04748)
- **SAGA: Summarization-Guided Assert Statement Generation** (2023), arxiv, Zhang, Yuwei, et al. [[pdf]](https://arxiv.org/pdf/2305.14808)
- **Towards More Realistic Evaluation for Neural Test Oracle Generation** (2023), ISSTA'23, Liu, Zhongxin, et al. [[pdf]](https://arxiv.org/pdf/2305.17047)
- **LTM: Scalable and Black-box Similarity-based Test Suite Minimization based on Language Models** (2023), arxiv, Pan, Rongqi, et al. [[pdf]](https://arxiv.org/pdf/2304.01397)
- **ChatGPT and Software Testing Education: Promises & Perils** (2023), arxiv, Jalil, Sajed, et al. [[pdf]](https://arxiv.org/pdf/2302.03287)
- **Adaptive Test Generation Using a Large Language Model** (2023), arxiv, Schäfer, Max, et al. [[pdf]](https://arxiv.org/pdf/2302.06527)
- **CODAMOSA: Escaping Coverage Plateaus in Test Generation with Pre-trained Large Language Models** (2023), ICSE'23, Lemieux, Caroline, et al. [[pdf]](https://www.carolemieux.com/codamosa_icse23.pdf) 
- **Learning Deep Semantics for Test Completion** (2023), arxiv, Nie, Pengyu, et al. [[pdf]](https://arxiv.org/pdf/2302.10166)
- **A3Test: Assertion-Augmented Automated Test Case Generation** (2023), arxiv, Alagarsamy, Saranya, et al. [[pdf]](https://arxiv.org/pdf/2302.10352)
- **Efficient Mutation Testing via Pre-Trained Language Models** (2023), arxiv, Khanfir, Ahmed, et al. [[pdf]](https://arxiv.org/pdf/2301.03543)
- **Test2Vec: An Execution Trace Embedding for Test Case Prioritization** (2022), arxiv, Jabbar, Emad, et al. [[pdf]](https://arxiv.org/pdf/2206.15428.pdf)
- **Generating Accurate Assert Statements for Unit Test Cases using Pretrained Transformers** (2022), AST'22, Tufano, Michele, et al.
- **On Learning Meaningful Assert Statements for Unit Test Cases** (2020), ICSE'20, Watson, Cody, et al.

## Code Clone Detection

- **Comparison and Evaluation of Clone Detection Techniques with Different Code Representations** (2023), ICSE'23, Wang, Yuekun, et al. [[pdf]](https://wu-yueming.github.io/Files/ICSE2023_TACC.pdf)
- **Towards Understanding the Capability of Large Language Models on Code Clone Detection: A Survey** (2023), arxiv, Dou, Shihan, et al. [[pdf]](https://arxiv.org/pdf/2308.01191)
- **CCT-Code: Cross-Consistency Training for Multilingual Clone Detection and Code Search** (2023), arxiv, Sorokin, Nikita, et al. [[pdf]](https://arxiv.org/pdf/2305.11626.pdf)
- **Neuro-symbolic Zero-Shot Code Cloning with Cross-Language Intermediate Representation** (2023), arxiv, Hasija, Krishnam, et al. [[pdf]](https://arxiv.org/pdf/2304.13350)
- **Pathways to Leverage Transcompiler based Data Augmentation for Cross-Language Clone Detection** (2023), arxiv, Pinku, Subroto Nag et al. [[pdf]](https://arxiv.org/pdf/2303.01435)
- **Graph-based code semantics learning for efficient semantic code clone detection** (2022), IST journal, Yu, Dongjin, et al.
- **Efficient transformer with code token learner for code clone detection** (2022), arxiv, Zhang, Aiping, et al.
- **Evaluation of Contrastive Learning with Various Code Representations for Code Clone Detection** (2022), arxiv, Zubkov, Maksim, et al. [[pdf]](https://arxiv.org/pdf/2206.08726)
- **Cross-Language Source Code Clone Detection Using Deep Learning with InferCode** (2022), arxiv 2022, Yahya, M., and Kim, D., [[pdf]](https://arxiv.org/pdf/2205.04913)
- **funcGNN: A Graph Neural Network Approach to Program Similarity** (2020), ESEM'20, Nair, Aravind, et al. [[pdf]]()
- **Cross-Language Clone Detection by Learning Over Abstract Syntax Trees** (2019), MSR'19, Perez, Daniel, et al.
- **The Adverse Effects of Code Duplication in Machine Learning Models of Code** (2019), Onward! 2019, Allamanis, Miltiadis, [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3359591.3359735)

## Code Search

- **Self-Supervised Query Reformulation for Code Search** (2023), FSE'23, Mao, Yuetian, et al. [[pdf]](https://arxiv.org/pdf/2307.00267)
- **Evaluating and Optimizing the Effectiveness of Neural Machine Translation in Supporting Code Retrieval Models: A Study on the CAT Benchmark** (2023), arxiv, P. Hung, and A. Jannesari. [[pdf]](https://arxiv.org/pdf/2308.04693)
- **Improving Code Search with Multi-Modal Momentum Contrastive Learning** (2023), ICPC'23, Shi, Zejian, et al. [[pdf]](https://szj2935.github.io/icpc2023mococs.pdf)
- **MulCS: Towards a Unified Deep Representation for Multilingual Code Search** (2023), SANER'23, Ma, Yingwei, et al. [[pdf]](https://yuyue.github.io/res/paper/MulCS-saner2023.pdf)
- **A mutual embedded self-attention network model for code search** (2023), JSS, Hu, Haize, et al.
- **You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search** (2022), FSE'22, Wan, Yao, et al.
- **How to Better Utilize Code Graphs in Semantic Code Search?** (2022), FSE'22, Shi, Yucen, et al.
- **Exploring Representation-Level Augmentation for Code Search** (2022), EMNLP'22, Li, Haochen, et al. [[pdf]](https://arxiv.org/pdf/2210.12285)[[code]](https://github.com/Alex-HaochenLi/RACS)
- **A code search engine for software ecosystems** (2022), CEUR, Pfaff, Chris, et al. [[pdf]](https://benevol2022.github.io/papers/ChrisPfaff.pdf)
- **Cross-Domain Deep Code Search with Meta Learning** (2022), ICSE'22, Chai, Yitian, et al. [[pdf]](https://guxd.github.io/papers/cdcs.pdf)

## Code Language Models

- **Gorilla: Large Language Model Connected with Massive APIs** (2023), arxiv, Patil, Shishir G., et al. [[pdf]](https://arxiv.org/pdf/2305.15334)
- **CodeT5+: Open Code Large Language Models for Code Understanding and Generation** (2023), arxiv, Wang, Yue, et al. [[pdf]](https://arxiv.org/pdf/2305.07922)
- **Better Language Models of Code through Self-Improvement** (2023), arxiv, To, Hung Quoc, et al. [[pdf]](https://arxiv.org/pdf/2304.01228)
- **A Systematic Evaluation of Large Language Models of Code** (2022), arxiv 2022, Xu, Frank F., et al. [[pdf]](https://arxiv.org/pdf/2202.13169)[[code]](https://github.com/VHellendoorn/Code-LMs)
- **CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation** (2021), EMNLP'21, Wang, Yue, et al. [[pdf]](https://aclanthology.org/2021.emnlp-main.685.pdf)
- **JavaBERT: Training a Transformer-Based Model for the Java Programming Language** (2021), ASEW'21, De Sousa, N. T., and W. Hasselbring
- **TreeBERT: A Tree-Based Pre-Trained Model for Programming Language** (2021), UAI'21, Jiang, Xue, et al. [[pdf]](https://proceedings.mlr.press/v161/jiang21a/jiang21a.pdf)
- **PLBART: Unified Pre-training for Program Understanding and Generation** (2021), NAACL'21, Ahmad, Wasi Uddin, et al. [[pdf]](https://arxiv.org/pdf/2103.06333)
- **Evaluating Large Language Models Trained on Code** (2021), arxiv 2021, Chen, Mark, et al. [[pdf]](https://arxiv.org/pdf/2107.03374.pdf?ref=https://githubhelp.com) [[code]](https://github.com/openai/human-eval)
- **GraphCodeBERT: Pre-training Code Representations with Data Flow** (2021), arxiv, Guo, Daya, et al. [[pdf]](https://arxiv.org/pdf/2009.08366)
- **C-BERT: Exploring Software Naturalness through Neural Language Models** (2020), arxiv, Buratti, Luca, et al. [[pdf]](https://arxiv.org/pdf/2006.12641)
- **Codebert: A Pre-trained Model for Programming and Natural Languages** (2020), arxiv 2020, Feng, Zhangyin, et al. [[pdf]](https://arxiv.org/pdf/2002.08155)

## Code Review

- **Learning to Predict Code Review Completion Time In Modern Code Review** (2023), EMSE journal, Chouchen, Moataz, et al. 
- **ReviewRanker: A Semi-Supervised Learning Based Approach for Code Review Quality Estimation** (2023), arxiv, Mahbub, Saifullah, et al. [[pdf]](https://arxiv.org/pdf/2307.03996)
- **ToxiSpanSE: An Explainable Toxicity Detection in Code Review Comments** (2023), arxiv, Saker, Jaydeb, et al. [[pdf]](https://arxiv.org/pdf/2307.03386)
- **Generation-based Code Review Automation: How Far Are We?** (2023), arxiv, Zhou, Xin, et al. [[pdf]](https://arxiv.org/pdf/2303.07221)
- **D-ACT: Towards Diff-Aware Code Transformation for Code Review Under a Time-Wise Evaluation** (2023), arxiv, Pornprasit, Chanathip, et al. [[pdf]](https://www.researchgate.net/profile/Chakkrit-Tantithamthavorn/publication/367075263_D-ACT_Towards_Diff-Aware_Code_Transformation_for_Code_Review_Under_a_Time-Wise_Evaluation/links/63c03c9556d41566df5e52f2/D-ACT-Towards-Diff-Aware-Code-Transformation-for-Code-Review-Under-a-Time-Wise-Evaluation.pdf)
- **Automating Code Review Activities by Large-Scale Pre-training** (2022), FSE'22, Li, Zhiyu, et al. [[pdf]]


## Empirical Studies

- **Are Code Pre-trained Models Powerful to Learn Code Syntax and Semantics?** (2023), arxiv, Ma, Wei, et al. [[pdf]](https://openreview.net/pdf?id=z_juuZ7yZWc)
- **Can Transformers Learn to Solve Problems Recursively?** (2023), arxiv, Zhang, S. D., et al. [[pdf]](https://arxiv.org/pdf/2305.14699)
- **CODEIPPROMPT: Intellectual Property Infringement Assessment of Code Language Models** (2023), ICML'23, Yu, Zhiyuan, et al. [[pdf]](https://openreview.net/pdf?id=zdmbZl0ia6)
- **Towards Understanding What Code Language Models Learned** (2023), arxiv, Ahmed, Toufique, et al. [[pdf]](https://arxiv.org/pdf/2306.11943)
- **Exploring the Effectiveness of LLMs in Automated Logging Generation: An Empirical Study** (2023), arxiv, Li, Yichen, et al. [[pdf]](https://arxiv.org/pdf/2307.05950)
- **Is this Snippet Written by ChatGPT? An Empirical Study with a CodeBERT-Based Classifier** (2023), arxiv, Nguyen, Phuong T., et al. [[pdf]](https://arxiv.org/pdf/2307.09381)
- **An Empirical Study on the Effectiveness of Noisy Label Learning for Program Understanding** (2023), arxiv, Wang, Wenhan, et al. [[pdf]](https://arxiv.org/pdf/2307.08990)
- **Who Answers It Better? An In-Depth Analysis of ChatGPT and Stack Overflow Answers to Software Engineering Qestions** (2023), arxiv, Kabir, Samia, et al. [[pdf]](https://arxiv.org/pdf/2308.02312)
- **Adaptive Intellect Unleashed: The Feasibility of Knowledge Transfer in Large Language Models** (2023), arxiv, Huang, Qing, et al. [[pdf]](https://arxiv.org/pdf/2308.04788)
- **Can Large Language Models Reason About Program Invariants?** (2023), ICML'23, Sutton, Charles, et al.
- **The Scope of ChatGPT in Software Engineering: A Thorough Investigation** (2023), arxiv, Ma, Wei, et al. [[pdf]](https://arxiv.org/pdf/2305.12138)
- **Evaluating AIGC Detectors on Code Content** (2023), arxiv, Wang, Jian, et al. [[pdf]](https://arxiv.org/pdf/2304.05193)
- **“What It Wants Me To Say”: Bridging the Abstraction Gap Between End-User Programmers and Code-Generating Large Language Models** (2023), CHI'23, Liu, Michael Xieyang, et al. [[pdf]](https://arxiv.org/pdf/2304.06597.pdf)
- **Constructing Effective In-Context Demonstration for Code Intelligence Tasks: An Empirical Study** (2023), arxiv, Gao, Shuzheng, et al. [[pdf]](https://arxiv.org/pdf/2304.07575)
- **Automated Program Repair Based on Code Review: How do Pre-trained Transformer Models Perform?** (2023), arxiv, Paul, Rishov, et al. [[pdf]](https://arxiv.org/pdf/2304.07840) 
- **Investigating Code Generation Performance of ChatGPT with Crowdsourcing Social Data** (2023), COMPSAC'23, Feng, Yunhe, et al. [[pdf]](https://yunhefeng.me/material/COMPSAC___SETA_23.pdf)
- **Evaluating the Code Quality of AI-Assisted Code Generation Tools: An Empirical Study on GitHub Copilot, Amazon CodeWhisperer, and ChatGPT** (2023), arxiv, Yetiştiren, Burak, et al. [[pdf]](https://arxiv.org/pdf/2304.10778)
- **Is ChatGPT the Ultimate Programming Assistant - How far is it?** (2023), arxiv, Tian, Haoye, et al. [[pdf]](https://arxiv.org/pdf/2304.11938)
- **Study of Distractors in Neural Models of Code** (2023), InteNSE'23, Rabin, Md Rafiqul Islam, et al. [[pdf]](https://arxiv.org/pdf/2303.01739)
- **Judging Adam: Studying the Performance of Optimization Methods on ML4SE Tasks** (2023), arxiv, Pasechnyuk, Dmitry, et al. [[pdf]](https://arxiv.org/pdf/2303.03540)
- **Boosting Source Code Learning with Data Augmentation: An Empirical Study** (2023), arxiv, Dong, Zeming, et al. [[pdf]](https://arxiv.org/pdf/2303.06808)
- **Source Code Recommender Systems: The Practitioners’ Perspective** (2023), arxiv, Ciniselli, Matteo, et al. [[pdf]](https://arxiv.org/pdf/2302.04098)
- **An Empirical Comparison of Pre-Trained Models of Source Code** (2023), arxiv, Niu, Changan, et al. [[pdf]](https://arxiv.org/pdf/2302.04026)
- **On the Reliability and Explainability of Automated Code Generation Approaches** (2023), arxiv, Liu, Yue, et al. [[pdf]](https://arxiv.org/pdf/2302.09587)
- **On the Robustness of Code Generation Techniques: An Empirical Study on GitHub Copilot** (2023), arxiv, Mastropaolo, Antonio, et al. [[pdf]](https://arxiv.org/pdf/2302.00438)
- **Practitioners’ Expectations on Code Completion** (2023), arxiv, Wang, Chaozheng, et al. [[pdf]](https://arxiv.org/pdf/2301.03846)
- **Is Self-Attention Powerful to Learn Code Syntax and Semantics?** (2022), arxiv, Ma, Wei, et al. [[pdf]](https://arxiv.org/pdf/2212.10017)
- **Piloting Copilot and Codex: Hot Temperature, Cold Prompts, or Black Magic?** (2022), arxiv, Döderlein et al. [[pdf]](https://arxiv.org/pdf/2210.14699)
- **Explainable AI for Pre-Trained Code Models: What Do They Learn? When They Do Not Work?** (2022), arxiv, Mohammadkhani, Ahmad Haji, et al. [[pdf]](https://arxiv.org/pdf/2211.12821)
- **How Important are Good Method Names in Neural Code Generation? A Model Robustness Perspective** (2022), arxiv, Yang, Guang, et al. [[pdf]](https://arxiv.org/pdf/2211.15844)
- **“It would work for me too”: How Online Communities Shape Software Developers’ Trust in AI-Powered Code Generation Tools** (2022), arxiv, Cheng, Ruijia, et al. [[pdf]](https://arxiv.org/pdf/2212.03491)
- **Are Neural Bug Detectors Comparable to Software Developers on Variable Misuse Bugs?** (2022), ASE'22, Richter, Cedric, et al. [[pdf]](https://fpauck.de/papers/ase22-203.pdf)
- **Do Pre-trained Language Models Indeed Understand Software Engineering Tasks?** (2022), arxiv, Li, Yao, et al. [[pdf]](https://arxiv.org/pdf/2211.10623)
- **A large-scale empirical study of commit message generation: models, datasets and evaluation** (2022), EMSE, Tao, Wei, et al.
- **Examining Zero-Shot Vulnerability Repair with Large Language Models** (2022), IEEE SP, Pearce, H., et al.
- **Extracting Meaningful Attention on Source Code: An Empirical Study of Developer and Neural Model Code Exploration** (2022), arxiv, Paltenghi, M., et al. [[pdf]](https://arxiv.org/pdf/2210.05506)
- **SimSCOOD: Systematic Analysis of Out-of-Distribution Behavior of Source Code Models** (2022), arxiv, Hajipour, H., et al. [[pdf]](https://arxiv.org/pdf/2210.04802)
- **Are Neural Bug Detectors Comparable to Software Developers on Variable Misuse Bugs?** (2022), ASE'22, Richter, Cedric, et al. [[pdf]](https://fpauck.de/papers/ase22-203.pdf)
- **Open Science in Software Engineering: A Study on Deep Learning-Based Vulnerability Detection** (2022), TSE, Nong, Yu, et al. [[pdf]](https://www.researchgate.net/profile/Haipeng-Cai/publication/363535723_Open_Science_in_Software_Engineering_A_Study_on_Deep_Learning-Based_Vulnerability_Detection/links/63217b3b071ea12e3630cd6c/Open-Science-in-Software-Engineering-A-Study-on-Deep-Learning-Based-Vulnerability-Detection.pdf)
- **A controlled experiment of different code representations for learning-based program repair** (2022), EMSE, Namavar, M., et al. 
- **What is it like to program with artificial intelligence?** (2022), arxiv, Sarkar, Advait, et al. [[pdf]](https://arxiv.org/pdf/2208.06213)
- **Security Implications of Large Language Model Code Assistants: A User Study** (2022), arxiv, Sandoval, Gustavo, et al. [[pdf]](https://arxiv.org/pdf/2208.09727)
- **An Empirical Study of Code Smells in Transformer-based Code Generation Techniques** (2022), arxiv, Siddiq, M. L. et al. [[pdf]](https://lsiddiqsunny.github.io/public/scam_2022.pdf)
- **No More Fine-Tuning? An Experimental Evaluation of Prompt Tuning in Code Intelligence** (2022), FSE'22, Wang, Chaozheng, et al. [[pdf]](https://arxiv.org/pdf/2207.11680)
- **Generating Realistic Vulnerabilities via Neural Code Editing: An Empirical Study** (2022), FSE'22, Nong, Yu, et al. [[pdf]](https://chapering.github.io/pubs/fse22yu.pdf)
- **GitHub Copilot AI pair programmer: Asset or Liability?** (2022), arxiv, Dakhel, Arghavan Moradi, et al. [[pdf]](https://arxiv.org/pdf/2206.15331)
- **Evaluating the Impact of Source Code Parsers on ML4SE Models** (2022), arxiv, Utkin, Ilya, et al [[pdf]](https://arxiv.org/pdf/2206.08713)
- **An extensive study on pre-trained models for program understanding and generation** (2022), ISSTA'22, Zeng, Zhengran, et al.
- **Code Generation Tools (Almost) for Free? A Study of Few-Shot, Pre-Trained Language Models on Code** (2022), arxiv, Bareiß, Patrick, et al. [[pdf]](https://arxiv.org/pdf/2206.01335)
- **Assessing Project-Level Fine-Tuning of ML4SE Models** (2022), arxiv, Bogomolov, Egor, et al. [[pdf]](https://arxiv.org/pdf/2206.03333)
- **On the Transferability of Pre-trained Language Models for Low-Resource Programming Languages** (2022), ICPC'22, Chen, Fuxiang, et al. [[pdf]](https://arxiv.org/pdf/2204.09653.pdf)
- **Learning Program Semantics with Code Representations: An Empirical Study** (2022), SANER'22, Siow, Jing Kai, et al. [[pdf]](https://arxiv.org/pdf/2203.11790)[[code]](https://github.com/jingkai92/learning-program-representation)
- **Assessing Generalizability of CodeBERT** (2021), ICSME'21, Zhou, Xin, et al.
- **Thinking Like a Developer? Comparing the Attention of Humans with Neural Models of Code** (2021), ASE'21, Paltenghi, M. & Pradel, M.
- **An Empirical Study of Transformers for Source Code** (2021), FSE'21, Chirkova, N., & Troshin, S.
- **An Empirical Study on the Usage of Transformer Models for Code Completion** (2021), MSR'21, Ciniselli, Matteo, et al.

## Surveys

- **When Neural Model Meets NL2Code: A Survey** (2023), ACL'23, Zan, Daoguang, et al. [[pdf]](https://aclanthology.org/2023.acl-long.411.pdf)
- **Deep Learning Meets Software Engineering: A Survey on Pre-Trained Models of Source Code** (2022), arxiv 2022, Niu, Changan, et al. [[pdf]](https://arxiv.org/pdf/2205.11739)
- **A Survey of Deep Learning Models for Structural Code Understanding** (2022), arxiv 2022, Wu, Ruoting, et al. [[pdf]](https://arxiv.org/pdf/2205.01293)
- **A Survey on Machine Learning Techniques for Source Code Analysis** (2021), arxiv 2021, Sharma, Tushar, et al. [[pdf]](https://arxiv.org/pdf/2110.09610) 
- **Deep Learning & Software Engineering: State of Research and Future Directions** (2020), arxiv 2020, Devanbu, Prem, et al. [[pdf]](https://arxiv.org/pdf/2009.08525.pdf)
- **A Systematic Literature Review on the Use of Deep Learning in Software Engineering Research** (2020), arxiv 2020, Watson, Cody, et al. [[pdf]](https://arxiv.org/pdf/2009.06520.pdf)
- **Machine Learning for Software Engineering: A Systematic Mapping** (2020), arxiv 2020, Shafiq, Saad, et al. [[pdf]](https://arxiv.org/pdf/2005.13299.pdf)
- **Synergy between Machine/Deep Learning and Software Engineering: How Far Are We?** (2020), arxiv 2020, Wang, Simin, et al. [[pdf]](https://arxiv.org/pdf/2008.05515.pdf)
- **Software Engineering Meets Deep Learning: A Literature Review** (2020), arxiv 2020, Ferreira, Fabio, et al. [[pdf]](https://arxiv.org/pdf/1909.11436.pdf)
- **Software Vulnerability Detection Using Deep Neural Networks: A Survey** (2020), Proceedings of the IEEE, Lin, Guanjun, et al.
- **Deep Learning for Source Code Modeling and Generation: Models, Applications and Challenges** (2020), arxiv 2020, Le, Triet HM, et al. [[pdf]](https://arxiv.org/pdf/2002.05442)
- **A Survey of Machine Learning for Big Code and Naturalness** (2018), ACM Computing Surveys, Allamanis, Miltiadis, et al. [[pdf]](https://miltos.allamanis.com/publicationfiles/allamanis2018survey/allamanis2018survey.pdf)

## Misc

- **Predicting Dynamic Properties of Heap Allocations using Neural Networks Trained on Static Code** (2023), ISMM'23, Navasca, Christian, et al. 
- **Prompting Is All You Need: Automated Android Bug Replay with Large Language Models** (2023), ICSE'24, Feng, S., & Chen, C. [[pdf]](https://arxiv.org/pdf/2306.01987v)
- **LmPa: Improving Decompilation by Synergy of Large Language Model and Program Analysis** (2023), arxiv, Xu, Xiangzhe, et al. [[pdf]](https://arxiv.org/pdf/2306.02546) 
- **Stack Over-Flowing with Results: The Case for Domain-Specific Pre-Training Over One-Size-Fits-All Models** (2023), arxiv, Mukherjee, M. and Hellendoorn, V.J. [[pdf]](https://arxiv.org/pdf/2306.03268)
- **Faster sorting algorithms discovered using deep reinforcement learning** (2023), Nature, Mankowitz, Daniel J., et al. [[pdf]](https://www.nature.com/articles/s41586-023-06004-9)
- **SELFEVOLVE: A Code Evolution Framework via Large Language Models** (2023), arxiv, Jiang, S., et al. [[pdf]](https://arxiv.org/pdf/2306.02907)
- **The “Code” of Ethics: A Holistic Audit of AI Code Generators** (2023), arxiv, Ma, Wanlun, et al. [[pdf]](https://arxiv.org/pdf/2305.12747)
- **Too long; didn’t read: Automatic summarization of GitHub README.MD with Transformers** (2023), EASE'23, Doan, Thu TH, et al. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3593434.3593448) 
- **ARIST: An Effective API Argument Recommendation Approach** (2023), JSS, Nguyen, Son, et al. [[pdf]](https://arxiv.org/pdf/2306.06620.pdf)
- **A statistical approach for finding property-access errors** (2023), arxiv, Arteca, E., et al. [[pdf]](https://arxiv.org/pdf/2306.08741)
- **A Chain of AI-based Solutions for Resolving FQNs and Fixing Syntax Errors in Partial Code** (2023), arxiv, Huang, Qing, et al. [[pdf]](https://arxiv.org/pdf/2306.11981)
- **Guiding Language Models of Code with Global Context using Monitors** (2023), arxiv, Agrawal, Lakshya A., et al. [[pdf]](https://arxiv.org/pdf/2306.10763)
- **HotGPT: How to Make Software Documentation More Useful with a Large Language Model?** (2023), HOTOS'23, Su, Yiming, et al.
- **Can Large Language Models Reason about Program Invariants?** (2023), ICML'23, Pei, Kexin, et al. [[pdf]](https://openreview.net/pdf?id=mXv2aVqUGG) 
- **LLM4CBI: Taming LLMs to Generate Effective Test Programs for Compiler Bug Isolation** (2023), arxiv, Tu, Haoxin, et al. [[pdf]](https://arxiv.org/pdf/2307.00593)
- **Improving Binary Code Similarity Transformer Models by Semantics-Driven Instruction Deemphasis** (2023), ISSTA'23, Xu, Xiangzhe, et al. [[pdf]](https://qingkaishi.github.io/public_pdfs/ISSTA23.pdf) 
- **Exploring and Characterizing Large Language Models For Embedded System Development and Debugging** (2023), arxiv, Englhardt, Zachary, et al.  [[pdf]](https://arxiv.org/pdf/2307.03817)

- **Explaining Competitive-Level Programming Solutions using LLMs** (2023), arxiv, Li, Jierui, et al. [[pdf]](https://arxiv.org/pdf/2307.05337)
- **BTLink : automatic link recovery between issues and commits based on pre-trained BERT model** (2023), EMSE journal, Lan, Jinpeng, et al. 
- **In-IDE Generation-based Information Support with a Large Language Model** (2023), arxiv, Nam, Daye, et al. [[pdf]](https://arxiv.org/pdf/2307.08177)
- **Utilization of Pre-trained Language Model for Adapter-based Knowledge Transfer in Software Engineering** (2023), arxiv, Saberi, Iman, et al. [[pdf]](https://arxiv.org/pdf/2307.08540)
- **Contrastive Learning for API Aspect Analysis** (2023), arxiv, Shahariar, G. M., et al. [[pdf]](https://arxiv.org/pdf/2307.16878)
- **Fixing Rust Compilation Errors using LLMs** (2023), arxiv, Deligiannis, Pantazis, et al. [[pdf]](https://arxiv.org/pdf/2308.05177)
- **CodeLens: An Interactive Tool for Visualizing Code Representations** (2023), arxiv, Guo, Yuejun, et al. [[pdf]](https://arxiv.org/pdf/2307.14902)
- **Contrastive Learning for API Aspect Analysis** (2023), arxiv, Shahariar, G. M., et al. [[pdf]](https://arxiv.org/pdf/2307.16878)
- **COME: Commit Message Generation with Modification Embedding** (2023), ISSTA'23, He, Yichen, et al. 
- **Predicting Bug Fix Time in Students’ Programming with Deep Language Models** (2023), EDM'23, Tsabari, Stav, et al. [[pdf]](https://educationaldatamining.org/EDM2023/proceedings/2023.EDM-short-papers.40/2023.EDM-short-papers.40.pdf)
- **LaFiCMIL: Rethinking Large File Classification from the Perspective of Correlated Multiple Instance Learning** (2023), arxiv, Sun, Tiezhu, et al. [[pdf]](https://arxiv.org/pdf/2308.01413)
- **Evaluating and Explaining Large Language Models for Code Using Syntactic Structures** (2023), arxiv, Palacio, David N., et al. [[pdf]](https://arxiv.org/pdf/2308.03873)
- **Tuning Models of Code with Compiler-Generated Reinforcement Learning Feedback** (2023), arxiv, Jain, Abhinav, et al. [[pdf]](https://arxiv.org/pdf/2305.18341)
- **Evidence of Meaning in Language Models Trained on Programs** (2023), arxiv, Jin, C., & Rinard, M. [[pdf]](https://arxiv.org/pdf/2305.11169.pdf%22)
- **Neural Task Synthesis for Visual Programming** (2023), arxiv, Pădurean, V. A., et al. [[pdf]](https://arxiv.org/pdf/2305.18342)
- **AI for Low-Code for AI** (2023), arxiv, Rao, Nikitha, et al. [[pdf]](https://arxiv.org/pdf/2305.20015)
- **RefBERT: A Two-Stage Pre-trained Framework for Automatic Rename Refactoring** (2023), ISSTA'23, Liu, Hao, et al. [[pdf]](https://arxiv.org/pdf/2305.17708) 
- **Towards Tracing Code Provenance with Code Watermarking** (2023), arxiv, Li, Wei, et al. [[pdf]](https://arxiv.org/pdf/2305.12461)
- **SLaDe: A Portable Small Language Model Decompiler for Optimized Assembler** (2023), arxiv, Armengol-Estapé, Jordi, et al. [[pdf]](https://arxiv.org/pdf/2305.12520)
- **Text-to-SQL Error Correction with Language Models of Code** (2023), arxiv, Chen, Ziru, et al. [[pdf]](https://arxiv.org/pdf/2305.13073)
- **Improving API Knowledge Discovery with ML: A Case Study of Comparable API Methods** (2023), ICSE'23, Nam, Daye, et al. [[pdf]](https://dayenam.com/assets/pdf/icse23_SOREL.pdf)
- **Beryllium: Neural Search for Algorithm Implementations** (2023), arxiv, Kulkarni, Adithya, et al. [[pdf]](https://arxiv.org/pdf/2305.15690)
- **Zero-shot Prompting for Code Complexity Prediction Using GitHub Copilot** (2023), arxiv, Siddiq, Mohammed Latif, et al. [[pdf]](https://lsiddiqsunny.github.io/public/NLBSE_2023_zero_shot.pdf)
- **One Adapter for All Programming Languages? Adapter Tuning for Code Search and Summarization** (2023), arxiv, Wang, Deze, et al. [[pdf]](https://arxiv.org/pdf/2303.15822)
- **GraphBinMatch: Graph-based Similarity Learning for Cross-Language Binary and Source Code Matching** (2023), arxiv, TehraniJamsaz, Ali, et al. [[pdf]](https://arxiv.org/pdf/2304.04658)
- **Teaching Large Language Models to Self-Debug** (2023), arxiv, Chen, Xinyun, et al. [[pdf]](https://arxiv.org/pdf/2304.05128)
- **Improving Few-shot Prompts with Relevant Static Analysis Products** (2023), arxiv, Ahmed, Toufique, et al. [[pdf]](https://arxiv.org/pdf/2304.06815)
- **Self-Supervised Learning to Prove Equivalence Between Straight-Line Programs via Rewrite Rules** (2023), TSE, Kommrusch, Steve, et al.
- **XCODEEVAL: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval** (2023), arxiv, Khan, Mohammad Abdullah Matin, et al. [[pdf]](https://arxiv.org/pdf/2303.03004.pdf)
- **BenchDirect: A Directed Language Model for Compiler Benchmarks** (2023), arxiv, Tsimpourlas, Foivos, et al. [[pdf]](https://arxiv.org/pdf/2303.01557)
- **Creating CREATE queries with multi-task deep neural networks** (2023), KBS journal, Diker, S. N., and C. Okan Sakar
- **Representation Learning for Stack Overflow Posts: How Far are We?** (2023), arxiv, He, Junda, et al. [[pdf]](https://arxiv.org/pdf/2303.06853)
- **Model-Agnostic Syntactical Information for Pre-Trained Programming Language Models** (2023), arxiv, Saberi, I., and Fatemeh F. [[pdf]](https://arxiv.org/pdf/2303.06233)
- **Automating Method Naming with Context-Aware Prompt-Tuning** (2023), arxiv, Zhu, Jie, et al. [[pdf]](https://arxiv.org/pdf/2303.05771)
- **Knowledge Transfer for Pseudo-code Generation from Low Resource Programming Language** (2023), arxiv, Sontakke, Ankita, et al. [[pdf]](https://arxiv.org/pdf/2303.09062)
- **LExecutor: Learning-Guided Execution** (2023), arxiv, Souza, B., and M. Pradel [[pdf]](https://arxiv.org/pdf/2302.02343)
- **Keeping Pace with Ever-Increasing Data: Towards Continual Learning of Code Intelligence Models** (2023), arxiv, Gao, Shuzheng, et al. [[pdf]](https://arxiv.org/pdf/2302.03482)
- **CrossCodeBench: Benchmarking Cross-Task Generalization of Source Code Models** (2023), arxiv, Niu, Changan, et al. [[pdf]](https://arxiv.org/pdf/2302.04030)
- **On the Applicability of Language Models to Block-Based Programs** (2023), arxiv, Niu, Changan, et al. [[pdf]](https://arxiv.org/pdf/2302.04026)
- **AttSum: A Deep Attention-Based Summarization Model for Bug Report Title Generation** (2023), IEEE TOR, Ma, Xiaoxue, et al.
- **CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code** (2023), arxiv, Zhou, Shuyan, et al. [[pdf]](https://arxiv.org/pdf/2302.05527)
- **VULGEN: Realistic Vulnerability Generation Via Pattern Mining and Deep Learning** (2023), ICSE'23, Nong, Yu, et al. [[pdf]](https://software-lab.org/publications/icse2023_VulGen.pdf)
- **When to Say What: Learning to Find Condition-Message Inconsistencies** (2023), ICSE'23, B., Islem, and M. Pradel. [[pdf]](https://software-lab.org/publications/icse2023_CMI-Finder.pdf)
- **Automated Summarization of Stack Overflow Posts** (2023), ICSE'23, Kou, Bonan, et al. [[pdf]](https://tianyi-zhang.github.io/files/icse2023-assort.pdf)
- **Learning Graph-based Code Representations for Source-level Functional Similarity Detection** (2023), arxiv, Liu, Jiahao, et al. [[pdf]](https://jun-zeng.github.io/file/tailor_paper.pdf)
- **Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning** (2023), ICSE'23, Nashid, Noor, et al. [[pdf]](https://people.ece.ubc.ca/amesbah/resources/papers/icst23.pdf)

- **API Entity and Relation Joint Extraction from Text via Dynamic Prompt-tuned Language Model** (2023), arxiv, Huang, Qing, et al [[pdf]](https://arxiv.org/pdf/2301.03987)
- **FLAME: A small language model for spreadsheet formulas** (2023), arxiv, Joshi, Harshit, et al. [[pdf]](https://arxiv.org/pdf/2301.13779)
- **Callee: Recovering Call Graphs for Binaries with Transfer and Contrastive Learning** (2023), IEEE SP, Zhu, Wenyu, et al. 
- **Asteria-Pro: Enhancing Deep-Learning Based Binary Code Similarity Detection by Incorporating Domain Knowledge** (2023), arxiv, Yang, Shouguo, et al. [[pdf]](https://arxiv.org/pdf/2301.00511)
- **Fuzzing Deep-Learning Libraries via Large Language Models** (2022), arxiv, Deng, Yinlin, et al. [[pdf]](https://arxiv.org/pdf/2212.14834)
- **Extending Source Code Pre-Trained Language Models to Summarise Decompiled Binaries** (2023), SANER23, Al-Kaswan, Ali, et al. [[pdf]](https://arxiv.org/pdf/2301.01701)
- **CFG2VEC: Hierarchical Graph Neural Network for Cross-Architectural Software Reverse Engineering** (2023), arxiv, Yu, Shih-Yuan, et al.  [[pdf]](https://arxiv.org/pdf/2301.02723)
- **Recommending Root-Cause and Mitigation Steps for Cloud Incidents using Large Language Models** (2023), ICSE'23, Ahmed, Toufique, et al. [[pdf]](https://arxiv.org/pdf/2301.03797.pdf)
- **Detect-Localize-Repair: A Unified Framework for Learning to Debug with CodeT5** (2022), arxiv, Bui, Nghi DQ, et al. [[pdf]](https://arxiv.org/pdf/2211.14875)
- **Unleashing the power of pseudo-code for binary code similarity analysis** (2022), Cybersecurity journal, Zhang, Weiwei, et al.
- **Reinforcement Learning assisted Loop Distribution for Locality and Vectorization** (2022), Jain, Shalini, et al. [[pdf]](https://www.researchgate.net/profile/Dibyendu-Das/publication/365475992_Reinforcement_Learning_assisted_Loop_Distribution_for_Locality_and_Vectorization/links/637679e937878b3e87bb988e/Reinforcement-Learning-assisted-Loop-Distribution-for-Locality-and-Vectorization.pdf)
- **Learning to Parallelize Source Code via OpenMP with Transformers** (2022), Harel, Re’em, et al. [[pdf]](https://www.researchgate.net/profile/Gal-Oren/publication/365319736_Learning_to_Parallelize_Source_Code_via_OpenMP_with_Transformers/links/636ef0ab54eb5f547cc5aace/Learning-to-Parallelize-Source-Code-via-OpenMP-with-Transformers.pdf)
- **Codex Hacks HackerRank: Memorization Issues and a Framework for Code Synthesis Evaluation** (2022), arxiv, Karmakar, Anjan, et al. [[pdf]](https://arxiv.org/pdf/2212.02684)
- **BCGen: a comment generation method for bytecode** (2022), ASE, Huang, Yuan, et al.
- **Explaining Software Bugs Leveraging Code Structures in Neural Machine Translation** (2022), arxiv, Mahbub, Parvez, et al. [[pdf]](https://arxiv.org/pdf/2212.04584)
- **Neural Language Models for Code Quality Identification** (2022), arxiv, Sengamedu, S., et al.
- **Detecting Security Patches in Java Projects Using NLP Technology** (2022), ICNLSP'22, Stefanoni, Andrea, et al. [[pdf]](https://re.public.polimi.it/bitstream/11311/1223328/1/paper_sgj%2B.pdf)
- **Program Merge Conflict Resolution via Neural Transformers** (2022), FSE'22, Svyatkovskiy, Alexey, et al.

- **Teaching Algorithmic Reasoning via In-context Learning** (2022), arxiv, Zhou, Hattie, et al [[pdf]](https://arxiv.org/pdf/2211.09066)
- **Improved Evaluation of Automatic Source Code Summarisation** (2022), arxiv, Phillips, Jesse, et al. [[pdf]](https://www.lancaster.ac.uk/~elhaj/docs/gem2022.pdf)
- **Towards Generalizable and Robust Text-to-SQL Parsing** (2022), arxiv, Gao, Chang, et al. [[pdf]](https://arxiv.org/pdf/2210.12674)
- **CodeEditor: Learning to Edit Source Code with Pre-trained Models** (2022), arxiv, Li, Jia, et al. [[pdf]](https://arxiv.org/pdf/2210.17040)
- **Poison Attack and Defense on Deep Source Code Processing Models** (2022), arxiv, Li, Jia, et al. [[pdf]](https://arxiv.org/pdf/2210.17029)
- **NEUDEP: Neural Binary Memory Dependence Analysis** (2022), FSE'22, Pei, Kexin, et al. [[pdf]](https://arxiv.org/pdf/2210.02853)
- **Novice Type Error Diagnosis with Natural Language Models** (2022), arxiv, Geng, Chuqin, et al. [[pdf]](https://arxiv.org/pdf/2210.03682)
- **CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure** (2022), arxiv, Chen, Nuo, et al. [[pdf]](https://arxiv.org/pdf/2210.04633)
- **Using Large Language Models to Enhance Programming Error Messages** (2022), SIGCSE'22, Leinonen, J., et al. [[pdf]](https://arxiv.org/pdf/2210.11630)
- **Automatic Code Documentation Generation Using GPT-3** (2022), ASE'22, Khan, J. Y., and G. Uddin. [[pdf]](https://arxiv.org/pdf/2209.02235)
- **So Much in So Little: Creating Lightweight Embeddings of Python Libraries** (2022), arxiv, Golubev, Yaroslav, et al. [[pdf]](https://arxiv.org/pdf/2209.03507)
- **Code Compliance Assessment as a Learning Problem** (2022), arxiv, Sawant, N., and S. H. Sengamedu [[pdf]](https://arxiv.org/pdf/2209.04602.pdf)
- **Learning-based Identification of Coding Best Practices from Software Documentation** (2022), ICSME'22, Sawant, N., and S. H. Sengamedu [[pdf]](https://assets.amazon.science/9d/1f/8610ceff42eeb27e813fb580d447/learning-based-identification-of-coding-best-practices-from-software-documentation.pdf)
- **Learning to Answer Semantic Queries over Code** (2022), arxiv, Sahu, Surya Prakash, et al. [[pdf]](https://arxiv.org/pdf/2209.08372)
- **XFL: Naming Functions in Binaries with Extreme Multi-label Learning** (2022), arxiv, Patrick-Evans, J., et al. [[pdf]](http://static.aixpaper.com/pdf/8/63/arxiv.2107.13404.pdf)
- **SymLM: Predicting Function Names in Stripped Binaries via Context-Sensitive Execution-Aware Code Embeddings** (2022), Jin, Xin, et al. [[pdf]](https://web.cse.ohio-state.edu/~lin.3021/file/CCS22d.pdf)
- **Out of the BLEU: how should we assess quality of the Code Generation models?** (2022), arxiv, Evtikhiev, Mikhail, et al. [[pdf]](https://arxiv.org/pdf/2208.03133)
- **Compressing Pre-trained Models of Code into 3 MB** (2022), arxiv, Shi, Jieke, et al. [[pdf]](https://arxiv.org/pdf/2208.07120)
- **A Scalable and Extensible Approach to Benchmarking NL2Code for 18 Programming Languages** (2022), arxiv, Cassano, Federico, et al. [[pdf]](https://arxiv.org/pdf/2208.08227)
- **AUGER: Automatically Generating Review Comments with Pre-training Models** (2022), FSE'22, Li, Lingwei, et al. [[pdf]](https://arxiv.org/pdf/2208.08014)
- **Overwatch: Learning Patterns in Code Edit Sequences** (2022), arxiv, Zhang, Yuhao, et al. [[pdf]](https://arxiv.org/pdf/2207.12456)
- **Proton: Probing Schema Linking Information from Pre-trained Language Models for Text-to-SQL Parsing** (2022), KDD'22, Wang, Lihan, et al. [[pdf]](https://arxiv.org/pdf/2206.14017.pdf)
- **DIRE and its Data: Neural Decompiled Variable Renamings with Respect to Software Class** (2022), TOSEM, Dramko, Luke, et al. 
- **Making Python Code Idiomatic by Automatic Refactoring Non-Idiomatic Python Code with Pythonic Idioms** (2022), arxiv, Zhang, Zejun, et al. [[pdf]](https://arxiv.org/pdf/2207.05613)
- **DeepPERF: A Deep Learning-Based Approach For Improving Software Performance** (2022), arxiv, Garg, Spandan, et al. [[pdf]](https://arxiv.org/pdf/2206.13619)
- **CrystalBLEU: Precisely and Efficiently Measuring the Similarity of Code** (2022), ICSE ’22 Companion, Eghbali, Aryaz, and Michael, P. [[pdf]](https://software-lab.org/publications/icse2022_poster_CrystalBLEU.pdf)
- **Impact of Evaluation Methodologies on Code Summarization** (2022), ACL, Nie, Pengyu, et al. [[pdf]](https://cozy.ece.utexas.edu/~pynie/p/NieETAL22EvalMethodologies.pdf)

# PhD Theses

- **Deep Language Models for Software Testing and Optimisation** (2023), Foivos Tsimpourlas [[pdf]](https://era.ed.ac.uk/bitstream/handle/1842/40677/Tsimpourlas2023.pdf?sequence=1&isAllowed=y)
- **Improving Programming Productivity with Statistical Models** (2022), Tam Nguyen [[pdf]](https://etd.auburn.edu/bitstream/handle/10415/8152/Dissertation_TamNguyen.pdf)
- **Learning to Find Bugs in Programs and their Documentation** (2021), Andrew Habib [[pdf](https://tuprints.ulb.tu-darmstadt.de/17377/)]
- **Machine Learning and the Science of Software Engineering** (2020), Vincent Hellendoorn
- **Deep learning for compilers** (2020), Christopher E. Cummins [[pdf]](https://era.ed.ac.uk/handle/1842/36866)
- **Deep Learning in Software Engineering** (2020), Cody Watson [[pdf]](http://www.cs.wm.edu/~denys/pubs/dissertations/Watson_Dissertation.pdf)
- **Learning Code Transformations via Neural Machine Translation** (2019), Michele Tufano [[pdf]](https://scholarworks.wm.edu/cgi/viewcontent.cgi?article=6811&context=etd)
- **Improving the Usability of Static Analysis Tools Using Machine Learning** (2019), Ugur Koc [[pdf]](https://drum.lib.umd.edu/bitstream/handle/1903/25464/Koc_umd_0117E_20465.pdf?sequence=2&isAllowed=y)
- **Learning Natural Coding Conventions** (2016), Miltiadis Allamanis [[pdf]](https://miltos.allamanis.com/publicationfiles/allamanis2017dissertation/allamanis2017dissertation.pdf)

# Talks
- **Machine Learning for Software Engineering: AMA**, MSR 2020 [[video]](https://youtu.be/cphPhsehw2M)
- **Understanding Source Code with Deep Learning**, FOSDEM 2019 [[video]](http://bofh.nikhef.nl/events/FOSDEM/2019/H.2213/ml_on_code_understanding.webm)

# Datasets

- [StudentEval](https://arxiv.org/pdf/2306.04556.pdf) - A Benchmark of Student-Written Prompts for Large Language Models of Code
- [PySecDB](https://arxiv.org/pdf/2307.11853.pdf) - Exploring Security Commits in Python
- [DiverseVul](https://arxiv.org/pdf/2304.00409.pdf) - A Vulnerable Source Code Dataset for Deep Learning Based Vulnerability Detection
- [RunBugRun](https://arxiv.org/pdf/2304.01102.pdf) - An Executable Dataset for Automated Program Repair
- [ODEX](https://arxiv.org/pdf/2212.10481.pdf) - An open-domain execution-based natural language (NL) to code generation dataset
- [PI-Link](https://www.kaggle.com/datasets/zakareaalshara/android-closed-issues-20110101-20210101-clean) - A Ground-Truth Dataset of Links Between Pull-Requests and Issues in GitHub
- [ml-Codesmell](https://figshare.com/articles/conference_contribution/ml-Codesmell/21343299/2) - A code smell prediction dataset for machine
learning approaches
- [JEMMA](https://github.com/giganticode/jemma/) - An Extensible Java Dataset for ML4Code
Applications
- [CS1QA](https://aclanthology.org/2022.naacl-main.148.pdf) (2022) -  A Dataset for Assisting Code-based Question Answering in an Introductory Programming Course
- [XLCoST](https://arxiv.org/pdf/2206.08474) (2022) - A Benchmark Dataset for Cross-lingual Code Intelligence
- [CodeS](https://arxiv.org/pdf/2206.05480.pdf) (2022) - CodeS: A Distribution Shift Benchmark Dataset for
Source Code Learning
- [methods2test](https://github.com/microsoft/methods2test) (2022) - A supervised dataset consisting of Test Cases and their corresponding Focal Methods from a set of Java repositories
- [ManyTypes4TypeScript](https://www.kevinrjesse.com/pdfs/ManyTypes4TypeScript.pdf) (2022) - Type prediction dataset for TypeScript 
- [HumanEval](https://github.com/openai/human-eval) - Program synthesis from code comments
- [GitHub Code](https://huggingface.co/datasets/lvwerra/github-code) (2022) - 115M LoC in 32 programming languages
- [D2A](https://arxiv.org/pdf/2102.07995.pdf) (2021) - A Dataset Built for AI-Based Vulnerability Detection Methods Using Differential Analysis
- [CodeXGLUE](https://huggingface.co/datasets?search=code_x_glue) (2021)
- [ogbg-code2](https://arxiv.org/pdf/2005.00687.pdf) (2021)
- [ManyTypes4Py](https://github.com/saltudelft/many-types-4-py-dataset) (2021) - Type prediction dataset for Python
- [CodeSearchNet](https://github.com/github/CodeSearchNet) (2020)
- [ManySStuBs4J](https://datashare.is.ed.ac.uk/handle/10283/3424) (2019)
- [150k Python Dataset](https://eth-sri.github.io/py150) (2016)
- [150k Javascript Dataset](https://eth-sri.github.io/js150) (2016)
- [GitHub Java Corpus](http://groups.inf.ed.ac.uk/cup/javaGithub/) (2013)

# Tools
## Source Code Analysis & Processing
- [COMEX](https://github.com/IBM/tree-sitter-codeviews) - A Tool for Generating Customized Source Code Representations
- [LibSA4Py](https://github.com/saltudelft/libsa4py) - LibSA4Py: Light-weight static analysis for extracting type hints and features
- [LibCST](https://github.com/Instagram/LibCST) - A concrete syntax tree parser library for Python
- [python-graphs](https://github.com/google-research/python-graphs) - A static analysis library for computing graph representations of Python programs suitable for use with graph neural networks.
- [Semantic](https://github.com/github/semantic) - Parsing, analyzing, and comparing source code across many languages
- [GraphGen4Code](https://wala.github.io/graph4code/) - A toolkit for creating code knowledge graphs based on WALA code analysis and extraction of documentation
- [Joern](https://github.com/joernio/joern) - Code analysis platform for C/C++/Java/Binary/Javascript/Python/Kotlin based on code property graphs
- [NaturalCC](https://xcodemind.github.io/papers/icse22_naturalcc_camera_submitted.pdf) - An Open-Source Toolkit for Code Intelligence
- [Scalpel](https://github.com/SMAT-Lab/Scalpel) - The Python Static Analysis Framework
- [WALA](https://github.com/wala/WALA) - T.J. Watson Libraries for Analysis, with frontends for Java, Android, and JavaScript
- [CodeGen](https://github.com/facebookresearch/CodeGen) - General toolkit to apply machine learning to code, from dataset creation to model training and evaluation (from Facebook AI Research)

## Machine Learning
- [CodeTF](https://arxiv.org/pdf/2306.00029.pdf) - One-stop Transformer Library for State-of-the-art Code LLM
- [SentencePiece](https://github.com/google/sentencepiece) - Unsupervised text tokenizer for Neural Network-based text generation
- [Hugging Face](https://github.com/huggingface/transformers) - Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.

## Code de-duplication
- [CD4Py](https://github.com/saltudelft/CD4Py) - Code De-Duplication for Python
- [Near-duplicate Source Code Detector](https://github.com/microsoft/near-duplicate-code-detector)

## Misc
- [Utilities by the DPU team of Microsoft](https://github.com/microsoft/dpu-utils)
- [A set of tools to work with Big Code](https://github.com/danhper/bigcode-tools) - Fetching GitHub repos, tokenizers, embeddings and etc
- [cloc](https://github.com/AlDanial/cloc) - Counts blank lines, comment lines, and physical lines of source code in many programming languages.

# Research Groups

- [Software Engineering Research Group (SERG)](https://se.ewi.tudelft.nl/), Delft University of Technology
- [Secure, Reliable, and Intelligent Systems Lab (SRI)](https://www.sri.inf.ethz.ch/), ETH Zurich
- [Software Lab (SOLA)](https://software-lab.org/index.html), University of Stuttgart
- [Machine Learning for the Analysis of Source Code Text (MAST)](http://mast-group.github.io/), Edinburgh University
- [Deep Program Understanding](https://www.microsoft.com/en-us/research/project/program/), Microsoft Research
- [DECAL (Davis Excellent/Eclectic/Extreme Computational Analytics Lab)](https://decallab.cs.ucdavis.edu/), UC Davis
- [JetBrains Research](https://research.jetbrains.org/groups/ml_methods/)
- [SMart software Analysis and Trustworthy computing Lab (SMAT)](https://smat-lab.github.io/), Monash University

# Venues
## Conferences
- **ICSE**, the International Conference on Software Engineering
- **FSE**, Symposium on the Foundations of Software Engineering
- **ASE**, the International Conference on Automated Software Engineering
- **MSR**, the Mining Software Repositories conference
- **ICPC**, the International Conference on Program Comprehension
- **ISSTA**, the International Symposium on Software Testing and Analysis
- **ICLR**, the International Conference on Learning Representations
- **ICML**, the International Conference on Machine Learning
- **ICMLA**, the International Conference on Machine Learning and Applications
- **AAAI**, the Association for the Advancement of Artificial
Intelligence 
- **ACL**, the Association for Computational Linguistics
- **OOPSLA**, the ACM Conference on Systems, Programming, Languages, and Applications

## Journals
- **TSE**, the IEEE Transactions on Software Engineering
- **TOSEM**, ACM Transactions on Software Engineering and Methodology
- **JSS**, Journal of Systems and Software
- **EMSE**, Journal of Empirical Software Engineering