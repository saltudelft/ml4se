# Machine Learning for Software Engineering
![GitHub last commit](https://img.shields.io/github/last-commit/saltudelft/ml4se)

This repository contains a curated list of papers, datasets, and tools that are devoted to research on Machine Learning for Software Engineering. The papers are organized into popular research areas so that researchers can find recent papers and state-of-the-art approaches easily.

Please feel free to send a pull request to add papers and relevant content that are not listed here.

## Content
- [Papers](#papers)
    - [Type Inference](#type-inference)
    - [Code Completion](#code-completion)
    - [Code Generation](#code-generation)
    - [Code Summarization](#code-summarization)
    - [Code Embeddings/Representation](#code-embeddingsrepresentation)
    - [Code Changes](#code-changes)
    - [Bug/Vulnerability Detection](#bugvulnerability-detection)
    - [Source Code Modeling](#source-code-modeling)
    - [Program Repair](#program-repair)
    - [Program Translation](#program-translation)
    - [Code Clone Detection](#code-clone-detection)
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
- **CodeFill: Multi-token Code Completion by Jointly Learning from Structure and Naming Sequences**, ICSE'22, Izadi, Maliheh, et al. [[pdf]](https://arxiv.org/pdf/2202.06689.pdf) [[code]](https://github.com/saltudelft/codefill)
- **Code Completion by Modeling Flattened Abstract Syntax Trees as Graphs** (2021), AAAI'21, Wang, Yanlin, et al. [[pdf]](https://www.aaai.org/AAAI21Papers/AAAI-1654.WangY.pdf)
- **Code Prediction by Feeding Trees to Transformers** (2021), ICSE'21, Kim, Seohyun, et al. [[pdf]](https://arxiv.org/pdf/2003.13848)
- **Fast and Memory-Efficient Neural Code Completion** (2020), arxiv 2020, Svyatkovskoy, Alexey, et al. [[pdf]](https://arxiv.org/pdf/2004.13651)
- **Pythia: AI-assisted Code Completion System** (2019), KDD'19, Svyatkovskiy, Alexey, et al. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3292500.3330699)
- **Code Completion with Neural Attention and Pointer Networks** (2018), arxiv 2018, Li, Jian, et al. [[pdf]](https://arxiv.org/pdf/1711.09573)

## Code Generation
- **Compilable Neural Code Generation with Compiler Feedback** (2022), arxiv 2022, Wang, Xin, et al. [[pdf]](https://arxiv.org/pdf/2203.05132.pdf)
- **Predictive Synthesis of API-Centric Code** (2022), arxiv 2022, Nam, Daye, et al. [[pdf]](https://arxiv.org/pdf/2201.03758.pdf) 
- **Evaluating Large Language Models Trained on Code** (2021), arxiv 2021, Chen, Mark, et al. [[pdf]](https://arxiv.org/pdf/2107.03374.pdf?ref=https://githubhelp.com) [[code]](https://github.com/openai/human-eval)
- **Code Prediction by Feeding Trees to Transformers** (2020), arxiv 2020, Kim, Seohyun, et al. [[pdf]](https://arxiv.org/pdf/2003.13848)
- **TreeGen: A Tree-Based Transformer Architecture for Code Generation** (2019), arxiv 2019, Zhu, Qihao, et al. [[pdf]](https://arxiv.org/abs/1911.09983)
- **A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation** (2017), arxiv 2017, Barone, Antonio V. M., et al. [[pdf]](https://arxiv.org/pdf/1707.02275)

## Code Summarization
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
- **TreeCaps: Tree-Based Capsule Networks for Source Code Processing** (2021), AAAI'21, Bui, Nghi DQ, et al. [[pdf]](https://www.aaai.org/AAAI21Papers/AAAI-9746.BuiNDQ.pdf) [[code]](https://github.com/bdqnghi/treecaps)
- **Language-Agnostic Representation Learning of Source Code from Structure and Context** (2021), ICLR'21, Zügner, Daniel, et al. [[pdf]](https://arxiv.org/pdf/2103.11318)
- **Learning and Evaluating Contextual Embedding of Source Code** (2020), ICML 2020, Kanade, Aditya, et al. [[pdf]](http://proceedings.mlr.press/v119/kanade20a/kanade20a.pdf)
- **Learning Semantic Program Embeddings with Graph Interval Neural Network** (2020), OOPSLA'20, Wang, Yu, et al.
- **Contrastive Code Representation Learning** (2020), arxiv 2020, Jain, Paras, et al. [[pdf]](https://arxiv.org/pdf/2007.04973.pdf)
- **Codebert: A Pre-trained Model for Programming and Natural Languages** (2020), arxiv 2020, Feng, Zhangyin, et al. [[pdf]](https://arxiv.org/pdf/2002.08155)
- **SCELMo: Source Code Embeddings from Language Models** (2020), arxiv 2020, Karampatsis, Rafael-Michael, et al. [[pdf]](https://arxiv.org/pdf/2004.13214)
- **code2vec: Learning Distributed Representations of Code** (2019), ACM POPL 2019, Alon, Uri, et al. [[pdf]](http://www.cs.technion.ac.il/~mbs/publications/code2vec-popl19.pdf)
- **COSET: A Benchmark for Evaluating Neural Program Embeddings** (2019), arxiv 2019, Wang, Ke, et al. [[pdf]](https://arxiv.org/pdf/1905.11445)
- **A Literature Study of Embeddings on Source Code** (2019), arxiv 2019, Chen, Zimin, et al. [[pdf]](https://arxiv.org/pdf/1904.03061)
- **code2seq: Generating Sequences from Structured Representations of Code** (2018), arxiv 2018, Alon, Uri, et al. [[pdf]](https://arxiv.org/pdf/1808.01400)
- **Neural Code Comprehension: A Learnable Representation of Code Semantics** (2018), NIPS 2018, Ben-Nun, Tal, et al. [[pdf]](http://papers.nips.cc/paper/7617-neural-code-comprehension-a-learnable-representation-of-code-semantics.pdf)
- **Convolutional Neural Networks over Tree Structures for Programming Language Processing** (2016), AAAI'16, Mou, Lili, et al. [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11775/11735)

## Code Changes

- **Commit2Vec: Learning Distributed Representations of Code Changes** (2021), SN Computer Science, Lozoya, Rocío Cabrera, et al.
- **CODIT: Code Editing with Tree-Based Neural Models** (2020), TSE 2020, Chakraborty, Saikat, et al.
- **On learning meaningful code changes via neural machine translation** (2019), ICSE 2019, Tufano, Michele, et al.

## Bug/Vulnerability Detection
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
- **A Systematic Evaluation of Large Language Models of Code**, arxiv 2022, Xu, Frank F., et al. [[pdf]](https://arxiv.org/pdf/2202.13169)[[code]](https://github.com/VHellendoorn/Code-LMs)
- **Multilingual training for Software Engineering**, ICSE'22, Ahmed, Toufique, et al. [[pdf]](https://arxiv.org/pdf/2112.02043)
- **Unified Pre-training for Program Understanding and Generation**, NAACL'21, Ahmad, Wasi Uddin, et al. [[pdf]](https://arxiv.org/pdf/2103.06333)
- **Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code**, ICSE'20, Karampatsis, Rafael-Michael, et al.
- **Maybe Deep Neural Networks are the Best Choice for Modeling Source Code** (2019), arxiv 2019, Karampatsis, Rafael-Michael, et al. [[pdf]](https://arxiv.org/pdf/1903.05734)
- **Are Deep Neural Networks the Best Choice for Modeling Source Code?** (2017), FSE 2017, Hellendoorn, Vincent J., et al. [[pdf]](https://vhellendoorn.github.io/PDF/fse2017.pdf)

## Program Repair
- **DEAR: A Novel Deep Learning-based Approach for Automated Program Repair** (2022), ICSE'22, Li, Yi, et al. [[pdf]](https://arxiv.org/pdf/2205.01859)
- **TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer** (2021), ICML'21, Berabi, Berkay, et al. [[pdf]](http://proceedings.mlr.press/v139/berabi21a/berabi21a.pdf)
- **Neural Transfer Learning for Repairing Security Vulnerabilities in C Code** (2021), Chen, Zimin, et al. [[pdf]](https://arxiv.org/pdf/2104.08308)
- **Generating Bug-Fixes Using Pretrained Transformers** (2021), arxiv 2021, Drain, Dawn, et al. [[pdf]](https://arxiv.org/pdf/2104.07896)
- **Global Relational Models of Source Code** (2020), ICLR'20, Hellendoorn, Vincent J., et al. [[pdf]](https://openreview.net/pdf?id=B1lnbRNtwr)
- **Neural Program Repair by Jointly Learning to Localize and Repair** (2019), arxiv 2019, Vasic, Marko, et al. [[pdf]](https://arxiv.org/pdf/1904.01720)

## Program Translation
- **Multilingual Code Snippets Training for Program Translation** (2022), AAAI'22, Zhu, Ming, et al. [[pdf]](https://people.cs.vt.edu/~reddy/papers/AAAI22.pdf)
- **Semantics-Recovering Decompilation through Neural Machine Translation** (2021), arxiv 2021, Liang, Ruigang, et al. [[pdf]](https://arxiv.org/pdf/2112.15491.pdf)
- **Unsupervised Translation of Programming Languages** (2020), arxiv 2020, Lachaux, Marie-Anne et al. [[pdf]](https://arxiv.org/abs/2006.03511)

## Code Clone Detection
- **Cross-Language Source Code Clone Detection Using Deep Learning with InferCode** (2022), arxiv 2022, Yahya, M., and Kim, D., [[pdf]](https://arxiv.org/pdf/2205.04913)
- **funcGNN: A Graph Neural Network Approach to Program Similarity** (2020), ESEM'20, Nair, Aravind, et al. [[pdf]]()
- **Cross-Language Clone Detection by Learning Over Abstract Syntax Trees** (2019), MSR'19, Perez, Daniel, et al.
- **The Adverse Effects of Code Duplication in Machine Learning Models of Code** (2019), Onward! 2019, Allamanis, Miltiadis, [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3359591.3359735)

## Code Search
- **Cross-Domain Deep Code Search with Meta Learning** (2022), ICSE'22, Chai, Yitian, et al. [[pdf]](https://guxd.github.io/papers/cdcs.pdf)

## Empirical Studies
- **On the Transferability of Pre-trained Language Models for Low-Resource Programming Languages** (2022), ICPC'22, Chen, Fuxiang, et al. [[pdf]](https://arxiv.org/pdf/2204.09653.pdf)
- **Learning Program Semantics with Code Representations: An Empirical Study** (2022), SANER'22, Siow, Jing Kai, et al. [[pdf]](https://arxiv.org/pdf/2203.11790)[[code]](https://github.com/jingkai92/learning-program-representation)
- **Assessing Generalizability of CodeBERT** (2021), ICSME'21, Zhou, Xin, et al.
- **Thinking Like a Developer? Comparing the Attention of Humans with Neural Models of Code** (2021), ASE'21, Paltenghi, M. & Pradel, M.
- **An Empirical Study of Transformers for Source Code** (2021), FSE'21, Chirkova, N., & Troshin, S.
- **An Empirical Study on the Usage of Transformer Models for Code Completion** (2021), MSR'21, Ciniselli, Matteo, et al.

## Surveys
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
- **Impact of Evaluation Methodologies on Code Summarization** (2022), ACL, Nie, Pengyu, et al. [[pdf]](https://cozy.ece.utexas.edu/~pynie/p/NieETAL22EvalMethodologies.pdf)

# PhD Theses
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

- [ManyTypes4TypeScript](https://www.kevinrjesse.com/pdfs/ManyTypes4TypeScript.pdf) (2022) - Type prediction dataset for TypeScript 
- [HumanEval](https://github.com/openai/human-eval) - Program synthesis from code comments
- [GitHub Code](https://huggingface.co/datasets/lvwerra/github-code) (2022) - 115M LoC in 32 programming languages
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
- [LibSA4Py](https://github.com/saltudelft/libsa4py) - LibSA4Py: Light-weight static analysis for extracting type hints and features
- [LibCST](https://github.com/Instagram/LibCST) - A concrete syntax tree parser library for Python
- [python-graphs](https://github.com/google-research/python-graphs) - A static analysis library for computing graph representations of Python programs suitable for use with graph neural networks.
- [Semantic](https://github.com/github/semantic) - Parsing, analyzing, and comparing source code across many languages
- [GraphGen4Code](https://wala.github.io/graph4code/) - A toolkit for creating code knowledge graphs based on WALA code analysis and extraction of documentation
- [Joern](https://github.com/joernio/joern) - Code analysis platform for C/C++/Java/Binary/Javascript/Python/Kotlin based on code property graphs
- [NaturalCC](https://xcodemind.github.io/papers/icse22_naturalcc_camera_submitted.pdf) - An Open-Source Toolkit for Code Intelligence

## Machine Learning
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

# Venues

- **ICSE**, the International Conference on Software Engineering
- **FSE**, Symposium on the Foundations of Software Engineering
- **ASE**, the International Conference on Automated Software Engineering
- **MSR**, the Mining Software Repositories conference
- **ICPC**, the International Conference on Program Comprehension
- **ICLR**, the International Conference on Learning Representations
- **ICML**, the International Conference on Machine Learning
- **AAAI**, Association for the Advancement of Artificial
Intelligence 
- **OOPSLA**, the ACM Conference on Systems, Programming, Languages, and Applications
- **TSE**, the IEEE Transactions on Software Engineering