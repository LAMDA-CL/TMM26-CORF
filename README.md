# Cross-Sample Relational Fusion: Unifying Domain Generalization and Class-Incremental Learning




<div align="center">
    <div>
        <a href='https://www.lamda.nju.edu.cn/wenzh' target='_blank'>Zhen-Hao Xie</a>&emsp;
        <a href='http://www.lamda.nju.edu.cn/wangy' target='_blank'>Yan Wang</a>&emsp;
        <a href='http://www.lamda.nju.edu.cn/sunh' target='_blank'>Hao Sun</a>&emsp;
        <a href='http://www.lamda.nju.edu.cn/yehj' target='_blank'>Han-Jia Ye</a>&emsp;
        <a href='http://www.lamda.nju.edu.cn/zhandc' target='_blank'>De-Chuan Zhan</a>&emsp;
        <a href='http://www.lamda.nju.edu.cn/zhoudw' target='_blank'>Da-Wei Zhou</a>&emsp;
    </div>
    <div>
    State Key Laboratory for Novel Software Technology, Nanjing University
    </div>
</div>

<div align="center">
    
  <a href="https://tex.nju.edu.cn/share/92c1ef8b570543c09e563daf5c4762cc">
    <img src="https://img.shields.io/badge/Paper-red" alt="arXiv">
  </a>

</div>

The code repository for "Cross-Sample Relational Fusion: Unifying Domain Generalization and Class-Incremental Learning" in PyTorch.  If you use any content of this repo for your work, please cite the following bib entry: 

```bibtex
@article{Xie26CORF,
  author       = {Zhen-Hao Xie and Yan Wang and Hao Sun and Han-Jia Ye and De-Chuan Zhan and Da-Wei Zhou},
  title        = {Cross-Sample Relational Fusion: Unifying Domain Generalization and Class-Incremental Learning},
  journal      = {{IEEE} Trans. Multim.},
  year         = {2026},
}
```



# 📢 **Updates**

[05/2026] Accepted to [TMM 2026](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6046).

[05/2026] Code has been released.

[05/2026] [arXiv]() paper has been released.


# 📝 Introduction
Class-Incremental Learning (CIL) requires a learning system to learn new classes while retaining previously learned knowledge. However, in real-world scenarios like autonomous driving, a system trained on urban roads in sunny weather may later need to operate in rural or highway environments under different traffic patterns and weather conditions. This requires the model not only to overcome catastrophic forgetting, but also to effectively handle domain shifts. In this paper, we propose CrOss-sample Relational Fusion (CORF), a unified framework to address domain shift and catastrophic forgetting simultaneously. To enhance generalizability, we perform selective refinement of training samples by leveraging spatial contribution maps to highlight semantically informative regions. Furthermore, we incorporate predictive confidence to adaptively weigh samples, thereby facilitating the learning of domain-agnostic representations. To alleviate forgetting, we propose a cascaded distillation framework that captures cross-sample relational dependencies across multiple feature hierarchies, enabling the multi-grained transfer of knowledge from previous tasks. CORF can be seamlessly integrated into existing CIL algorithms to enhance their generalizability, achieving competitive performance across various benchmark datasets.
<div align="center">
<img src="resources/overview.png" width="95%">
</div>

## 🔧 Requirements

**Environment**

1. [torch 2.0.1](https://github.com/pytorch/pytorch)

2. [torchvision 0.15.2](https://github.com/pytorch/vision)

3. [tqdm](https://github.com/tqdm/tqdm)
   
4. [numpy](https://github.com/numpy/numpy)
   
5. [scipy](https://github.com/scipy/scipy)
   
6. [quadprog](https://github.com/quadprog/quadprog)
 
7. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)
   
8. [easydict](https://github.com/makinacorpus/easydict)


## 🔎 Datasets

We have implemented the pre-processing datasets as follows:

 - **OfficeHome**: Official website: [link](https://www.hemanthdv.org/officeHomeDataset.html) or Hugging Face: [link](https://huggingface.co/datasets/flwrlabs/office-home)
- **PACS**: Official benchmark website: [link](https://domaingeneralization.github.io/) or Hugging Face: [link](https://huggingface.co/datasets/flwrlabs/pacs)
- **DomainNet**: Official website: [link](https://ai.bu.edu/DomainNet/) or TensorFlow Datasets: [link](https://www.tensorflow.org/datasets/catalog/domainnet)

## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. All main experiments from the paper are already provided in the `exps` folder, you can simply execute them to reproduce the results found in the `logs` folder.

```
python main.py --config ./exps/[config_name]/[dataset_name]/[increment]/[target_name].json
```

## 🎈 Acknowledgement

This repo is based on [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey) and [PyCIL](https://github.com/G-U-N/PyCIL). 

## 💭 Correspondence

If you have any questions, please  contact me via [email](mailto:zhoudw@lamda.nju.edu.cn).


