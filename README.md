# Orientation Matters
Official repository for the paper "Orientation Matters: Making 3D Generative Models Orientation-Aligned"

<div align="center">
 
![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='https://xdimlab.github.io/Orientation_Matters/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://arxiv.org/abs/2506.08640'><img src='https://img.shields.io/badge/arXiv-2506.08640-b31b1b.svg'></a> &nbsp;
 <a href='https://huggingface.co/spaces/Louischong/Trellis-OA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-red'></a> &nbsp;
 <a href='https://huggingface.co/Louischong/Trellis-OA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> &nbsp;
<img src='assets/teaser.png'></img>
</div>

Humans intuitively perceive object shape and orientation from a single image, guided by strong priors about canonical poses. However, existing 3D generative models often produce misaligned results due to inconsistent training data, limiting their usability in downstream tasks. To address this gap, we introduce the task of orientation-aligned 3D object generation: producing 3D objects from single images with consistent orientations across categories. To facilitate this, we construct Objaverse-OA, a dataset of 14,832 orientation-aligned 3D models spanning 1,008 categories. Leveraging Objaverse-OA, we fine-tune two representative 3D generative models based on multi-view diffusion and 3D variational autoencoder frameworks to produce aligned objects that generalize well to unseen objects across various categories. Experimental results demonstrate the superiority of our method over post-hoc alignment approaches. Furthermore, we showcase downstream applications enabled by our aligned object generation, including zero-shot model-free object orientation estimation via analysis-by-synthesis and efficient arrow-based object rotation manipulation in 3D simulation systems.

## 🔥 News
- 2025.6.25 We release the [Huggingface Demo](https://huggingface.co/spaces/Louischong/Trellis-OA)🤗 and pre-trained [model](https://huggingface.co/Louischong/Trellis-OA)🤗 of Trellis-OA. Have a try!

### 📝 TODO List
- [x] Add Huggingface demo and pre-trained checkpoint of Trellis-OA.
- [ ] Release the Objaverse-OA dataset.
- [ ] Add code of Wonder3D-OA.
- [ ] Add code of the orientation estimation method.
- [ ] Add code of the augmented reality application.
- [ ] Add the Blender add-on of our arrow-based manipulation.

## 🔗 Citation
If you find this work helpful, please consider citing:
```BibTeXw
@misc{lu2025orientationmatters,
      title={Orientation Matters: Making 3D Generative Models Orientation-Aligned}, 
      author={Yichong Lu and Yuzhuo Tian and Zijin Jiang and Yikun Zhao and Yuanbo Yang and Hao Ouyang and Haoji Hu and Huimin Yu and Yujun Shen and Yiyi Liao},
      year={2025},
      eprint={2506.08640},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.08640}, 
}
```

