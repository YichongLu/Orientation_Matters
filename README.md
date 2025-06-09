# Orientation Matters
Official repository for the paper "Orientation Matters: Making 3D Generative Models Orientation-Aligned"

![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='http://fuxiao0719.github.io/projects/robomaster'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://arxiv.org/pdf/2506.01943'><img src='https://img.shields.io/badge/arXiv-2506.01943-b31b1b.svg'></a> &nbsp;
 <a href='https://huggingface.co/KwaiVGI/RoboMaster'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> &nbsp;


<div align="center">
<img src='assets/teaser.png'></img>
</div>

Humans intuitively perceive object shape and orientation from a single image, guided by strong priors about canonical poses. However, existing 3D generative models often produce misaligned results due to inconsistent training data, limiting their usability in downstream tasks. To address this gap, we introduce the task of orientation-aligned 3D object generation: producing 3D objects from single images with consistent orientations across categories. To facilitate this, we construct Objaverse-OA, a dataset of 14,832 orientation-aligned 3D models spanning 1,008 categories. Leveraging Objaverse-OA, we fine-tune two representative 3D generative models based on multi-view diffusion and 3D variational autoencoder frameworks to produce aligned objects that generalize well to unseen objects across various categories. Experimental results demonstrate the superiority of our method over post-hoc alignment approaches. Furthermore, we showcase downstream applications enabled by our aligned object generation, including zero-shot model-free object orientation estimation via analysis-by-synthesis and efficient arrow-based object manipulation.

### 📝 TODO List
- [ ] Add Huggingface demos.
- [ ] Release Objaverse-OA dataset.
- [ ] Add training codes.
- [ ] Add evaluation codes.
- [ ] Add codes of our augmented reality applications.
- [ ] Add Blender add-on of our arrow-based manipulation.





