# Projet-IA-Telecom-Paris

## Title
Correction du flou de défocalisation dans les images microscopiques à l’aide de modèles génératifs en deep learning

## Summary
Les artefacts de flou sont fréquents dans les images acquises par microscopie, en particulier dans les applications biologiques où la mise au point peut varier localement. Ces dégradations, souvent liées à des limitations physiques ou techniques de l’acquisition, compromettent la lisibilité et l’extraction d’informations pertinentes. Dans certains cas, le flou n’affecte qu’une zone restreinte du champ de vision, ce qui rend sa correction encore plus délicate.


Dans ce contexte, les modèles d’apprentissage profond offrent des perspectives intéressantes pour la correction automatique de flou, sans intervention humaine. Des architectures comme CycleGAN [4] permettent d’apprendre des transformations d’images entre deux domaines (flou ↔ net) sans nécessiter de paires d’images parfaitement alignées. Plus récemment, les modèles de diffusion probabilistes (DDPM) [6] ont émergé comme des approches particulièrement robustes et performantes pour la génération et la restauration d’images.


Problématique : Comment adapter des architectures de deep learning non supervisées, comme CycleGAN ou DDPM, pour corriger efficacement des flous locaux présents dans des images de microscopie ?


Pour y répondre, je divise mon étude en plusieurs étapes. Tout d’abord, j’évalue les architectures existantes. Je propose une analyse approfondie des résultats obtenus avec le modèle CycleGAN, avant de faire de tester le modèle DDPM. Enfin, je tente d’améliorer les résultats obtenus avec CycleGAN grâce au modèle COMI [8]. L’ensemble est analysé via des métriques quantitatives et une évaluation qualitative visuelle. Tout au long du projet, j’utiliserai le jeu de données de parasites Mendeley [1] issu d’une cohorte ouverte.


## Sources

### Datasets :
[1] Mendeley dataset: https://data.mendeley.com/datasets/m3jxgb54c9/4
[2] Facades dataset: https://www.kaggle.com/datasets/balraj98/facades-dataset
[3] Horse2zebra dataset: https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset

### CycleGAN :
[4] Zhu, Park, Isola & Efros (2020). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (version 7): https://arxiv.org/abs/1703.10593 
[5] aitorzip. PyTorch-CycleGAN : https://github.com/aitorzip/PyTorch-CycleGAN

### DDPM :
[6] Jonathan Ho, Ajay Jain, Pieter Abbeel. Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239 
[7] lucidrainns. DDPM (Denoising Diffusion Probabilistic Model): https://github.com/lucidrains/denoising-diffusion-pytorch 

### COMI :
[8] Chi Zhang, Hao Jiang, Weihuang Liu, Junyi Li, Shiming Tang, Mario Juhas, Yang Zhang. Correction of out-of-focus microscopic images by deep learning: https://www.sciencedirect.com/science/article/pii/S2001037022001192 
[9] jiangdat. COMI (Correction of Out-of-focus Microscopic Images by Deep Learning): https://github.com/jiangdat/COMI

### Autre :
[10] Prafulla Dhariwal, Alex Nichol. Diffusion Models Beat GANs on Image Synthesis: https://arxiv.org/abs/2105.05233

