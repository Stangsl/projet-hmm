# projet-hmm — Échantillonner un posterior avec un prior de diffusion

Projet ENSAE, cours *Modèles de Markov cachés et Monte Carlo séquentiel* (N. Chopin).

**Papier étudié :** Janati, Moufad, Durmus, Moulines, Olsson (2024),
[*Divide-and-Conquer Posterior Sampling for Denoising Diffusion Priors*](https://proceedings.neurips.cc/paper_files/paper/2024/file/b0ae046e198a5e43141519868a959c74-Paper-Conference.pdf), NeurIPS 2024.

**Objectif :** reproduire une partie des expériences numériques, en numpy (et avec le paquet
[`particles`](https://github.com/nchopin/particles)), sur des problèmes simples :
d'abord un jouet en 2D où le vrai posterior est connu, puis le **débruitage de Poisson** sur de petites images,
avec un modèle de diffusion **pré-entraîné**.

Une explication simple du papier se trouve dans [notes/comprendre_le_papier.md](notes/comprendre_le_papier.md).

## Avancement

- [x] **01 — Jouet 2D** ([notebook](notebooks/01_premiers_pas_jouet_2D.ipynb)) : prior mélange gaussien, bruitage,
  DDPM avec débruiteur exact, problème inverse et vrai posterior, échantillonnage préférentiel et dégénérescence des poids.
- [ ] **02 — Filtre particulaire** sur le modèle de Feynman–Kac (éq. 2.4), avec `particles`, sur le jouet 2D.
- [ ] **03 — DPS** (méthode de référence) sur le jouet.
- [ ] **04 — DCPS** sur le jouet, comparé au vrai posterior (10 répétitions, boxplots).
- [ ] **05 — Poisson** : d'abord sur le jouet (intensités positives, posterior calculable sur une grille 2D),
  puis sur de petites images avec un modèle pré-entraîné (à choisir).
- [ ] **06 — Bonus** : comparaison avec MCGdiff (Cardoso et al., 2024).
- [ ] Slides + notebook final.


## Organisation

```
notebooks/   les notebooks, numérotés dans l'ordre
notes/       explications écrites (papier, calculs)
papiers/     PDF des articles (non versionnés, voir .gitignore)
```

## Installation

```bash
pip install -r requirements.txt
jupyter lab
```

## Sources

- Papier DCPS et son code officiel : https://github.com/Badr-MOUFAD/dcps. On ne le reprend pas : on réécrit en numpy.
  Toute partie reprise sera citée ici.
- Cardoso, Janati, Le Corff, Moulines (2024), *Monte Carlo guided denoising diffusion models for Bayesian linear inverse problems*, ICLR (MCGdiff).
- Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models* (planning des $\beta_t$).
- Chopin & Papaspiliopoulos (2020), *An Introduction to Sequential Monte Carlo*, Springer, et le paquet `particles`.
