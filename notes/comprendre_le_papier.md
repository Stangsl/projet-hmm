# Comprendre le papier DCPS (version simple)

**Papier :** Y. Janati, B. Moufad, A. Durmus, E. Moulines, J. Olsson (2024),
*Divide-and-Conquer Posterior Sampling for Denoising Diffusion Priors*, NeurIPS 2024.
Les numéros d'équations ci-dessous renvoient au papier.

---

## 1. Le problème

On veut retrouver un signal inconnu $x$ (une image) à partir d'une observation $y$ dégradée et bruitée.
Approche bayésienne :

- **prior** $p_0(x)$ : à quoi ressemble une image « normale » ;
- **vraisemblance** $g_0(x) = p(y \mid x)$ : comment $y$ est produit à partir de $x$ ;
- **posterior** $\pi_0(x) = g_0(x)\,p_0(x) / Z$ (éq. 2.3) : ce qu'on veut **échantillonner**.

Exemples de vraisemblances dans le papier :

| Problème | Vraisemblance |
|---|---|
| linéaire gaussien (super-résolution, inpainting…) | $y = Ax + \sigma_y \varepsilon$ |
| **débruitage de Poisson** (notre cas) | $y_j \sim \text{Poisson}(\lambda (Ax)_j)$, avec $A = I$ pour le débruitage |
| JPEG | $y$ = image compressée |

Pour Poisson, le papier (comme DPS, Chung et al. 2023) remplace la vraie vraisemblance par son
approximation normale : $g_0(x) = \prod_j \mathcal N(y_j;\ \lambda (Ax)_j,\ y_j)$ (Section 3).

## 2. Le prior : un modèle de diffusion

- **Bruitage (forward)** : $X_k = \sqrt{\alpha_k}\,X_0 + \sqrt{1-\alpha_k}\,\varepsilon$, avec $\alpha_0 = 1$ et $\alpha_n \approx 0$.
- **Débruitage (backward)** : on part de $X_n \sim \mathcal N(0,I)$ et on tire $X_{n-1}, \dots, X_0$ avec les
  transitions DDPM $p_{k|k+1}$ (éq. 2.1–2.2).
- Ces transitions utilisent le **débruiteur** $\hat x_{0|k}(x_k) \approx \mathbb E[X_0 \mid X_k = x_k]$, un
  **réseau de neurones pré-entraîné** (on ne l'entraîne pas nous-mêmes).

On sait donc **simuler** selon le prior, mais on ne sait **pas calculer sa densité** $p_0(x)$. On ne peut donc pas
utiliser directement un MCMC classique (Metropolis–Hastings) sur le posterior.

## 3. Le lien avec le cours : un modèle de Feynman–Kac

Le modèle génératif est une **chaîne de Markov** qui remonte le temps, de $k = n$ à $k = 0$.
Le posterior est la loi de $X_0$ quand on pondère la trajectoire par $g_0(X_0)$ à la fin.

**Astuce (éq. 2.4)** : on peut ajouter des potentiels intermédiaires $g_k$ sans changer la loi cible,
parce que le produit des ratios $g_k / g_{k+1}$ se télescope :

$$
\pi_0(x_0) = \frac1Z\int g_n(x_n)\,p_n(x_n)\prod_{k=0}^{n-1}\frac{g_k(x_k)}{g_{k+1}(x_{k+1})}\,p_{k|k+1}(x_k\mid x_{k+1})\,\mathrm dx_{1:n}.
$$

C'est un **modèle de Feynman–Kac** : on peut le simuler avec un **filtre particulaire**. Le choix des $g_k$
détermine la qualité du filtre (comme le choix de la proposition dans le cours).

**Le potentiel optimal** est
$g^\star_k(x_k) = \int g_0(x_0)\,p_{0|k}(x_0 \mid x_k)\,\mathrm dx_0 = \mathbb E[g_0(X_0) \mid X_k = x_k]$
(« quelle est la vraisemblance de $y$ si je suis en $x_k$ maintenant ? »).
Avec lui, les transitions tordues $\pi_{k|k+1} \propto g^\star_k\,p_{k|k+1}$ (éq. 2.5) donnent **exactement**
le posterior, sans aucun poids. C'est l'analogue de la **fonction backward** du lissage dans un HMM
(vraisemblance des observations futures sachant l'état courant).
**Problème :** $g^\star_k$ n'est pas calculable, car c'est une espérance sous $p_{0|k}$.

## 4. Les approximations existantes

- **DPS** (Chung et al., 2023) : remplace $p_{0|k}(\cdot\mid x_k)$ par une masse de Dirac en $\hat x_{0|k}(x_k)$, donc
  $g^\star_k(x_k) \approx g_0(\hat x_{0|k}(x_k))$, puis fait un pas de gradient (éq. 2.7). C'est simple mais biaisé, et
  très grossier quand $k$ est grand (beaucoup de bruit).
- **ΠGDM** : remplace la Dirac par une gaussienne.
- **Méthodes SMC** (MCGdiff, TDS) : utilisent des particules pondérées. Elles sont asymptotiquement exactes quand $N \to \infty$,
  mais les poids dégénèrent en très grande dimension (Bickel et al., 2008).

## 5. L'idée de DCPS

1. **Sauter moins loin, c'est plus facile** (Proposition 3.1). Approcher la loi de $X_\ell$ sachant $X_k$
   (un saut court, $\ell$ proche de $k$) avec le pont gaussien fait une erreur plus petite qu'approcher celle de $X_0$
   sachant $X_k$. Le facteur est $\sqrt{\alpha_\ell}(1-\alpha_k/\alpha_\ell)/(1-\alpha_k) < 1$.
2. **Diviser pour régner.** On découpe $[0, n]$ en $L$ blocs ($0 = k_0 < k_1 < \dots < k_L = n$, avec $L = 3$ dans le papier)
   et on vise une suite de **posteriors intermédiaires** $\pi_{k_\ell} \propto g_{k_\ell}\,p_{k_\ell}$ (éq. 3.3), avec des
   $g_{k_\ell}$ choisis à la main. Pour Poisson (éq. 3.11) :
   $g_{k_\ell}(x) = \prod_j \mathcal N(\sqrt{\alpha_{k_\ell}}\,y_j;\ \lambda(Ax)_j,\ \sqrt{\alpha_{k_\ell}}\,y_j)$.
3. **Dans chaque bloc** (Algorithme 1, Annexe A.2) :
   - quelques pas de **Langevin** ($M$ pas) pour passer d'un échantillon de $\pi_{k_{\ell+1}}$ à la bonne loi
     de départ du bloc ;
   - puis on descend le bloc pas à pas. Chaque transition (intractable) est approchée par une **gaussienne
     diagonale ajustée par inférence variationnelle**, en $K$ pas de gradient (éq. 3.8).
4. **Pas de particules, pas de poids.** Chaque échantillon est produit indépendamment. La méthode passe
   donc à l'échelle des images, mais elle reste **biaisée** : elle ne converge pas vers le vrai posterior
   quand on augmente le nombre d'échantillons.

## 6. DCPS vs MCGdiff (Cardoso et al., ICLR 2024)

D'après la description qu'en donne le papier DCPS (Annexe B). À vérifier en lisant MCGdiff directement.

| | MCGdiff | DCPS |
|---|---|---|
| Outil | filtre particulaire (auxiliaire) sur le modèle de FK (2.4) | Langevin + inférence variationnelle, sans particules |
| Potentiels | $g_k(x_k) = \mathcal N(\sqrt{\alpha_k}\,y;\ Ax_k,\ (1-\alpha_k) I)$ (cas $\sigma_y = 0$) | posteriors intermédiaires $g_{k_\ell}$, avec la variance $\sigma_y^2$ |
| Garantie | exact quand $N \to \infty$ | biaisé |
| Grande dimension | les particules s'effondrent (tous les échantillons se ressemblent) | passe à l'échelle |
| Problèmes | linéaires gaussiens | linéaires, **Poisson**, JPEG |
| Jouet mélange gaussien (Table 1) | le meilleur | proche de MCGdiff avec $M = 500$ pas de Langevin |

## 7. Hyperparamètres utilisés dans le papier (Annexe C)

- DCPS : $L = 3$ blocs, $K = 2$ pas de gradient, pas $\zeta = 1$ ;
  $M = 50$ ou $500$ pas de Langevin (jouet), $M = 5$ (images) ;
  pas de Langevin $\gamma = 10^{-2}$ (jouet), $10^{-3}$ (images).
- Jouet : 25 gaussiennes, moyennes $(8i, 8j)$, poids Dirichlet ; $d_y = 1$, $A$ gaussienne,
  $\sigma_y \sim \mathcal U[0,1]$ ; métrique : sliced Wasserstein entre 2000 échantillons et 2000 tirages du vrai
  posterior, 30 répétitions.
- Poisson (images) : $\lambda = 0{,}1$ ; métrique LPIPS ; comparaison avec DPS (Table 6).
