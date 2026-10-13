# Sleep-to-wake ratio: drop-in replacements

Each block below replaces one whole paragraph, caption, table row or environment
in the manuscript. Find the paragraph by its first words (given in the heading) and
swap the entire thing. Everything not listed stays as it is.

Convention: $r = T_{\mathrm{sleep}}/T_{\mathrm{wake}}$ is the *sleep-to-wake
ratio*, always a decimal. Defined once in the Sleep protocol; afterwards only $r$.
"Sleep duration" means $T_{\mathrm{sleep}}$ only. Phases are *wake period* and
*sleep episode*.

---

## 1. Abstract (whole abstract)

```latex
Spike-timing-dependent plasticity (STDP) provides a biologically-plausible learning mechanism for spiking neural networks (SNNs); however, Hebbian weight updates in architectures with recurrent connections suffer from pathological weight dynamics, such as unbounded growth or loss of representational diversity. We propose a neuromorphic regularization scheme inspired by the synaptic homeostasis hypothesis: periodic offline phases during which external input is suppressed, intrinsic membrane noise drives spontaneous activity under continued STDP, and every synaptic weight is pulled a fixed fraction of its logarithmic distance toward a homeostatic target. We demonstrate that this sleep-wake cycle prevents weight saturation while preserving learned structure. Across four MNIST-family datasets and without data-specific hyperparameter tuning, every sleep-to-wake ratio we tested, from $0.1$ to $1$, lifted the unregularized network from chance-level accuracy, with performance plateauing from a ratio of $0.1$ onward and peaking, non-significantly, at $0.3$. Sleep was also found to outperform continuous weight decay significantly. For layer normalization and synaptic scaling, although the results of sleep were overall better, the improvement was not statistically significant on a population level. A full factorial ablation of the four sleep components showed that downscaling accounts for nearly the entire benefit, whereas sleep-phase STDP, membrane noise, and input suppression, which we had hypothesized to drive consolidation through replay, contributed only small and non-significant improvements. Taken together, these results suggest that sleep stabilizes local Hebbian learning primarily by downscaling weights toward a reference rather than through replay-like activity (noisy membrane with STDP), performing on par with established normalization methods while relying only on local updates.
```

## 2. Highlights (whole environment)

```latex
\begin{highlights}
\item Periodic sleep stabilizes STDP in a recurrent excitatory–inhibitory spiking network
\item Every sleep-to-wake ratio from 0.1 to 1 lifts accuracy from chance to $\sim66\%$
\item Sleep matches layer normalization and synaptic scaling, beats weight decay
\item Mixed-effects models generalize effects across a population of datasets
\end{highlights}
```

## 3. Introduction — "Our contributions are fourfold."

```latex
Our contributions are fourfold. First, we present a sleep--wake mechanism that combines homeostatic downscaling with spontaneous activity under continued STDP, and we show empirically and mechanistically that periodic downscaling bounds weight growth while preserving the rank order, and thereby the learned structure, of the weights; consistent with \citet{capone_sleep-like_2019}. Second, we provide empirical validation across MNIST-family benchmarks using hierarchical mixed-effects models that generalize conclusions to a population of datasets, and we show that every non-zero sleep-to-wake ratio substantially improves classification accuracy over the unregularized network, with performance plateauing from a sleep-to-wake ratio of $0.1$ onward. Third, in comparisons with conventional regularization techniques, sleep performs on par with layer normalization and synaptic scaling and significantly outperforms continuous weight decay and unregularized STDP, while relying only on local updates. Finally, a full factorial ablation identifies downscaling toward the target as the functional driver of these gains, whereas sleep-phase STDP, intrinsic membrane noise, and input suppression provide only small and non-significant improvements, contrary to our initial hypothesis that replay-like activity drives the benefit.
```

## 4. Introduction — "The remainder of the paper is organized as follows"

```latex
The remainder of the paper is organized as follows: Section~2 describes the methods used in this work, including the spiking neural network implementation, its architecture, learning rule, and sleep–wake regularization protocol. As an initial validation step, we first evaluate the model on a controlled toy classification task to verify stable learning dynamics under unsupervised STDP, before scaling to MNIST-family experiments on the sleep-to-wake ratio, ablations and benchmarking against other regularization techniques. Section~3 presents the experiments and results. Section~4 discusses the broader implications for neuromorphic hardware, limitations, and future research directions.
```

## 5. Sleep protocol — "Training alternates cyclically between wake and sleep" (up to Eq. decay)

This is the one place $r$ is defined.

```latex
Training alternates cyclically between wake and sleep: each wake period of duration $T_{\mathrm{wake}}$ is followed by an offline sleep episode of duration $T_{\mathrm{sleep}} = r\,T_{\mathrm{wake}}$, where $r$ is the \textit{sleep-to-wake ratio}, after which wake processing resumes. Because $r$ is defined relative to the preceding wake period rather than to total training time, sleep at $r = 1$ is as long as wake and occupies half of the simulated time, not all of it. During wake periods, the network processes external sensory input $S_j^{\text{in}}(t)$ in the absence of intrinsic noise ($\xi_i(t) = 0$), and synapses undergo standard STDP updates. However, during sleep episodes, external sensory input is suppressed ($S_j^{\text{in}}(t) = 0$), intrinsic Gaussian noise $\xi_i(t) \sim \mathcal{N}(0,\sigma^2_\xi)$ is injected into the membrane dynamics (Eq.~\eqref{eq:lif}) to drive spontaneous spiking activity, and at every time step $t$ each synaptic weight is downscaled toward the target magnitude $w_{\mathrm{tgt}}$:
```

## 6. Sleep protocol — "In logarithmic distance space"

```latex
In logarithmic distance space, defining $d_{ij}(t) = \ln(|w_{ij}(t)| / w_{\mathrm{tgt}})$, Eq.~\eqref{eq:decay} acts as a linear contraction $d_{ij}(t) = \lambda(r)\, d_{ij}(t - \Delta t_{\mathrm{step}})$. To evaluate the functional effect of sleep duration independently of the total magnitude of downscaling, the decay rate $\lambda(r)$ is calibrated to maintain an identical cumulative contraction factor across sleep episodes of different lengths, as detailed in \ref{sec:app:der_lambda}. This isolates the duration of spontaneous offline STDP from the net downscaling. Fig.~\ref{fig:weight_dynamics} illustrates the resulting dynamics, and Algorithm~\ref{alg:train} summarizes the procedure.
```

## 7. Algorithm 1 (whole environment)

Terminology only: wake period, sleep-to-wake ratio, sleep episode.

```latex
\begin{algorithm}[tpb]
\footnotesize
\caption{SNN training with scheduled sleep}
\label{alg:train}
\begin{algorithmic}[1]
\REQUIRE Initial weights $w_{ij}(0)$, sensory input $S_j^{\text{in}}(t)$, duration $T$, time step $\Delta t_{\mathrm{step}}$, wake period $T_{\mathrm{wake}}$, sleep-to-wake ratio $r$, decay rate $\lambda(r)$, target $w_{\mathrm{tgt}}$, noise variance $\sigma_\xi^2$
\STATE $T_{\mathrm{sleep}} \gets r T_{\mathrm{wake}}$
\STATE Initialize states: $U_i(0) \gets U_{\text{rest}}$, $I_i^{\text{syn}}(0) \gets 0$, $S_i(0) \gets 0$, $U_{\text{th}}^i(0) \gets \theta$, $\alpha_i(0) \gets 0$
\FOR{$t = \Delta t_{\mathrm{step}}, 2\Delta t_{\mathrm{step}}, \dots, T$}
    \IF{in wake period}
        \STATE Update synaptic currents $I_i^{\text{syn}}(t)$ from sensory spikes $S_j^{\text{in}}(t)$ and recurrent spikes $S_j(t - \Delta t_{\mathrm{step}})$
        \STATE Update membrane potentials $U_i(t)$ driven by synaptic currents $I_i^{\text{syn}}(t)$ \COMMENT{Eq.~\eqref{eq:lif}}
        \STATE Emit spikes $S_i(t) \gets \mathbb{I}(U_i(t) \ge U_{\text{th}}^i(t))$ and update thresholds $U_{\text{th}}^i(t)$ \COMMENT{Eq.~\eqref{eq:threshold}}
        \STATE Update weights via STDP/iSTDP with wake spikes \COMMENT{Eqs.~\eqref{eq:stdp}, \eqref{eq:istdp}}
    \ELSE[\textit{offline sleep episode, duration $T_{\mathrm{sleep}}$}]
        \STATE Suppress external input ($S_j^{\text{in}}(t) \gets 0$) and sample noise $\xi_i(t) \sim \mathcal{N}(0, \sigma_\xi^2)$
        \STATE Update membrane potentials $U_i(t)$ driven by intrinsic noise and recurrent spikes $S_j(t - \Delta t_{\mathrm{step}})$ \COMMENT{Eq.~\eqref{eq:lif}}
        \STATE Emit spontaneous spikes $S_i(t) \gets \mathbb{I}(U_i(t) \ge U_{\text{th}}^i(t))$ and update thresholds $U_{\text{th}}^i(t)$ \COMMENT{Eq.~\eqref{eq:threshold}}
        \STATE Downscale weights: $w_{ij}(t) \gets \operatorname{sgn}(w_{ij}(t - \Delta t_{\mathrm{step}}))\, w_{\mathrm{tgt}} \left(\frac{|w_{ij}(t - \Delta t_{\mathrm{step}})|}{w_{\mathrm{tgt}}}\right)^{\lambda(r)}$ \COMMENT{Eq.~\eqref{eq:decay}}
        \STATE Update weights via STDP/iSTDP with spontaneous spikes \COMMENT{Eqs.~\eqref{eq:stdp}, \eqref{eq:istdp}}
    \ENDIF
\ENDFOR
\RETURN $w_{ij}(T)$
\end{algorithmic}
\end{algorithm}
```

## 8. MNIST-family experiment — "To assess whether the benefits of sleep extend"

```latex
To assess whether the benefits of sleep extend beyond the toy setting to more complex image datasets, we vary the sleep-to-wake ratio across 11 values, from $r = 0$ (no sleep) to $r = 1$ in steps of $0.1$. Sleep is triggered after every $3{,}500$ steps of wake activity ($T_{\mathrm{wake}} = 3{,}500\,\text{ms}$), which amounts to 28 sleep episodes per training batch. During each episode, external input is suppressed and the network is driven by spontaneous, noise-induced activity, while synaptic downscaling and STDP continue to update the weights. At $r = 0.1$, every $3{,}500$ wake steps are therefore followed by $350$ sleep steps, and at $r = 1$ both last $3{,}500$ steps.
```

## 9. MNIST-family experiment — "Downscaling is applied at every sleep step"

```latex
Downscaling is applied at every sleep step, so with a fixed decay rate, longer sleep would shrink the weights further toward their targets, confounding sleep duration with the strength of downscaling. We therefore hold the cumulative downscaling per episode (and batch) constant across all values of $r$ by calibrating the per-step decay rate $\lambda(r)$ (see Appendix~\ref{sec:app:der_lambda} for the full derivation). All configurations therefore receive the exact same net homeostatic downscaling and differ only in the offline duration during which spontaneous activity and STDP act on the weights.
```

## 10. MNIST statistical model — "where $y_{ijk}$ is the test accuracy" (text after the equations)

```latex
where $y_{ijk}$ is the test accuracy, $j$ indexes the sleep-to-wake ratios $r \in \{0.1, 0.2, \dots, 1\}$, $k$ the datasets and $i$ the seeds.
Each $\beta_j^{\text{(sleep)}}$ quantifies the fixed effect of the sleep-to-wake ratio with index $j$ relative to the no-sleep baseline ($r = 0$, $\beta_0$); all values of $r$ are modeled as categorical factor levels for direct interpretability. Variation across datasets, random seeds, and dataset-specific seed realizations is captured by the random intercepts $b_k^{\text{(dataset)}} \sim \mathcal{N}(0, \sigma_{\text{dataset}}^2)$, $b_i^{\text{(seed)}} \sim \mathcal{N}(0, \sigma_{\text{seed}}^2)$, and $b_{ik}^{\text{(dataset$\times$seed)}} \sim \mathcal{N}(0, \sigma_{\text{dataset}\times\text{seed}}^2)$, respectively, with a single dispersion parameter $\phi$ shared across conditions. This design allows to generalize conclusions from our model to the whole population of similar image datasets and their splits.
```

## 11. MNIST results — "Every nonzero sleep-to-wake ratio improved accuracy"

```latex
Every nonzero sleep-to-wake ratio improved accuracy over the no-sleep condition by a wide margin (Table~\ref{tab:sleep_sweep}; $\beta_j$ effects between $3.00$ and $3.07$, all $p<.001$): for a typical dataset, the model-predicted accuracy rose from $8.5\%$ without sleep, close to random chance for ten classes, to $65$--$67\%$ with sleep. Beyond this initial improvement, varying $r$ made little difference: the coefficients for $r > 0$ spanned only $0.07$, which is of the same order as the standard error of about $0.06$ for each coefficient. The highest sample point was observed at $r = 0.3$, though it was not significantly different from the other nonzero ratios\footnote{Hence, in further experiments, we carried $r = 0.3$ forward to the comparison with conventional regularization methods and the ablation study.}.
```

## 12. MNIST results — "The random effects show where the remaining variation lies"

```latex
The random effects show where the remaining variation lies: Most of it is between datasets corresponding to a standard deviation of $0.37$, which reflects that the datasets in the population of MNIST-like images differ in difficulty, whereas the variation between seeds ($0.04$) and between seeds within a dataset ($0.06$) is small, so the effect of sleep is highly consistent across runs and splits.
Fig.~\ref{fig:sleep_benchmark} illustrates this pattern: on a population of MNIST-like datasets, accuracy rises steeply from $r = 0$ to $r = 0.1$ and remains on a plateau through $r = 1$.
```

## 13. Sweep figure (`fig:sleep_benchmark`) — caption line

```latex
    \caption{Population-level predicted accuracy ($\pm$95\% CI) versus sleep-to-wake ratio $r$ across the population of MNIST-family benchmarks. Each dataset-specific dot represents the observed mean test accuracy for a given dataset and value of $r$.}
```

## 14. Sweep table (`tab:betareg` / `tab:sleep_sweep`)

Replace the whole `{ ... longtable ... }` block with `results/glmm/sweep_table.tex`
(full-width version, rows labelled $r = 0.1 \dots 1$).

## 15. Baselines statistical model — "We fit a second hierarchical beta-regression model"

(Also fixes the missing parentheses on the second equation reference.)

```latex
We fit a second hierarchical beta-regression model with the same structure as model from Eqs.~\eqref{eq:glmm_sleep_sweep_main} and~\eqref{eq:glmm_sleep_sweep_full}, replacing in the fixed effects part of the linear predictor the sleep-to-wake ratios with the regularization methods (with sleep evaluated at $r = 0.3$), while keeping the structure of the random effects unchanged thus targeting the same population of MNIST-like data and their splits:
```

## 16. Baselines table — footnote row

```latex
    \multicolumn{6}{p{\linewidth}}{\footnotesize Sleep was applied at $r = 0.3$. Significance levels: * $p<.05$, ** $p<.01$, *** $p<.001$.}\\
```

## 17. Baselines figure (`fig:baselines`) — caption line

```latex
    \caption{Population level test accuracy of sleep (at $r = 0.3$) and the alternative regularization methods on each dataset, with five seeds per condition. Bars show the population level accuracy for each model; error bars show the respective $95\%$ confidence intervals. Further, \textit{none} denotes unregularized STDP and \textit{layer norm} the layer-wise weight normalization.}
```

## 18. Ablation — "To understand \textit{how} sleep improves the network"

```latex
To understand \textit{how} sleep improves the network, we isolate the contribution of each of its four components (downscaling, STDP, membrane noise, and input suppression) in an ablation study. Each component is switched on or off in a full factorial design, which yields $2 \times 2\times 2\times 2 = 16$ combinations, and every combination is trained on all four datasets with five seeds at $r = 0.3$, resulting in $320$ runs in total.
```

## 19. Discussion — "First, sleep reliably prevents the collapse"

```latex
First, sleep reliably prevents the collapse of the unregularized network (the baseline condition with no sleep, $r=0$). Every
sleep-to-wake ratio we tested, $r \in [0.1, 1]$, lifted the network from chance
level to roughly two-thirds correct, with little difference across ratios
(Table~\ref{tab:betareg}). Sleep is therefore an effective regularizer against
runaway weights, and its benefit does not depend on a finely tuned sleep-to-wake ratio
(Fig.~\ref{fig:sleep_benchmark}). Interestingly, the ratio that attained the highest sample accuracy, $r = 0.3$, loosely mirrors typical human sleep schedules: about $6$ hours of sleep after an $18$-hour wake day gives $r \approx 0.33$, although our findings demonstrate that computational stabilization remains broadly robust across a wide range of ratios.
```

## 20. Conclusion — "We introduced a sleep phase for recurrent spiking neural networks"

```latex
We introduced a sleep phase for recurrent spiking neural networks trained with
STDP and asked how it behaves across sleep-to-wake ratios, how it compares with
conventional regularizers, and which of its components drive performance. Without
regularization, runaway weight growth drove the network to chance level. Every
sleep-to-wake ratio we tested, from $0.1$ to $1$, prevented this collapse and
brought the network to roughly two-thirds correct, with little difference
across ratios. Sleep therefore stabilizes STDP learning without requiring a
finely tuned schedule.
```

## 21. Appendix A (hyperparameters) — two rows

```latex
                  & $r$ & Sleep-to-wake ratio & $0.3$ \\
```

```latex
                  & $T_{\mathrm{wake}}$ & Wake period duration & $3{,}500\,\text{ms}$ ($3{,}500$ steps) \\
```

## 22. Appendix B — "To evaluate the functional role of sleep duration" (up to Eq. episodes_batch)

```latex
To evaluate the functional role of sleep duration independently of cumulative downscaling, the episode retention factor $s$ is held constant across all values of $r$. The value of $s$ is calibrated against the training batch structure: with $B = 1{,}000$ image presentations of duration $T_{\mathrm{image}} = 100\,\text{ms}$ per batch and wake period $T_{\mathrm{wake}} = 3{,}500\,\text{ms}$, each batch contains
```

## 23. Appendix B — "Then for the sleep-to-wake ratio" (up to Eq. lambda_num)

```latex
Then for $r = 0.3$ ($T_{\mathrm{sleep}} = 0.3 \times 3{,}500\,\text{ms} = 1{,}050\,\text{ms}$), the value carried forward to the comparison with conventional regularizers (Section~\ref{sec:res:reg_comp}) and the ablation study (Section~\ref{sec:res:abl}), with simulation time step $\Delta t_{\mathrm{step}} = 1\,\text{ms}$, Eq.~\eqref{eq:lambda_deriv} evaluates to:
```

## 24. Appendix D, Shift in weight distributions — "To show how downscaling changes the weight distribution"

```latex
To show how downscaling changes the weight distribution, we trained the model on $4{,}000$ MNIST images at $r = 0.3$, with downscaling as the only active sleep component. We recorded all $12{,}426$ excitatory weights immediately before and after each sleep episode. Fig.~\ref{fig:shift_target} compares the excitatory weight distribution before and after a single sleep episode at two points in training. At the first episode, all weights are still close to their initial value of $0.3$, below the target of $0.5$, so sleep shifts the whole distribution upward (median from $0.304$ to $0.309$). By the $61$st episode, STDP has pushed most weights far above the target, and sleep shifts them down (median from $2.64$ to $2.50$), while a small group of rarely active synapses that STDP left near their initial value keeps rising toward $0.5$. Sleep therefore does not simply shrink the weights: it moves each one toward the target, up or down depending on where wake has left it.
```

## 25. Appendix D, Rank order — "Downscaling moves every weight closer to the target"

```latex
Downscaling moves every weight closer to the target, but (on its own) it never lets one weight overtake another: a synapse that is stronger than another before sleep remains stronger after it, only by a smaller margin (Eq.~\eqref{eq:decay}). We tested this in the same downscaling-only runs as in Section~\ref{sec:res:shift}, in which the model was trained on $4{,}000$ MNIST images at $r = 0.3$, with $112$ sleep episodes per run and five seeds. For each episode, we ranked all $12{,}426$ excitatory synapses by weight and compared each synapse's percentile rank before and after the sleep episode, and before and after the wake period that follows it. Figure~\ref{fig:rank_retention} shows this comparison for the first episode of one seed. Across sleep, the ranks lie on the diagonal, whereas across wake they scatter around it, because STDP strengthens some synapses more than others. Counting the pairs of synapses whose order is reversed, measured at every tenth episode in all five seeds, one sleep episode reversed on average $0.004\%$ of all pairs (at most $0.05\%$), compared with $1.0\%$ (at most $5.4\%$) for one wake period. The few reversals during sleep occur only among near-identical weights early in training and are caused by the finite numerical precision with which the weights are stored; applying the downscaling rule exactly to the same weights reverses none.
```
