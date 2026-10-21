# From the current DT–NB experiments to a controlled cleaning method

Prepared 9 October 2026. This is an explanation and a proposed research direction, not a report of a new experiment.

Updated scope: the next study uses Logistic Regression, Decision Tree and Naive Bayes as final classifiers, with no MLP. The newer [parallel strategy and fresh run synthesis](parallel-nb-dt-strategy.md) supersedes this guide's final-tree-first and correction-first recommendations. The historical explanations and results below remain useful background.

## 1. What I recommend

I recommend building on E12v2, with a decision tree as the first final classifier. Keep Naive Bayes and a decision tree as the two judges that suggest label corrections. Add a small number of feature subsets and a rule that can reject the proposed changes when the evidence is too weak.

The research question would be:

> Can we correct some training labels and use fewer features, while keeping each class's prediction error within a stated tolerance and staying within a measured computing budget?

This is a proposed study. We have not shown that the new method works, that it is novel, or that a paper based on it will be accepted. The tree-specific gain in E12v2 makes it worth testing. It does not establish the answer.

Do not start by replacing the whole project or adding many new models. First find out whether a small extension improves on the existing dual-correction rule and on doing nothing.

## 2. What the phrase means in simple words

**Label treatment** means deciding what to do with a row whose class label looks wrong. We could keep it, correct the label, reduce its training weight, or remove the row. For the first proposed study, I recommend only keeping or correcting. Keep deletion as a separate comparison.

**Feature reduction** means using fewer input columns. It does not mean deleting a class or removing test examples. If a column is removed from training, the final classifier also receives test examples without that column.

**Joint** means choosing the label rule and feature subset together. A feature can look useful when labels are wrong and less useful after correction. Removing features can also change which labels look suspicious. We therefore compare complete alternatives, not declare the two decisions independent.

**A class-error budget** means setting a limit on how much worse the model may become for each class compared with the same classifier trained without cleaning. A training class-count floor is different: retaining five rows of a class does not guarantee good predictions for that class.

**A computation budget** means setting and measuring limits such as total processing time, peak memory, model size, or prediction cost. It includes the cost of finding and applying the cleaning rule. A smaller final dataset does not automatically mean a cheaper overall method.

Illustration only: suppose an uncleaned model misses 10% of class A and 30% of class B. A two-percentage-point tolerance would allow at most 12% and 32%, respectively. A candidate that reaches 9% and 40% fails because class B suffers too much, even if overall accuracy improves. We would need uncertainty bounds, not just these sample percentages, to certify that comparison.

The tolerance is a research design choice. It must be justified and fixed before the final evaluation. Two points is an example, not an approved setting.

## 3. The basic mathematical language

Our training data are

\[
D=\{(x_i,\widetilde y_i)\}_{i=1}^{n},
\qquad x_i=(x_{i1},\ldots,x_{id}).
\]

- \(n\): number of training rows.
- \(d\): number of input features.
- \(x_i\): the input values for row \(i\).
- \(\widetilde y_i\): the recorded label, which might be wrong.
- \(y_i^*\): the actual correct label, usually unknown in real data.
- \(\widehat y_i\): a model's predicted label.

A wrong prediction, \(\widehat y_i\ne\widetilde y_i\), does not tell us which side is wrong. The model may be wrong, the recorded label may be wrong, or the inputs may not contain enough information to distinguish the classes.

Also distinguish three problems: a wrong class label, a corrupted input value, and an unnecessary input feature. The proposed first study addresses label treatment and feature reduction. It does not automatically repair corrupted measurements.

## 4. What we already implemented

### 4.1 Naive Bayes: combine evidence from the features

Plain NB predicts the class with the largest score:

\[
\widehat y(x)=\arg\max_c\left[
\log\widehat P(c)+\sum_{j=1}^{d}\log\widehat p(x_j\mid c)
\right].
\]

Here \(\widehat P(c)\) is the estimated frequency of class \(c\). The term \(\widehat p(x_j\mid c)\) says how compatible feature value \(x_j\) is with that class. For continuous features this can be a probability density, rather than a probability mass. Logs turn multiplication into addition and avoid numerical underflow.

NB treats features as conditionally independent within a class. This makes it inexpensive to fit and score. It can still classify well when independence is imperfect, but correlated features can count similar evidence several times. Poor likelihood assumptions or very few examples of a class can also lead to confident mistakes.

Our weighted NB changes the score to

\[
s_c(x)=\log\widehat P(c)+\sum_{j\in S}w_j\log\widehat p(x_j\mid c),
\]

where \(S\) is the retained feature set. A smaller \(w_j\) reduces that feature's contribution. This does not multiply the raw feature value by \(w_j\). Normalizing \(\exp(s_c)\) gives the implementation's class-probability scores; normalization alone does not prove that those scores are calibrated probabilities of being correct.

There are separate implementations. E9 uses FaithfulNB with original columns and Weka J48. The later pipeline uses sklearn trees and one-hot columns; its mixed NB uses Bernoulli likelihoods for binary columns and Gaussian likelihoods for other columns. Do not pool their numerical results as if they were one experiment.

### 4.2 Algorithm 1: remove rows NB predicts differently

The inherited filter fits NB on the training data, predicts those same training rows, and retains

\[
D_N=\{(x_i,\widetilde y_i)\in D:\widehat y_i^{NB}=\widetilde y_i\}.
\]

**Why it might help:** if the judge is usually right on disputed cases, removing wrong labels may prevent the final classifier from learning those errors.

**Why it can fail:** a valid difficult example or rare-class example can disagree with NB. Deleting it removes useful information. The judge also sees the same rows during fitting and judging, so its training predictions do not provide independent evidence of label correctness. Our inherited safeguard only cancels deletion if fewer than two classes would remain; it does not protect every class.

An elementary calculation explains one necessary distinction. Let the original wrong-label fraction be \(\eta\). Suppose a filter removes a fraction \(a\) of truly wrong-label rows and a fraction \(b\) of correctly labeled rows. The remaining wrong-label fraction is

\[
\eta_{\mathrm{after}}=
\frac{\eta(1-a)}{\eta(1-a)+(1-\eta)(1-b)}.
\]

For \(0<\eta<1\) and a nonempty retained population, this fraction decreases exactly when \(a>b\). The filter must remove wrong-label rows at a higher rate than correct-label rows. This is a simple identity, not a new theorem. Even satisfying it does not prove better classification: the removed correct rows may contain important boundary or minority-class information.

### 4.3 Algorithm 2: keep features used by a tree

A decision tree chooses feature tests that divide training rows into groups. For an entropy-based tree, a split is useful when it reduces class uncertainty:

\[
H(Y)=-\sum_c p_c\log p_c,
\qquad
\mathrm{gain}=H(\mathrm{parent})-
\sum_{v\in\mathrm{children}}\frac{n_v}{n_{\mathrm{parent}}}H(v).
\]

This expresses the basic idea. J48 and sklearn CART have different split and pruning details; they are not interchangeable implementations.

Our inherited attribute rule is

\[
w_j=\begin{cases}
1/\sqrt{d_j},&\text{if the tree tests feature }j,\\
0,&\text{otherwise},
\end{cases}
\qquad S=\{j:w_j>0\},
\]

where \(d_j\) is the smallest depth at which feature \(j\) appears, with the root at depth 1. Depths 1, 4, and 9 give weights 1, 0.5, and approximately 0.33.

**Why it might help:** the tree can identify features useful for separating classes, and fewer features can reduce unnecessary contributions to NB.

**Why it can fail:** one tree is only one fitted model. A useful feature may be absent because a correlated feature was chosen instead, because its benefit appears only in a small class, or because the sample is too small. Noisy labels can also create misleading splits. The depth formula is a heuristic; we have no proof that it gives the best NB weights.

An empty tree has implementation-specific fallback behavior. The CART pipeline retains all features with unit weights. In the original-column implementation, standalone attribute-selection NB can use class priors; combined paths retain the current columns. No empty-selection safeguard occurred in the stored E9 conference runs.

### 4.4 The two combined paths

Let \(N\) mean row filtering and \(A\) mean attribute selection and weighting. We compared

\[
D\xrightarrow{N}\text{retained rows}\xrightarrow{A}\text{selected features}
\]

and

\[
D\xrightarrow{A}\text{selected features and weights}
\xrightarrow{N_w}\text{retained rows}.
\]

The second path uses the attribute weights inside the NB judge. Therefore, the comparison changes both order and judge weighting. It is not a pure order-only experiment. The second path does not refit the attribute weights after deletion.

Combining two useful-looking operations does not guarantee a useful result. If the first operation removes valid information, the second cannot recover it.

### 4.5 E5 and E11: less aggressive treatment and multiple judges

E5 tested soft row weights, majority deletion, and label correction using NB, J48, and logistic-regression judges. Those judges predicted their own training rows. Committee deletion was less damaging than single-NB deletion in the saved mean results, but it did not exceed the plain-tree mean.

E11 used NB and CART predictions made out of fold, with repeated judging, and checked transfer to LR and a small MLP. Some treatments could still eliminate small classes. Attribute selection and input scaling also changed together, which prevents attributing their effects to either one alone.

These experiments taught us that agreement and softer treatment can change the result, but neither establishes safety or a universal benefit.

### 4.6 E12v2: correct only when both judges support the same alternative

Out-of-fold means that a row is judged by a model fitted without that row. Write the judge probability as

\[
p_i^{(g)}(c)=\widehat P_g(c\mid x_i;D\setminus F(i)),
\]

where \(g\) is NB or DT and \(F(i)\) is the inner fold containing row \(i\). This avoids using that row to fit its own judge. It does not guarantee that the judge is accurate.

For each judge and class, the existing threshold is

\[
t_c^{(g)}=
\frac{1}{|I_c|}\sum_{i\in I_c}p_i^{(g)}(c),
\qquad I_c=\{i:\widetilde y_i=c\}.
\]

In words: take the average probability assigned to class \(c\) among rows recorded as class \(c\). The labels defining this group can themselves be noisy.

Let \(a_i^{(g)}=\arg\max_c p_i^{(g)}(c)\). The rule proposes correction to \(c\) when

\[
a_i^{(NB)}=a_i^{(DT)}=c\ne\widetilde y_i,
\quad p_i^{(NB)}(c)\ge t_c^{(NB)},
\quad p_i^{(DT)}(c)\ge t_c^{(DT)}.
\]

The row must also be scorable and pass the class safeguard. Otherwise its label stays unchanged. The final rule is

\[
y_i'=\begin{cases}
c,&\text{if all correction conditions pass},\\
\widetilde y_i,&\text{otherwise}.
\end{cases}
\]

The main treatment retains every training row. The safeguard preserves at least \(\min(5,n_c)\) original-label rows of each observed class from departure; singletons are protected. This preserves some representation, not class recall.

**Why it might help:** two different judges agreeing on a replacement can be more selective than one judge disagreeing with the original label. Correction preserves the input example for training.

**Why it can fail:** both judges can make the same error. Their mistakes are not independent just because the algorithms differ. A class-average confidence threshold is not a statistically certified reliability threshold. Original feature problems can affect both judges.

This is our threshold-and-agreement heuristic. It is not a full Confident Learning implementation, despite the historical function name `confident_joint`.

## 5. What the experiments actually show

The following are saved experiment results, not fresh runs for this document.

| Study | Measured result | Supported interpretation |
|---|---|---|
| E9, original columns and Weka J48 | Plain tree 86.33% accuracy; instances-first 81.21%; attributes-first 80.89% | The combinations had lower mean tree accuracy. |
| E9, NB | Plain NB 75.93%; attributes alone 77.35%; instances-first 75.63%; attributes-first 76.35% | Combining the steps did not exceed attribute weighting alone in the mean. |
| E9 inference | None of the short paper's eight comparisons passed Holm-adjusted 0.05 | We have no statistically established advantage in those comparisons. This does not prove equality. |
| E5 | Soft treatment 82.16%, committee deletion 85.77%, correction 85.12%, plain tree 86.33% accuracy | Reducing the harm of a weaker filter is not evidence of beating no cleaning. |
| E11 | Combined treatment changed LR accuracy by +0.50 points and MLP by -1.38 | Transfer was mixed, with unresolved class harms and treatment confounding. |
| E12v2 primary comparison | +2.5205 macro-F1 points at 20% added noise; 8/10 dataset effects positive; Wilcoxon p=0.083984375 | Overall improvement was not established at 0.05. |
| E12v2 by final learner | DT +5.642, NB +1.134, LR +0.786 macro-F1 points | The DT result motivates a new study; it was not the original standalone primary hypothesis. |
| E12v2 without added noise | 5.44% of reference labels changed; accuracy -0.094 and macro-F1 +0.350 points on average | Some datasets were harmed. Original labels are not certified ground truth. |
| E12v2 deletion sensitivity | +2.829 macro-F1 points at 20% noise | One seed; cannot establish that deletion is better than correction. |
| E12v2 small MLP check | +1.434 points at 20% noise and -0.162 with no added noise | Three datasets and one seed; no broad neural-network claim. |

The original E12 pilot had a correction/deletion implementation error and an unsuitable AUROC score. Its outputs must not support new correction claims. E12v2 is the repaired study.

Macro-F1 gives each class equal weight:

\[
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c},
\qquad \mathrm{MacroF1}=\frac1C\sum_{c=1}^{C}F1_c.
\]

E12v2 pools outer-fold confusion counts within a seed before computing the scores. E9 averages fold scores with a potentially varying class set. This is another reason to keep the studies separate.

The plausible failure explanations are loss of valid boundary examples, minority-class harm, correlated judge errors, unstable feature selection, and model-specific responses to treatment. The records support investigating these explanations. We have not isolated each as a proven cause.

## 6. How we kept evaluation separate from cleaning

Protocol B fits cleaning and the classifier inside the training fold. It applies the retained columns to the test fold but keeps every test row. This is our main evaluation design.

Protocol A cleans before cross-validation. That exposes future test labels and, with row filtering, changes which examples get tested. Its higher score cannot be read as better generalization. Protocol C fits and evaluates on the same data, so it is a training diagnostic.

In E12v2, artificial errors are added only to outer-training labels. The cleaner sees the corrupted labels, not the corruption mask or original reference labels. Outer-test labels remain unchanged. Those test labels are reference labels, not independently certified truth.

More seeds improve measurement of split sensitivity. They do not turn ten datasets into hundreds of independent datasets.

## 7. The smallest next method worth testing

### 7.1 Keep the existing dual rule and add a conservative margin

For a row on which both judges propose the same alternative \(c\), define

\[
s_i=\min_{g\in\{NB,DT\}}
\left[p_i^{(g)}(c)-p_i^{(g)}(\widetilde y_i)\right].
\]

This is the weaker judge's preference for the replacement over the original label. A proposed extension would require both the existing E12 conditions and \(s_i\ge\tau\). A larger \(\tau\) makes correction more conservative. Rows failing the rule remain unchanged.

Treat this margin as a score, not a probability that the original label is wrong. Keep the existing E12 rule as a comparator. The margin itself is not a novelty claim.

### 7.2 Propose a few feature subsets

Fit a small number \(B\) of trees on different training subsamples. Let \(S_b\) be the attributes used by tree \(b\). Compute

\[
\pi_j=\frac1B\sum_{b=1}^{B}\mathbf1\{j\in S_b\}.
\]

If \(\pi_j=0.8\), that feature appears in 80% of these fitted trees. It does not mean an 80% probability that the feature is truly important.

Use the frequencies to propose a few feature budgets, always including all features. Count a categorical attribute as one group when its dummy columns are selected. Correlated attributes can substitute for one another, so inspect group behavior and do not automatically discard every low-frequency feature.

For the first prototype, use a fixed tie-breaking rule and a small number of subsets. Do not search all \(2^d\) possible subsets. Repeated tree selection is a candidate-generation heuristic; its success must be measured.

### 7.3 Compare complete candidate pipelines

Let \(\lambda=(\tau,k)\) identify a correction threshold and feature budget. A candidate model is

\[
f_\lambda=\operatorname{TrainDT}
\left(\{(x_{i,S_\lambda},y_i^{(\lambda)})\}_{i=1}^{n}\right).
\]

For a manageable first version, score label suspicion once with full-feature judges, apply each proposed correction strength, and discover feature subsets within that treated training data. This jointly selects the final pair without creating a repeated correction/selection feedback loop. Iteratively refitting judges on selected features would be a separate later question.

Keep the final tree settings matched across alternatives. Use NB as a judge first; testing final NB can follow if the main question survives. Do not add learned NB weights and many new classifiers to this first extension.

The choices must include an untouched baseline \(f_0\), correction only, selection only, and correction plus selection. Otherwise we cannot tell whether combining the operations adds value.

### 7.4 Make the decision subject to explicit limits

Define class error as

\[
R_c(f)=P(f(X)\ne Y^*\mid Y^*=c)=1-\operatorname{Recall}_c(f).
\]

A research objective is

\[
\min_\lambda \operatorname{DeploymentCost}(f_\lambda)
\quad\text{subject to}\quad
R_c(f_\lambda)-R_c(f_0)\le\epsilon_c\ \text{for every class }c,
\]

with separate limits on total search time and peak memory. For the first prototype, the number of retained original attributes can be the simple cost objective. Later measure prediction time and model size directly; fewer attributes need not mean proportionally faster tree prediction.

We cannot directly observe population risks, so this expression states the goal. An independent audit and statistical bounds are needed to support its constraints.

## 8. What we can prove, under which conditions

### 8.1 Separate three kinds of evidence

There are different possible claims:

- The treated training labels contain fewer errors.
- The final classifier predicts better.
- The final classifier uses fewer resources with limited predictive loss.

Proving or measuring one does not automatically establish the others. Our recommended first theoretical target is the third claim, with prediction improvement tested separately.

### 8.2 A simple class-error bound

Freeze the baseline and \(M\) candidate predictors before accessing an independent audit sample with trustworthy labels. For class \(c\), let \(m_c\) be its audit count. Compute paired error differences:

\[
\widehat\Delta_{\lambda,c}=
\frac1{m_c}\sum_{i:y_i^*=c}
\left[\mathbf1\{f_\lambda(x_i)\ne y_i^*\}
-\mathbf1\{f_0(x_i)\ne y_i^*\}\right].
\]

Each difference is -1, 0, or 1: the candidate fixes a baseline error, has the same correctness, or introduces an error. Define

\[
U_{\lambda,c}=\widehat\Delta_{\lambda,c}
+\sqrt{\frac{2\log(MC/\delta)}{m_c}}.
\]

Accept a candidate only if \(U_{\lambda,c}\le\epsilon_c\) for every class. Choose the cheapest accepted candidate; if none pass, return the unchanged baseline.

Conditional on the training procedure, independent and identically distributed audit observations from the target population give

\[
P\left(\forall c:\ R_c(f_{\mathrm{chosen}})
\le R_c(f_0)+\epsilon_c\right)\ge1-\delta.
\]

Why: Hoeffding's inequality bounds the chance that one true paired error difference exceeds its measured value by \(t\) by \(\exp(-m_ct^2/2)\). Give each of the \(MC\) comparisons failure allowance \(\delta/(MC)\), then add these allowances. This is a standard concentration argument, not a new theorem.

The baseline is always the fallback because its difference from itself is exactly zero. If a class has no audit examples, we cannot certify a new candidate for that class with this rule.

### 8.3 Important limits

- It controls loss of class recall, not macro-F1 or correctness of every corrected label.
- A passing candidate may still be slightly worse, within the chosen tolerance. A negative upper bound supports improvement for the corresponding risk.
- The candidate family, preprocessing, feature subsets, and models must be fixed before audit use. Choosing among them is allowed; inventing new candidates after looking at audit outcomes is not covered.
- The certified predictor must be the predictor used afterward. Refitting on the audit set invalidates this simple argument.
- Audit labels must be trustworthy. With noisy reference labels, the claim concerns those reference labels, not necessarily true classes.
- Patient, device, or temporal dependence and distribution shift require an appropriate design and possibly different bounds. Randomly splitting dependent rows does not fix this.

### 8.4 Why trusted-label cost is a feasibility gate

This simple bound can be conservative. For an overall-error comparison with 20 candidates, \(\delta=0.05\), and observed error difference near zero, a two-point tolerance requires approximately

\[
m\ge\frac{2\log(20/0.05)}{0.02^2}\approx29{,}958
\]

audit examples. Class-wise protection can demand substantial counts in each rare class. This is a sufficient count for this bound and scenario, not a universal lower limit. Strong observed improvement or a valid bound using the variance of paired differences can reduce the requirement.

Our tiny datasets cannot support strong class-wise certificates of this kind. Do not promise certification from a few hundred labels. First calculate the likely audit requirement. If no suitable trusted data are available, run an explicitly empirical study and omit the clean-label guarantee. Collecting or validating labels becomes part of the research cost.

## 9. What changes are needed for large data

The present documented datasets range from 24 to 25,192 rows. The loader creates dense one-hot arrays. The out-of-fold judges allocate full row-by-class probability arrays. These are reasonable experimental choices at this scale, but they do not demonstrate a big-data method.

For fixed fitted judges, one NB/tree scoring pass costs approximately

\[
O\bigl(N(Cd+h)\bigr),
\]

where \(N\) is the number of scored rows, \(d\) the input width, \(C\) the class count, and \(h\) the tree depth. This assumes standard dense NB calculations and ordinary tree traversal. It excludes judge fitting, repeated folds, feature search, final model training, and disk input/output.

The current full probability storage costs \(O(NC)\), in addition to the dense data matrix's \(O(Nd)\). Multiple arrays and cached judges raise the actual memory use further.

After a small prototype shows value, introduce chunked scoring, bounded training samples for feature discovery, cached reusable suspicion scores, and an explicit maximum number of candidate fits. Judges must still exclude the rows they score during fitting. Do not silently replace full fitting with sampling and claim unchanged accuracy.

Measure

\[
T_{\mathrm{total}}=T_{\mathrm{judges}}+T_{\mathrm{scoring}}
+T_{\mathrm{feature\ search}}+T_{\mathrm{candidate\ training}}+T_{\mathrm{audit}}.
\]

Report peak memory, final prediction cost, original-attribute count, and annotation cost as separate quantities. A cheaper final model may only repay an expensive preparation step after many uses. State which use case justifies that tradeoff.

## 10. From the current stage to the proposed stage

| Stage | Concrete work | Evidence needed before moving on |
|---|---|---|
| Preserve the current evidence | Keep E9 and E12v2 outputs unchanged. Start a separate study and output directory. | Existing numbers and their provenance remain traceable. |
| Check feasibility | Identify larger fresh datasets and a source of trustworthy audit labels. Estimate per-class audit counts and available RAM/time. | Either a realistic certification route, or an explicit decision to make empirical claims only. |
| Build a small prototype | Reuse the current out-of-fold judges, correction application, and class floors. Add the conservative margin and a few tree-derived feature subsets. | All corrections remain inside training data; all test rows remain; results include the baseline and component-only alternatives. |
| Test the mechanism | Compare untouched, E12 correction, conservative correction, selection only, and joint treatment. Include NB-only and DT-only judges. | We can identify whether the second judge and the feature step add value. |
| Test the selection safeguard | Freeze candidates and evaluate the audit rule, including fallback. | Report how often it certifies a candidate, how often it abstains, and the resulting class errors and costs. |
| Freeze a new protocol | Choose the primary outcome, class tolerances, candidate count, compute budget, and comparisons before fresh confirmation. | Development results are explicitly separated from confirmation. |
| Confirm and measure scale | Evaluate fresh datasets, natural label errors where available, several noise mechanisms, and increasing row/feature counts. | Reproducible benefit or cost savings, with losses and limitations reported. |

For development, a small fixed grid of roughly 6–12 complete candidates is enough to expose whether the idea has value. This is a practical starting recommendation, not a completed or registered protocol. Avoid a large search that consumes the compute budget and makes certification harder.

Use the current ten datasets for development and diagnostics. They have already influenced the hypothesis, so they should not be the sole evidence for a new confirmatory claim.

The data roles should remain separate:

```text
Development/training data -> fit judges, propose corrections and subsets
                         -> fit and freeze candidate models
Independent trusted audit -> accept/reject the fixed candidates
Untouched final test data -> report the selected method's final performance
```

The audit is not the final test. For nested evaluation, these roles must be reproduced within each outer training split, with the outer test kept untouched. All learned encoding, imputation, scaling, and selection belong in the corresponding training portion.

## 11. What the new experiments must include

Keep a final CART decision tree as the primary downstream learner first. Compare it with its own uncleaned baseline under matched settings. Add a strong uncleaned boosted-tree comparator to test practical value, and a proper Confident Learning implementation to test cleaning value. Comparators should receive the same permitted audit-label and compute resources. The label-treatment action must be specified: a detector's ranking score is not automatically a full correction method.

Test zero added noise, symmetric noise, class-dependent noise, and instance-dependent noise. Add irrelevant features and redundant features separately. Useful features with corrupted values are a different stress test. On real data, use independently checked errors where possible; original labels alone cannot establish detection truth.

Report accuracy, macro-F1, per-class recall, error-detection precision/recall, replacement-label correctness, number of changed rows, feature count, feature-selection variability, runtime, memory, and fallback frequency. Keep the primary claim narrow and label the other analyses as secondary.

Important tests include proving that the corruption mask never enters a filter, audit labels never enter candidate fitting, feature groups are applied consistently to test data, every test row survives, and the fallback returns the baseline. A deliberately harmful candidate should be rejected in a controlled simulation when the audit has sufficient evidence; this is a diagnostic, not empirical proof of a theorem.

For cross-dataset inference, use dataset-level paired effects and a specified comparison family. Report confidence intervals and practical effect sizes. Do not treat overlapping folds as independent observations or keep changing thresholds until a p-value passes 0.05.

## 12. When to continue, simplify, or stop

Continue if joint treatment gives a repeatable advantage over both component-only alternatives, or meaningful resource savings at the chosen predictive tolerance. A certifying method should also accept useful candidates often enough to matter; always returning the baseline can be valid but unhelpful.

Simplify if conservative correction alone matches the joint method. Drop the feature component if it adds cost without a useful result. If NB contributes no measurable benefit over a DT-only judge under matched budgets, do not preserve the hybrid solely to support the title.

Stop the proposed new-method claim if gains disappear on fresh data, natural errors, or matched comparators. The existing negative and conditional findings remain reportable research outcomes. They do not become invalid because a proposed extension fails.

## 13. How recent work informs the proposal

The literature search behind this guide was a focused review through 9 October 2026. These are source-backed starting points, not an exhaustive novelty search.

- [Nazaretyan, Leser and Kircher, 2025, Artificial Intelligence Review](https://doi.org/10.1007/s10462-025-11293-9): benchmark of mislabeled tabular-data detection. It motivates serious comparator selection and attention to false accusations of noise.
- [Pickler et al., 2025 preprint, revised 2026](https://arxiv.org/abs/2510.16211): separates gathering strategy, disagreement measure, and aggregation. Out-of-fold judging is not automatically the best detector in every setting.
- [Ali, Kang and Seneviratne, September 2026 preprint](https://arxiv.org/html/2609.32976v1): adaptive detector ensembles already exist. Its evaluation uses small subsamples and known corruption budgets for detection F1; it does not settle our proposed large-data, downstream class-error question.
- [Swaroop, Krishnamurti and Wilder, NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/5b3f001200198bb4ecaac0a1ea89dd99-Abstract-Conference.html): feature selection across populations motivates checking class or group harm rather than only average performance.
- [Huang et al., 2025 preprint, revised 2026](https://arxiv.org/abs/2510.14581): selective labeling with statistical guarantees already exists. Controlling accepted-label error does not automatically guarantee the performance of a retrained classifier.
- [Yu, Wei and Jing, July 2026 preprint](https://arxiv.org/html/2607.22985v1): contaminated calibration data complicate error guarantees. Its asymptotic results have assumptions and do not give our heuristic an automatic certificate.
- [Angelopoulos et al., Learn then Test, journal publication June 2025](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk): established risk-control framework. The proposed audit logic must credit this line of work; it is not our theoretical invention.
- [Meinshausen and Buhlmann, stability selection, 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x) and [Northcutt et al., Confident Learning](https://arxiv.org/abs/1911.00068): older foundations that remain necessary comparisons. Their assumptions and guarantees must not be claimed for a different implementation without proof.

The possible contribution is an effective, inexpensive way to construct and select joint DT–NB treatment/feature candidates under explicit constraints. The basic ingredients and the displayed concentration bound are established ideas. Novelty requires a closer prior-art comparison and a demonstrated advance, not a new acronym.

## 14. Repository evidence and validation limits

Current-source checks for this guide included `code/algorithm1.py`, `code/algorithm2.py`, `code/nb.py`, `code/data.py`, and `code/confident_filter.py`. Historical experiment descriptions and values are grounded in:

- [Project defense guide](project-defense-guide.md)
- [E12v2 fixed protocol](e12v2-protocol.md)
- [Current findings](../findings.md)
- [E12v2 dataset-level primary effects](../results/EXP-E12v2_main/primary_by_dataset.csv)
- [E9 metrics](../results/EXP-E9_farid/metrics.csv)

The equations explain the current rules or explicitly proposed extensions. They do not claim that the current code implements the proposed audit or joint search. Exact reproduction of stored runs from current source bytes remains subject to the provenance limits recorded in the defense guide.

No new model experiments or performance guarantees were empirically validated for this document. No model code, manuscript, or saved results were changed. The main unresolved risks are trustworthy-label availability, rare-class audit size, selection cost, and whether the proposed extension offers anything beyond simpler established methods.
