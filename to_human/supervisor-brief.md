# Short explanation for the supervisor

“We began with Farid's two methods: NB removes rows it misclassifies, and a
DT selects and weights attributes. Our earlier tests found that deleting
misclassified rows often removes hard but useful examples, and cleaning
before cross-validation inflates accuracy. Combining the stages did not
solve that problem.

In this run, we tested a more cautious strategy: NB and DT independently
judge each training row without training on it, and we correct its label
only if both confidently agree on the same replacement. We injected known
training-label errors so we could measure detection as well as classification.

Across ten datasets, the strategy gained 2.52 macro-F1 points at 20% added
noise, but the main statistical test was p=0.084, so we cannot claim overall
superiority. Gains were strongest when the final classifier was a DT; some
datasets were harmed. The contribution is a reproducible evaluation of when
conservative DT–NB agreement helps and when it fails—not a universally safe
noise cleaner.”

## If asked for the next research step

Study the decision-tree benefit and low-noise failure cases with a new locked
protocol and independent evidence. Do not select thresholds after inspecting
these same test results. Attribute weighting, natural-noise detection, and
broad MLP improvement remain separate unanswered questions.
