# The Anti-Psychotic Q-Learner Trick

In brief, we formulate a theory of a satisficing reinforcement learning
agent. We posit that such an agent is desirable from an AI safety perspective,
and offer theoretical arguments for this, with some experimental validation on
a toy model. Finally, we offer this theory as a potential explanation of the
role of the neurotransmitter serotonin in the basal ganglia of mammals.

It should be noted that this paper is inspired in large part by very detailed
and unproven hypotheses on the role of various neurotransmitters and systems in
the mammalian brain. We offer these hypotheses at this point mainly as a source
of inspiration and as an aid to the intuition. We make no claims to having
demonstrated (even inconclusively) the correctness of these hypotheses, but we
feel that the contribution to the AI safety domain is clear and worth pursuing.

We consider this technique to be strong enough to hold a naive Deep Q-type
system, but unable to hold a policy gradients-based system. (But we think the
same trick will have notable stabilizing effects even on policy gradients-based
systems.) It is based on the action mechanism of the class of drugs known as
anti-psychotics. We therefore term it the Anti-Psychotic Q-Learner Trick.
