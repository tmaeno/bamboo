# Samples the scoring rounds drew

Committed, not sealed.  A sealed holdout was tried once and retired: the
extraction script and the sample file were both gone by the time it was due to
be opened, so "not tuned against these cases" had become unverifiable and the
only honest move was to throw it away.

What is kept here is the draw itself, so a later round can say what was scored
and redraw the same thing.  The procedure is in the tool that wrote it --
`score_trace.py --draw --seed N` for walk cases, `review_symptoms.py --seed N`
for vocabulary terms -- and the seed is in the file.
