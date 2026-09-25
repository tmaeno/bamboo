# Samples the scoring rounds drew

Committed, not sealed.  A sealed holdout was tried once and retired: the
extraction script and the sample file were both gone by the time it was due to
be opened, so "not tuned against these cases" had become unverifiable and the
only honest move was to throw it away.

What is kept here is the draw itself, so a later round can say what was scored
and redraw the same thing.  The procedure is in the tool that wrote it --
`score_trace.py --draw --seed N` for walk cases, `review_symptoms.py --seed N`
for vocabulary terms.

**A draw is only repeatable against the map it was drawn from.**  Both tools
stratify over the stored map, so the seed alone does not reproduce it: the
version stamp does the other half.  `p1-35-walk.json` audited against the map
rebuilt in P1-40 covers 98 of 102 names where it had covered 147 of 152 --
the arms are still there, the line numbers are a release behind.  So a sample
is superseded when the stored map is rebuilt, and the old one is kept as the
record of what was scored rather than re-run.

| sample | seed | drawn from |
|---|---|---|
| `p1-35-walk.json`, `p1-35-verdicts.txt` | not recorded | `panda-server-source 1.0.2` |
| `p1-40-walk.json` | 20260925 | `git:1.0.4-295-gbf2812ba` |

The first row is why this table exists.  This file used to say "the seed is in
the file".  A `review_symptoms` draw does record it; a walk sample is a bare
list of arms and does not, and for `p1-35-walk.json` the seed is not in the
commit message or the round's record either, so that draw cannot be repeated
even against its own map.  Written down here from now on.
