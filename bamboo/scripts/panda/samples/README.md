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
the arms are still there, the line numbers are a release behind.

**What supersedes a sample is the release moving, not the rebuild.**  P1-41
rebuilt the map again from the same corpus at the same commit -- the readers
widened, the source did not -- and `p1-40-walk.json` came through it scoring
better than before (181 of 184 names against 175 of 180) with nothing
fabricated and nothing silent.  A sample survives a rebuild that does not move
a line number.

**Score a sample with the `--source-root` the map was built from.**  Without
it `_resolve_roots(None)` reads the *installed* distribution, which is a
different release from the one the stored map describes, and the audit comes
back smaller rather than wrong -- 120 of 123 instead of 181 of 184, which
reads like a regression and is not one.  The command is part of the baseline:

    score_trace --sample samples/p1-40-walk.json --audit \
                --source-root /path/to/the/release/the/map/names

| sample | seed | drawn from |
|---|---|---|
| `p1-35-walk.json`, `p1-35-verdicts.txt` | not recorded | `panda-server-source 1.0.2` |
| `p1-40-walk.json` | 20260925 | `git:1.0.4-295-gbf2812ba` |

The first row is why this table exists.  This file used to say "the seed is in
the file".  A `review_symptoms` draw does record it; a walk sample is a bare
list of arms and does not, and for `p1-35-walk.json` the seed is not in the
commit message or the round's record either, so that draw cannot be repeated
even against its own map.  Written down here from now on.
