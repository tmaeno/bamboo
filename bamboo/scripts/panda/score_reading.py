"""Throwaway diagnostic: is the function enough to pick the arm?

Not production code.  It exists to answer one design question with a number,
the way ``census_code_map.py`` did -- and the question is the one this round
was nearly shipped without asking.  The read unit was chosen by measuring what
an arm *refers to* (only 35 of 927 refer to nothing outside their own guard, so
the guard is not the unit and the function very nearly is).  That says the
function contains what is needed.  It does not say a reader can use it, and the
last time sufficiency was argued from a measurement rather than tested, the
argument was wrong.

**Ground truth comes from the map, and no network is involved.**  A junction
with several arms records, per arm, the message frame the block writes -- and
production matched those frames for 28 of 28 records in the round that built
them.  Rendering a frame and asking which line wrote it therefore poses the
real question with an exact answer already known.  What it does not reproduce
is production's wording drift between releases; that half was measured
separately and is not what this is for.

Reported as a lower bound.  A local 36B model failing says the input was
insufficient *or* the reader was, and only running the same harness against a
stronger model separates them -- which is why ``--model`` exists.
"""

from __future__ import annotations

import logging
import random
import re
import statistics
import time
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import reading as reading_mod
from bamboo.codemap.models import JunctionNode

logger = logging.getLogger(__name__)

#: What a rendered frame puts in the holes.  Distinct per hole so that a reader
#: matching on the values rather than the wording cannot be told apart from one
#: matching on the wording -- both are legitimate here, since production's line
#: carries both.
_FILLER = ("917", "3", "42.5", "0.18", "7", "120", "12", "0.9")

_ASK_ARM = """\
Below is one Python function from the PanDA server, with line numbers.
At run time it wrote the value {observed!r} to {subject}, and the record it
left carries this message:

    {message}

Which line performed that write?

Answer with exactly two lines:
LINE: <the line number>
WHY: <one sentence>
"""

_ASK_ORIGIN = """\
Below is Python from the PanDA server, with line numbers.

At line {line} the code tests {name}.

On which line does {name} get the value it holds there?

Answer with exactly two lines:
LINE: <the line number>
WHY: <one sentence>
"""

_ANSWER = re.compile(r"LINE:\s*(\d+)", re.IGNORECASE)


def _render(frame: str) -> str:
    """Fill a frame's holes, the way production would."""
    out = frame
    for filler in _FILLER:
        if "{}" not in out:
            break
        out = out.replace("{}", filler, 1)
    return out


def _binding(module, func_line: int, use_line: int, name: str) -> Optional[int]:
    """The line that last set *name* before *use_line*, inside the same function.

    The ground truth for the ``origin`` question, and the reason it is a fair
    one: the map does not supply this, so a reader either finds it in what it
    was handed or does not.
    """
    import ast

    holder = None
    for node in ast.walk(module.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= func_line <= (
            node.end_lineno or node.lineno
        ):
            span = (node.end_lineno or node.lineno) - node.lineno
            if holder is None or span < holder[1]:
                holder = (node, span)
    if holder is None:
        return None
    best = None
    for node in ast.walk(holder[0]):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        for target in targets:
            for found in ast.walk(target):
                if isinstance(found, ast.Name) and found.id == name and node.lineno < use_line:
                    if best is None or node.lineno > best:
                        best = node.lineno
    return best


_IDENT = re.compile(r"[A-Za-z_]\w*")


def _cases(junctions: list[JunctionNode], observed: str) -> list[tuple]:
    """(junction, arm) pairs where *which arm* is a real question.

    Only junctions with more than one arm reaching the value, and only arms
    whose frames are unique within it.  An arm no frame distinguishes is not a
    question a reader can be scored on -- the map cannot answer it either, and
    scoring it would measure the corpus rather than the reading.
    """
    found = []
    for junction in junctions:
        arms = [
            b
            for b in junction.branches
            if b.outcome == observed and b.line is not None and b.messages
        ]
        if len(arms) < 2:
            continue
        for arm in arms:
            mine = set(arm.messages)
            others = {m for b in arms if b is not arm for m in b.messages}
            unique = sorted(mine - others)
            if unique:
                found.append((junction, arm, max(unique, key=len)))
    return found


@click.command("score-reading")
@click.option("--subject", default="JediTaskSpec.status", show_default=True)
@click.option("--observed", default="exhausted", show_default=True)
@click.option(
    "--source-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Build the map from this tree instead of the installed distribution.",
)
@click.option(
    "--input",
    "unit",
    type=click.Choice(["function", "region"]),
    default="function",
    show_default=True,
    help="Ablation: hand over the whole function, or only the arm's own guard.",
)
@click.option(
    "--ask",
    "question",
    type=click.Choice(["arm", "origin"]),
    default="origin",
    show_default=True,
    help=(
        "What to score.  'arm' asks which line wrote the value, which the "
        "message already names -- both units pass it, so it discriminates "
        "nothing.  'origin' asks where a tested name got its value, which is "
        "the thing the measurement says the guard does not contain."
    ),
)
@click.option("--repeat", default=1, show_default=True, help="Runs per case.")
@click.option("--limit", default=0, help="Score at most this many cases (0 = all).")
@click.option("--model", default=None, help="Override LLM_MODEL for this run.")
@click.option("--dry-run", is_flag=True, help="Report the sample and ask nothing.")
def main(
    subject: str,
    observed: str,
    source_root: Optional[Path],
    unit: str,
    question: str,
    repeat: int,
    limit: int,
    model: Optional[str],
    dry_run: bool,
) -> None:
    """Score a reader on picking the arm, against the map's own answer."""
    logging.basicConfig(level=logging.WARNING)
    from bamboo.codemap.panda.plugin import PandaCodeMapPlugin

    plugin = PandaCodeMapPlugin()
    version = plugin.prepare(source_root)
    fragment = plugin.run()
    modules = {m.rel_path: m for m in plugin._modules}

    junctions = [j for j in fragment.junctions if j.subject == subject]
    cases = _cases(junctions, observed)
    if limit:
        cases = cases[:limit]

    click.echo(f"map        {fragment.map_id} @ {version}")
    click.echo(
        f"sample     {len(cases)} arm(s) across "
        f"{len({j.owner for j, _a, _m in cases})} function(s) -- "
        f"only junctions where more than one arm writes {observed!r}"
    )
    if not cases:
        click.echo("nothing to score: no junction writes this value from several named arms")
        return

    prompts = []
    skipped = 0
    for junction, arm, frame in cases:
        module = modules[junction.anchor.file]
        region = reading_mod.region_in(
            module.source,
            module.tree,
            file=junction.anchor.file,
            owner=junction.owner,
            line=arm.line,
            expected_sha=module.blob_sha,
        )
        if region is None:
            continue
        text = region.marked([]) if unit == "function" else _guard(module, arm.line)
        if question == "arm":
            want = arm.line
            ask = _ASK_ARM.format(
                observed=observed, subject=subject, message=_render(frame)
            )
        else:
            # The name the arm's own guard tests, and where it was set.  Chosen
            # from the condition the map recorded rather than from the text, so
            # the question is about the decision and not about whatever token
            # happened to be nearby.
            name = want = None
            for condition in arm.conditions if hasattr(arm, "conditions") else arm.path_condition:
                for candidate in _IDENT.findall(condition):
                    line = _binding(module, arm.line, arm.line, candidate)
                    if line is not None:
                        name, want = candidate, line
                        break
                if name:
                    break
            if name is None:
                skipped += 1
                continue
            ask = _ASK_ORIGIN.format(line=arm.line, name=name)
        prompts.append((junction, arm, want, ask + "\n" + text))

    sizes = [len(p.split()) for _j, _a, _w, p in prompts]
    click.echo(
        f"input      {unit}: median {statistics.median(sizes):.0f} words, "
        f"max {max(sizes)} -- the served context is 262144 tokens"
    )
    if skipped:
        click.echo(
            f"           {skipped} arm(s) left out: no name its condition tests is set "
            "anywhere in the function, so there is no answer to score against"
        )
    if dry_run:
        for junction, arm, want, _p in prompts[:10]:
            click.echo(f"  {junction.owner.split('::')[-1]:<34} arm {arm.line} -> {want}")
        return

    if model:
        import os

        os.environ["LLM_MODEL"] = model
    from bamboo.llm.llm_client import get_extraction_llm, resolve_context_window

    click.echo(f"reader     context window {resolve_context_window()}")
    llm = get_extraction_llm()

    random.Random(0).shuffle(prompts)
    hits: list[int] = []
    elapsed: list[float] = []
    wrong: list[tuple[str, int, Optional[int]]] = []
    for run in range(repeat):
        run_hits = 0
        for junction, _arm, want, prompt in prompts:
            start = time.time()
            try:
                answer = llm.invoke(prompt).content
            except Exception as exc:  # pragma: no cover - a local server going away
                click.echo(f"  reader failed on {junction.owner}: {exc}")
                continue
            elapsed.append(time.time() - start)
            match = _ANSWER.search(str(answer))
            said = int(match.group(1)) if match else None
            if said == want:
                run_hits += 1
            else:
                wrong.append((junction.owner, want, said))
        hits.append(run_hits)
        click.echo(f"run {run + 1}      {run_hits}/{len(prompts)} arms named correctly")

    click.echo(
        f"\nagreement  {statistics.mean(hits):.1f}/{len(prompts)} "
        f"({100 * statistics.mean(hits) / len(prompts):.0f}%) -- a LOWER BOUND on "
        "whether the input suffices, since a weaker reader fails the same way"
    )
    if repeat > 1:
        click.echo(f"spread     {min(hits)}-{max(hits)} across {repeat} runs at temperature 0")
    if elapsed:
        click.echo(f"cost       {statistics.median(elapsed):.1f}s median per case")
    if wrong:
        click.echo(f"\nmissed ({len(wrong)}), first 10 -- classify these by hand:")
        for owner, want, said in wrong[:10]:
            click.echo(f"  {owner.split('::')[-1]:<34} wanted {want}, said {said}")


def _guard(module, line: int) -> str:
    """The arm's own smallest enclosing compound statement, for the ablation."""
    import ast

    best = None
    for node in ast.walk(module.tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)) and node.lineno <= line <= (
            node.end_lineno or node.lineno
        ):
            span = (node.end_lineno or node.lineno) - node.lineno
            if best is None or span < best[1]:
                best = (node, span)
    if best is None:
        return ""
    start, end = best[0].lineno, best[0].end_lineno or best[0].lineno
    return "\n".join(
        f"    {start + offset:5d}  {text}"
        for offset, text in enumerate(module.source.splitlines()[start - 1 : end])
    )


if __name__ == "__main__":
    main()
