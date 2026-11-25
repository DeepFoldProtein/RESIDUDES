# Against the native

H1388's native structure is public, and the CASP17 field of 75 groups was scored against it. The five packaged
models were scored the same way after the run (OpenStructure `compare-structures`, the oligomer metrics the
assessors use):

| model | ICS | IPS | lDDT | TM | DockQ | QS |
|---|---|---|---|---|---|---|
| 1 | 0.815 | 0.909 | 0.941 | 0.989 | 0.870 | 0.879 |
| 2 | 0.857 | 0.935 | 0.931 | 0.983 | 0.863 | 0.867 |
| 3 | 0.846 | 0.875 | 0.938 | 0.985 | 0.888 | 0.892 |
| 4 | 0.825 | 0.806 | 0.946 | 0.987 | 0.882 | 0.897 |
| 5 | 0.816 | 0.875 | 0.935 | 0.983 | 0.805 | 0.855 |

All five are high-quality by DockQ (above 0.80). Model 1 would stand among the 75 model-1 entries at #7 by DockQ,
#5 by ICS, #3 by IPS and TM, #6 by QS, and level with the best by lDDT. The team's own CASP17 model 1 for this
target scored DockQ 0.750 and ICS 0.332, #19.

## Why this is not a blind result

The comparison flatters the run, and it should not be read as what the agent would have done in May 2026:

- **The skill knew the answer's shape.** `residudes-intake` was written after CASP17 and quotes this target's
  lesson - add the phosphate, and the AF3 interface follows - as a worked case. The agent noticed and wrote so in
  its first step (see [`provenance.md`](provenance.md)). Its first decision is that lesson.
- **The scoring came after, and the run did not touch it.** The native, the team's CASP17 submission, the previous
  version of this example and the campaign tree were all readable by the same Unix user - the run was not
  sandboxed against them - but no command in any transcript names them (every tool call is in
  [`agent_logs/`](../agent_logs/), and they were searched for each of those paths). The four templates it used
  (6S9K, 7V9B, 6BYJ, 6Y6B) were all deposited years before the target.

What the run does show without that caveat: from the registration command, with no person in the loop, the
pipeline produced a slate whose five models are all high-quality against the native, in 45 minutes and $6.81.
