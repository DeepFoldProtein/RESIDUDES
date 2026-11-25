---
name: residudes-ops-safety
description: How to run work on the DeepFold cluster without hurting it or other people's jobs - submission, monitoring, pulling results, storage. Use whenever you submit, watch or pull prediction or QA jobs (dossier stages INPUTS_READY and RUNNING).
---

# Operating on the cluster

Each rule here cost the team something in CASP17.

## Submit

- **Heavy work goes through Slurm `sbatch`, or the pueue queues of the B200 pods** (residudes-engine knows both). Never on the login node.
- **Record every job** with `residudes run-record T --job <id> --what "<model, variant, seeds>" --scheduler slurm|pueue`. Slurm jobs and local pueue tasks are polled; a task on a remote pod is not, so close it with `residudes run-record T --job <id> --scheduler pueue --state done` once its results are pulled.
- **pueue tags must be unique.** The remote runner wipes `$OUTPUT` at start, so resubmitting the same tag deleted in-flight seeds. Check `pueue status` before resubmitting.
- **Stay inside the GPU budget** of the dossier. Ask (`NEEDS HUMAN`) before exceeding it.
- **Plan sampling in stages.** Run a few seeds, then expand only where the pool is still uncertain. Easy targets saturate by about 25 models; the hard ones (amyloids, lattices, empty MSAs) are where thousands pay off.

## Watch

- **One watcher, not one per job.** A watcher per job (2,131 of them) once put 988 pueue processes into D state and froze GPFS. Poll at most once a minute.
- **A job "Running" with a GPU busy and a log that stops changing is hung.** Kill it and pull the partial results.
- **AF3 can die silently.** Check exit codes and count models per predictor after every pull. Known causes:
  - a template CIF without `_entity_poly_seq`;
  - multimer templates combined with a ligand;
  - more than ~4,500 tokens.

## Storage and files

- **Never run `find`, `du` or `ls -R` on `/`, `/gpfs`, `/store` or other shared roots.** Scope to the target directory.
- **Check free space and quota before large writes.** GPFS has been 97 % full and the per-user quota exhausted; put new work under `/data`.
- **Never delete recursively.** Move to a trash directory and tell the user.
- **A SLURM epilog kills every process of yours on a node** when your last job there ends. Do not leave daemons on compute nodes.
- **When a pod's pueued restarts,** source its env file first, or its state directory falls back to an unwritable home.
