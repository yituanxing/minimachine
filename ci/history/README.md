# Historical CI workflows

This directory preserves GitHub Actions workflows that were useful while
MiniMachine was discovering earlier Linux/BusyBox/runtime frontiers, but are no
longer part of active CI.

Moving a workflow here is intentionally different from deleting it:

- the exact diagnostic recipe remains readable;
- Git history still preserves the run/commit context;
- GitHub no longer registers it as an active workflow;
- current release and performance pipelines stay easier to reason about.

At the first structure-reset checkpoint, 55 historical workflows were moved
here. The active set fell from 98 to 43 permanent/current workflows, plus one
temporary refactor smoke workflow.

The frozen v1 release gates remain active:

- native-dynamic-service-contract.yml
- lua-runtime-hot.yml
- busybox-real-hot.yml
- linux-minimachine-target.yml

Do not move a workflow here merely because it is slow. Archive it only when it
is superseded, tied to an old frontier/run-id/branch, or no longer produces an
input consumed by the current pipeline.
