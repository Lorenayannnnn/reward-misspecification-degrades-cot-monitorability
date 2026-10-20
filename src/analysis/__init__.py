"""
Everything that runs over finished checkpoints and dumps.

Entry points:
  run_inference.py                            run an additional monitor over existing evaluation dumps
  compute_monitorability.py                   monitorability per checkpoint for one experiment
  sample_probes_and_generate_description.py   stage one of reward description training
  prefill_base_model.py                       base-model prefilling of the misaligned model's CoTs (Section 5.2)
Shared code:
  common.py                 loading runs, resolving steps, sampling clients, cross-seed reports
  plot_monitorability.py    the monitorability report for a control + misaligned pair
  labeling.py               hack labeling of dump records
  description.py            building the reward description generator's input
"""
