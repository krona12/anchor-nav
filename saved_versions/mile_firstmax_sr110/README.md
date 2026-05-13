# mile first-max SR110 snapshot

This snapshot preserves the pure-math VDD code family used by the run:

- Remote run dir: `/home/chenlin/krona/anchor-nav/output_logs/anchor/mile_all_0.05_0.5/20260510-224053-detailed-msgnav-pure-vdd-height15-rerun1`
- Tmux session at the time: `mile-all-0_05-0_5-pure-vdd-rerun1`
- Exact aligned checkpoint mentioned by the user: count `110`, SR delta `+0.090909`, SPL delta `+0.029592`
- Later exact aligned checkpoint before intervention: count `145`, SR delta `+0.041379`, SPL delta `-0.000088`

Policy preserved here:

- `uses_vlm=false`
- `camera_height=1.50`
- `enable_vvd_replacement=true`
- `prefer_visible_baseline=false`
- `selection_policy=msgnav_first_max_followable_visibility_no_baseline_threshold`

This is kept as the high-SR baseline for further mile/VDD improvements.
