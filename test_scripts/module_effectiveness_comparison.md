project_root=/home/chenlin/krona/anchor-nav
tasks_per_episode=5
scan_elapsed=True

Brainstormed Analysis Axes
- coverage: did the module get called often enough to matter, and did it actually apply a change
- correction quality: among valid target-distance pairs, did the proposed target move closer or farther from the goal
- threshold correction: did the module convert target distance across the 1m success-relevant boundary, including 00/01/10/11 cases
- real outcome impact: on exactly matched tasks, how often did the module turn a Baseline failure into success or a success into failure
- negative correction: target_worsened, SR_losses, and 10 cases are tracked separately because average gains can hide regressions
- runtime cost: compare end-to-end task_time_sec against Baseline and, where available, stream module_info elapsed_ms from effectiveness logs
- level sensitivity: repeat the above by object/room/region/instance to find where the module is useful or risky
- routing/composition: for Vista2MQSC, separate MQSC-R1 semantic target usage from VISTA-LS viewpoint usage through module_reason and source fields

Metric Design
- correction call rate: module_hook_called / tasks
- correction application rate: module_hook_applied / tasks and / called
- target distance delta: baseline_target_to_goal_l2 - selected/corrected_target_to_goal_l2; positive means the module moved target closer to goal
- Infinity distances are counted for fix/break/improve/worsen decisions, but finite-distance means use only finite before/after pairs
- target 1m case: 00 outside->outside, 01 outside->inside fix, 10 inside->outside break, 11 inside->inside keep
- real task impact: matched-task SR win/loss vs Baseline, plus SPL/time deltas
- time: end-to-end task_time_sec is the trusted comparable runtime; MQSC-R1/Vista2MQSC module_info elapsed_ms is streamed from effectiveness JSON when available
- VISTA-LS partial note: 0.2-1.0 hit OOM, so its combined rows cover 2436/3600 tasks and are not a complete full-set estimate

Module Effectiveness Overview
experiment | status        | tasks     | call   | apply/tasks | apply/called | helpful | target+ | target- | finite_mean_delta_m | SR_wins | SR_losses | net | dSR     | dSPL     | avg_time | dtime  | time_ratio | module_elapsed
---------- | ------------- | --------- | ------ | ----------- | ------------ | ------- | ------- | ------- | ------------------- | ------- | --------- | --- | ------- | -------- | -------- | ------ | ---------- | --------------
Baseline   | complete      | 3600/3600 | 0.00%  | 0.00%       | 0.00%        | NA      | 0.00%   | 0.00%   | NA                  | 0       | 0         | 0   | +0.00pp | 0.000000 | 141.05s  | 0.00s  | 1.00x      | NA            
+ VISTA-LS | partial_error | 2436/3600 | 95.07% | 83.37%      | 87.69%       | NA      | 26.77%  | 56.61%  | -0.376411           | 303     | 248       | 55  | +2.26pp | 0.023687 | 199.08s  | 40.12s | 2.72x      | NA            
MQSC-R1    | complete      | 3600/3600 | 92.14% | 32.89%      | 35.69%       | 55.24%  | 19.72%  | 15.98%  | 0.129499            | 433     | 358       | 75  | +2.08pp | 0.012730 | 131.64s  | -9.41s | 1.81x      | 2.51s         
Vista2MQSC | complete      | 3600/3600 | 92.36% | 85.50%      | 92.57%       | 35.32%  | 32.69%  | 59.88%  | -0.284956           | 487     | 414       | 73  | +2.03pp | 0.023144 | 213.31s  | 72.26s | 3.15x      | 53.14s        

Target 1m Case And Distance
experiment | valid | finite_pairs | 00   | 01  | 10   | 11   | missing | fix_01 | break_10 | finite_mean_before_l2 | finite_mean_after_l2
---------- | ----- | ------------ | ---- | --- | ---- | ---- | ------- | ------ | -------- | --------------------- | --------------------
Baseline   | 0     | 0            | 0    | 0   | 0    | 0    | 0       | 0      | 0        | NA                    | NA                  
+ VISTA-LS | 2436  | 2316         | 1416 | 28  | 827  | 165  | 0       | 28     | 827      | 3.177391              | 3.553802            
MQSC-R1    | 3317  | 3317         | 1693 | 244 | 146  | 1234 | 0       | 244    | 146      | 3.281217              | 3.151719            
Vista2MQSC | 3325  | 3325         | 1850 | 96  | 1153 | 226  | 0       | 96     | 1153     | 3.261586              | 3.546542            

Matched-Task Impact By Subset
experiment | subset             | tasks | dSR      | dSPL      | wins | losses | net | dtime   | time_ratio
---------- | ------------------ | ----- | -------- | --------- | ---- | ------ | --- | ------- | ----------
+ VISTA-LS | module_applied     | 2031  | +3.40pp  | 0.031881  | 271  | 202    | 69  | 41.64s  | 2.83x     
+ VISTA-LS | module_not_applied | 405   | -3.46pp  | -0.017409 | 32   | 46     | -14 | 32.50s  | 2.15x     
+ VISTA-LS | target_improved    | 652   | -0.77pp  | -0.002407 | 66   | 71     | -5  | 41.35s  | 2.96x     
+ VISTA-LS | target_worsened    | 1379  | +5.37pp  | 0.048093  | 205  | 131    | 74  | 41.78s  | 2.78x     
MQSC-R1    | module_applied     | 1184  | +4.05pp  | 0.025236  | 205  | 157    | 48  | -29.01s | 1.17x     
MQSC-R1    | module_not_applied | 2416  | +1.12pp  | 0.006602  | 228  | 201    | 27  | 0.19s   | 2.12x     
MQSC-R1    | target_improved    | 654   | +20.95pp | 0.128400  | 180  | 43     | 137 | -29.76s | 1.19x     
MQSC-R1    | target_worsened    | 530   | -16.79pp | -0.102064 | 25   | 114    | -89 | -28.08s | 1.15x     
Vista2MQSC | module_applied     | 3078  | +3.80pp  | 0.033084  | 456  | 339    | 117 | 55.94s  | 2.94x     
Vista2MQSC | module_not_applied | 522   | -8.43pp  | -0.035465 | 31   | 75     | -44 | 168.44s | 4.42x     
Vista2MQSC | target_improved    | 1087  | +10.21pp | 0.061651  | 203  | 92     | 111 | 51.46s  | 2.90x     
Vista2MQSC | target_worsened    | 1991  | +0.30pp  | 0.017488  | 253  | 247    | 6   | 58.39s  | 2.96x     

Time Cost
experiment | tasks | avg_task | med_task | p95_task | sum_h  | avg_dtime | med_dtime | p95_dtime | avg_ratio | elapsed_n | elapsed_avg | elapsed_med | elapsed_p95 | elapsed_h
---------- | ----- | -------- | -------- | -------- | ------ | --------- | --------- | --------- | --------- | --------- | ----------- | ----------- | ----------- | ---------
Baseline   | 3600  | 141.05s  | 46.49s   | 487.44s  | 141.05 | 0.00s     | 0.00s     | 0.00s     | 1.00x     | 0         | NA          | NA          | NA          | 0.00     
+ VISTA-LS | 2436  | 199.08s  | 116.67s  | 579.02s  | 134.71 | 40.12s    | 41.97s    | 316.96s   | 2.72x     | 0         | NA          | NA          | NA          | 0.00     
MQSC-R1    | 3600  | 131.64s  | 44.70s   | 442.52s  | 131.64 | -9.41s    | -1.01s    | 174.10s   | 1.81x     | 3317      | 2.51s       | 2.34s       | 3.42s       | 2.31     
Vista2MQSC | 3600  | 213.31s  | 115.42s  | 585.43s  | 213.31 | 72.26s    | 44.41s    | 349.60s   | 3.15x     | 3325      | 53.14s      | 41.32s      | 135.85s     | 49.08    

By Level Effectiveness
experiment | level    | tasks | apply  | target+ | target- | finite_mean_delta_m | dSR     | dSPL      | wins | losses | avg_time | dtime  
---------- | -------- | ----- | ------ | ------- | ------- | ------------------- | ------- | --------- | ---- | ------ | -------- | -------
Baseline   | object   | 841   | 0.00%  | 0.00%   | 0.00%   | NA                  | +0.00pp | 0.000000  | 0    | 0      | 55.46s   | 0.00s  
Baseline   | room     | 917   | 0.00%  | 0.00%   | 0.00%   | NA                  | +0.00pp | 0.000000  | 0    | 0      | 106.14s  | 0.00s  
Baseline   | region   | 1040  | 0.00%  | 0.00%   | 0.00%   | NA                  | +0.00pp | 0.000000  | 0    | 0      | 178.83s  | 0.00s  
Baseline   | instance | 802   | 0.00%  | 0.00%   | 0.00%   | NA                  | +0.00pp | 0.000000  | 0    | 0      | 221.74s  | 0.00s  
+ VISTA-LS | object   | 555   | 85.23% | 20.00%  | 65.23%  | -0.570693           | +1.80pp | 0.020793  | 64   | 54     | 110.17s  | 47.20s 
+ VISTA-LS | room     | 649   | 83.82% | 23.42%  | 60.40%  | -0.444667           | +2.31pp | 0.030555  | 86   | 71     | 165.25s  | 43.55s 
+ VISTA-LS | region   | 705   | 83.55% | 30.07%  | 53.48%  | -0.309381           | +2.41pp | 0.025133  | 88   | 71     | 221.94s  | 15.12s 
+ VISTA-LS | instance | 527   | 80.65% | 33.59%  | 47.06%  | -0.154363           | +2.47pp | 0.016342  | 65   | 52     | 303.79s  | 61.89s 
MQSC-R1    | object   | 841   | 27.71% | 13.81%  | 14.17%  | 0.003565            | -1.66pp | -0.011373 | 69   | 83     | 49.88s   | -5.59s 
MQSC-R1    | room     | 917   | 24.10% | 14.61%  | 10.82%  | 0.039947            | +0.98pp | 0.001959  | 97   | 88     | 86.25s   | -19.89s
MQSC-R1    | region   | 1040  | 38.94% | 27.37%  | 15.76%  | 0.439972            | +6.92pp | 0.042187  | 174  | 102    | 179.28s  | 0.45s  
MQSC-R1    | instance | 802   | 40.52% | 22.93%  | 25.15%  | -0.031465           | +1.00pp | 0.012124  | 93   | 85     | 207.49s  | -14.25s
Vista2MQSC | object   | 841   | 90.25% | 25.45%  | 65.67%  | -0.440434           | -3.33pp | -0.003678 | 85   | 113    | 107.98s  | 52.52s 
Vista2MQSC | room     | 917   | 86.26% | 28.85%  | 62.07%  | -0.382572           | +2.29pp | 0.026959  | 127  | 106    | 172.48s  | 66.34s 
Vista2MQSC | region   | 1040  | 84.90% | 36.33%  | 56.92%  | -0.122762           | +7.02pp | 0.048433  | 180  | 107    | 248.05s  | 69.22s 
Vista2MQSC | instance | 802   | 80.42% | 41.48%  | 54.07%  | -0.194820           | +0.87pp | 0.014116  | 95   | 88     | 325.39s  | 103.66s

Module-Specific Rates
experiment | vistals_called     | vistals_corr       | vistals_vp         | vistals_applied    | mqsc_called        | mqsc_applied       | planner_adj     | planner_prevent | planner_unresolved
---------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | --------------- | --------------- | ------------------
Baseline   | -                  | -                  | -                  | -                  | -                  | -                  | -               | -               | -                 
+ VISTA-LS | 2316/2436 (95.07%) | 2031/2436 (83.37%) | 2032/2436 (83.42%) | -                  | -                  | -                  | 10/2436 (0.41%) | 10/2436 (0.41%) | 13/2436 (0.53%)   
MQSC-R1    | -                  | -                  | -                  | -                  | -                  | -                  | -               | -               | -                 
Vista2MQSC | 3325/3600 (92.36%) | -                  | -                  | 2884/3600 (80.11%) | 3325/3600 (92.36%) | 1248/3600 (34.67%) | -               | -               | -                 

Distributions And VISTA-LS Viewpoint Signals
experiment | signal                    | distribution                                                                                                                                                                          
---------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
+ VISTA-LS | vistals_case              | 11:1080, 00:1048, None:120, 01:114, 10:74                                                                                                                                             
+ VISTA-LS | vistals_target_source     | vista_ls_level_set_target:2022, vistals_no_feasible_component_explicit_keep_baseline:284, None:120, vista_ls_fallback_target:9, baseline_visibility_candidate_explicit_keep_baseline:1
+ VISTA-LS | viewpoint_geo             | n=2436, finite_pairs=2154, improved=1361, worsened=659, mean_before=3.214818, mean_after=3.145296, mean_delta=0.069522, cases=00:1168, 11:1080, 01:114, 10:74                         
Vista2MQSC | module_reason             | vistals_viewpoint_only:1830, mqsc_r1_semantic_target_then_vistals_viewpoint:1054, None:275, baseline_target_kept:247, mqsc_r1_semantic_target_only:194                                
Vista2MQSC | vistals_input_slot_source | pq3d_baseline_object:2077, mqsc_r1_selected_object:1248, None:275                                                                                                                     

Effectiveness JSON Files Scanned For elapsed_ms
experiment | path                                                                                                                                                                                  
---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MQSC-R1    | /home/chenlin/krona/anchor-nav/output_logs/anchor/mqsc_r1_all_0.0_0.2/20260514-144750-detailed/refhm3d_seq_mqsc_r1_refine1_effectiveness_0.0_0.2.json                                 
MQSC-R1    | /home/chenlin/krona/anchor-nav/output_logs/anchor/mqsc_r1_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d_seq_mqsc_r1_refine1_effectiveness_0.2_1.0.json      
Vista2MQSC | /home/chenlin/krona/anchor-nav/output_logs/anchor/vista2mqsc_all_0.0_0.2/20260514-150640-detailed-tmux-gpu0-20260514-150638/refhm3d_seq_vista2mqsc_refine1_effectiveness_0.0_0.2.json 
Vista2MQSC | /home/chenlin/krona/anchor-nav/output_logs/anchor/vista2mqsc_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d_seq_vista2mqsc_refine1_effectiveness_0.2_1.0.json
