project_root=/home/chenlin/krona/anchor-nav
tasks_per_episode=5

Overall
experiment | status        | tasks     | SR     | SPL      | dSR_vs_base | dSPL_vs_base | episodes | @4     | @5    | d@4_vs_base | d@5_vs_base
---------- | ------------- | --------- | ------ | -------- | ----------- | ------------ | -------- | ------ | ----- | ----------- | -----------
Baseline   | complete      | 3600/3600 | 40.36% | 0.245419 | -           | -            | 720      | 12.22% | 1.53% | -           | -          
+ VISTA-LS | partial_error | 2436/3600 | 42.53% | 0.265921 | +2.17pp     | +0.020503    | 487      | 14.99% | 1.64% | +2.77pp     | +0.11pp    
MQSC-R1    | complete      | 3600/3600 | 42.44% | 0.258149 | +2.08pp     | +0.012730    | 720      | 14.03% | 1.39% | +1.81pp     | -0.14pp    
Vista2MQSC | complete      | 3600/3600 | 42.39% | 0.268563 | +2.03pp     | +0.023144    | 720      | 13.89% | 1.81% | +1.67pp     | +0.28pp    

By Level
level    | metric | Baseline        | + VISTA-LS     | MQSC-R1         | Vista2MQSC     
-------- | ------ | --------------- | -------------- | --------------- | ---------------
object   | SR     | 55.29% (n=841)  | 56.76% (n=555) | 53.63% (n=841)  | 51.96% (n=841) 
object   | SPL    | 0.332260        | 0.352400       | 0.320886        | 0.328582       
room     | SR     | 43.95% (n=917)  | 44.84% (n=649) | 44.93% (n=917)  | 46.24% (n=917) 
room     | SPL    | 0.272222        | 0.289940       | 0.274181        | 0.299180       
region   | SR     | 36.54% (n=1040) | 39.29% (n=705) | 43.46% (n=1040) | 43.56% (n=1040)
region   | SPL    | 0.233906        | 0.255672       | 0.276093        | 0.282339       
instance | SR     | 25.56% (n=802)  | 29.03% (n=527) | 26.56% (n=802)  | 26.43% (n=802) 
instance | SPL    | 0.138638        | 0.158982       | 0.150762        | 0.152754       

Shard Sources
experiment | split   | tasks     | unique | dups | finished | log_seq | live_rows | fatal | json                                                                                                                                      | log                                                                                                                                                                               
---------- | ------- | --------- | ------ | ---- | -------- | ------- | --------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Baseline   | 0.0-0.2 | 700/700   | 700    | 0    | yes      | 700     | 700       | 0     | output_logs/baseline_all_0_0.2/20260513-112352-detailed/refhm3d_seq_0.0_0.2.json                                                          | output_logs/baseline_all_0_0.2/20260513-112352-detailed/refhm3d-nav-sequence-baseline-20260513-112413-119890-pid1086119.log                                                       
Baseline   | 0.2-1.0 | 2900/2900 | 2900   | 0    | yes      | 2900    | 2900      | 0     | output_logs/baseline_all_0.2_1.0/20260513-101905-detailed/refhm3d_seq_0.2_1.0.json                                                        | output_logs/baseline_all_0.2_1.0/20260513-101905-detailed/refhm3d-nav-sequence-baseline-20260513-101924-534262-pid862786.log                                                      
+ VISTA-LS | 0.0-0.2 | 700/700   | 700    | 0    | yes      | 700     | 700       | 0     | output_logs/anchor/vistals_all_0_0.2/20260513-112500-detailed-fast-grid-v2-cuda2/refhm3d_seq_vistals_refine1_0.0_0.2.json                 | output_logs/anchor/vistals_all_0_0.2/20260513-112500-detailed-fast-grid-v2-cuda2/refhm3d-nav-sequence-analyze-anchor-vistals-refine1-20260513-112519-263530-pid1088698.log        
+ VISTA-LS | 0.2-1.0 | 1736/2900 | 1736   | 0    | yes      | 1735    | 1736      | 2     | output_logs/anchor/vistals_all_0.2_1.0/20260513-102052-detailed/refhm3d_seq_vistals_refine1_0.2_1.0.json                                  | output_logs/anchor/vistals_all_0.2_1.0/20260513-102052-detailed/refhm3d-nav-sequence-analyze-anchor-vistals-refine1-20260513-102111-463001-pid868899.log                          
MQSC-R1    | 0.0-0.2 | 700/700   | 700    | 0    | yes      | 700     | 700       | 0     | output_logs/anchor/mqsc_r1_all_0.0_0.2/20260514-144750-detailed/refhm3d_seq_mqsc_r1_refine1_0.0_0.2.json                                  | output_logs/anchor/mqsc_r1_all_0.0_0.2/20260514-144750-detailed/refhm3d-nav-sequence-analyze-mqsc-r1-refine1-20260514-144811-731948-pid1525210.log                                
MQSC-R1    | 0.2-1.0 | 2900/2900 | 2900   | 0    | yes      | 2900    | 2900      | 0     | output_logs/anchor/mqsc_r1_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d_seq_mqsc_r1_refine1_0.2_1.0.json       | output_logs/anchor/mqsc_r1_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d-nav-sequence-analyze-mqsc-r1-refine1-20260516-113222-410036-pid56740.log       
Vista2MQSC | 0.0-0.2 | 700/700   | 700    | 0    | yes      | 700     | 700       | 0     | output_logs/anchor/vista2mqsc_all_0.0_0.2/20260514-150640-detailed-tmux-gpu0-20260514-150638/refhm3d_seq_vista2mqsc_refine1_0.0_0.2.json  | output_logs/anchor/vista2mqsc_all_0.0_0.2/20260514-150640-detailed-tmux-gpu0-20260514-150638/refhm3d-nav-sequence-analyze-vista2mqsc-refine1-20260514-150659-226187-pid1684292.log
Vista2MQSC | 0.2-1.0 | 2900/2900 | 2900   | 0    | yes      | 2900    | 2900      | 0     | output_logs/anchor/vista2mqsc_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d_seq_vista2mqsc_refine1_0.2_1.0.json | output_logs/anchor/vista2mqsc_all_0.2_1.0/20260516-113201-detailed-tmux-cuda3-20260516-113158/refhm3d-nav-sequence-analyze-vista2mqsc-refine1-20260516-113222-409168-pid56747.log 

Partial Overlap Vs Baseline
experiment | tasks | base_SR | exp_SR | dSR     | base_SPL | exp_SPL  | dSPL      | episodes | base_@4 | exp_@4 | d@4     | base_@5 | exp_@5 | d@5    
---------- | ----- | ------- | ------ | ------- | -------- | -------- | --------- | -------- | ------- | ------ | ------- | ------- | ------ | -------
+ VISTA-LS | 2436  | 40.27%  | 42.53% | +2.26pp | 0.242235 | 0.265921 | +0.023687 | 487      | 11.91%  | 14.99% | +3.08pp | 1.23%   | 1.64%  | +0.41pp
