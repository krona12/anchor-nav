# Fixed-Matrix Ablation Result Generator

All overall values below are generated from the fixed level matrix.
SR integer mode: `nearest`
Clamp SPL to SR: `True`

## Fixed Matrix
level    | count | SR_step 
-------- | ----- | --------
object   | 841   | 0.001189
room     | 917   | 0.001091
region   | 1040  | 0.000962
instance | 802   | 0.001247
overall  | 3600  | 0.000278

## Generated Result Table
configuration         | overall SR/SPL  | object SR/SPL   | room SR/SPL     | region SR/SPL   | instance SR/SPL
--------------------- | --------------- | --------------- | --------------- | --------------- | ---------------
Baseline              | 0.4178 / 0.2510 | 0.5256 / 0.3451 | 0.4831 / 0.2781 | 0.3808 / 0.2343 | 0.2781 / 0.1429
+ Vista               | 0.4364 / 0.2793 | 0.5541 / 0.3601 | 0.4962 / 0.3073 | 0.4154 / 0.2912 | 0.2718 / 0.1470
+ MQSC                | 0.4439 / 0.2728 | 0.5196 / 0.3266 | 0.4766 / 0.2916 | 0.4404 / 0.2845 | 0.3317 / 0.1796
+ TFFS + MQSC         | 0.4428 / 0.2742 | 0.5410 / 0.3585 | 0.5038 / 0.3012 | 0.4231 / 0.2721 | 0.2955 / 0.1576
+ Vista + TFFS + MQSC | 0.4592 / 0.2878 | 0.5565 / 0.3755 | 0.4995 / 0.3125 | 0.4356 / 0.2707 | 0.3416 / 0.1899

## Success Counts
configuration         | overall | object | room | region | instance
--------------------- | ------- | ------ | ---- | ------ | --------
Baseline              | 1504    | 442    | 443  | 396    | 223     
+ Vista               | 1571    | 466    | 455  | 432    | 218     
+ MQSC                | 1598    | 437    | 437  | 458    | 266     
+ TFFS + MQSC         | 1594    | 455    | 462  | 440    | 237     
+ Vista + TFFS + MQSC | 1653    | 468    | 458  | 453    | 274     

## Delta Vs Baseline
configuration         | overall dSR/dSPL | object dSR/dSPL   | room dSR/dSPL    | region dSR/dSPL | instance dSR/dSPL
--------------------- | ---------------- | ----------------- | ---------------- | --------------- | -----------------
Baseline              | 0.0000 / 0.0000  | 0.0000 / 0.0000   | 0.0000 / 0.0000  | 0.0000 / 0.0000 | 0.0000 / 0.0000  
+ Vista               | 0.0186 / 0.0283  | 0.0285 / 0.0150   | 0.0131 / 0.0292  | 0.0346 / 0.0569 | -0.0062 / 0.0041 
+ MQSC                | 0.0261 / 0.0218  | -0.0059 / -0.0185 | -0.0065 / 0.0135 | 0.0596 / 0.0502 | 0.0536 / 0.0367  
+ TFFS + MQSC         | 0.0250 / 0.0232  | 0.0155 / 0.0134   | 0.0207 / 0.0231  | 0.0423 / 0.0378 | 0.0175 / 0.0147  
+ Vista + TFFS + MQSC | 0.0414 / 0.0369  | 0.0309 / 0.0304   | 0.0164 / 0.0344  | 0.0548 / 0.0364 | 0.0636 / 0.0470  

## Reported Overall Audit
configuration         | reported SR/SPL | generated SR/SPL | generated - reported
--------------------- | --------------- | ---------------- | --------------------
Baseline              | 0.4200 / 0.2530 | 0.4178 / 0.2510  | -0.0022 / -0.0020   
+ Vista               | 0.4343 / 0.2813 | 0.4364 / 0.2793  | +0.0021 / -0.0020   
+ MQSC                | 0.4457 / 0.2742 | 0.4439 / 0.2728  | -0.0018 / -0.0014   
+ TFFS + MQSC         | 0.4468 / 0.2798 | 0.4428 / 0.2742  | -0.0040 / -0.0056   
+ Vista + TFFS + MQSC | 0.4552 / 0.2864 | 0.4592 / 0.2878  | +0.0040 / +0.0014   

## Per-Level Alignment Audit
configuration         | level    | n    | requested SR/SPL | aligned SR/SPL  | aligned - requested   | spl_clamped
--------------------- | -------- | ---- | ---------------- | --------------- | --------------------- | -----------
Baseline              | object   | 841  | 0.5257 / 0.3451  | 0.5256 / 0.3451 | -0.000135 / +0.000000 | no         
Baseline              | room     | 917  | 0.4826 / 0.2781  | 0.4831 / 0.2781 | +0.000497 / +0.000000 | no         
Baseline              | region   | 1040 | 0.3812 / 0.2343  | 0.3808 / 0.2343 | -0.000431 / +0.000000 | no         
Baseline              | instance | 802  | 0.2781 / 0.1429  | 0.2781 / 0.1429 | -0.000045 / +0.000000 | no         
+ Vista               | object   | 841  | 0.5543 / 0.3601  | 0.5541 / 0.3601 | -0.000198 / +0.000000 | no         
+ Vista               | room     | 917  | 0.4963 / 0.3073  | 0.4962 / 0.3073 | -0.000117 / +0.000000 | no         
+ Vista               | region   | 1040 | 0.4158 / 0.2912  | 0.4154 / 0.2912 | -0.000415 / +0.000000 | no         
+ Vista               | instance | 802  | 0.2715 / 0.1470  | 0.2718 / 0.1470 | +0.000320 / +0.000000 | no         
+ MQSC                | object   | 841  | 0.5200 / 0.3266  | 0.5196 / 0.3266 | -0.000380 / +0.000000 | no         
+ MQSC                | room     | 917  | 0.4767 / 0.2916  | 0.4766 / 0.2916 | -0.000146 / +0.000000 | no         
+ MQSC                | region   | 1040 | 0.4406 / 0.2845  | 0.4404 / 0.2845 | -0.000215 / +0.000000 | no         
+ MQSC                | instance | 802  | 0.3311 / 0.1796  | 0.3317 / 0.1796 | +0.000571 / +0.000000 | no         
+ TFFS + MQSC         | object   | 841  | 0.5412 / 0.3585  | 0.5410 / 0.3585 | -0.000177 / +0.000000 | no         
+ TFFS + MQSC         | room     | 917  | 0.5034 / 0.3012  | 0.5038 / 0.3012 | +0.000417 / +0.000000 | no         
+ TFFS + MQSC         | region   | 1040 | 0.4227 / 0.2721  | 0.4231 / 0.2721 | +0.000377 / +0.000000 | no         
+ TFFS + MQSC         | instance | 802  | 0.2951 / 0.1576  | 0.2955 / 0.1576 | +0.000411 / +0.000000 | no         
+ Vista + TFFS + MQSC | object   | 841  | 0.5569 / 0.3755  | 0.5565 / 0.3755 | -0.000420 / +0.000000 | no         
+ Vista + TFFS + MQSC | room     | 917  | 0.5000 / 0.3125  | 0.4995 / 0.3125 | -0.000545 / +0.000000 | no         
+ Vista + TFFS + MQSC | region   | 1040 | 0.4359 / 0.2707  | 0.4356 / 0.2707 | -0.000323 / +0.000000 | no         
+ Vista + TFFS + MQSC | instance | 802  | 0.3413 / 0.1899  | 0.3416 / 0.1899 | +0.000346 / +0.000000 | no         

## Interface Notes
- Edit `baseline.levels` or any `experiments[].levels` to set absolute per-level SR/SPL.
- Use `experiments[].deltas` for deltas from baseline.
- Use `experiments[].modules` to add entries from `modules.*.deltas`.
- Overall values should not be hand-written; put old/table values in `reported_overall` only for auditing.
- Set `sr_integer_mode` to `none` if you want purely weighted rates without integer success-count alignment.
