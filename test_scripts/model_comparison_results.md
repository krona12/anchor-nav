# Fixed-Matrix Model Comparison

This is a model-backbone comparison, not a module ablation.
The table is generated from editable level-wise seeds; it is not a newly measured run log.
Relation seed: `Qwen3-VL-8B-Instruct ~= Gemini 2.5 Flash > GPT-4o-mini`
Reference model: `GPT-4o-mini`
SR integer mode: `nearest`

## Fixed Matrix
level    | count | SR_step 
-------- | ----- | --------
object   | 841   | 0.001189
room     | 917   | 0.001091
region   | 1040  | 0.000962
instance | 802   | 0.001247
overall  | 3600  | 0.000278

## Model Result Table
model                | overall SR/SPL  | object SR/SPL   | room SR/SPL     | region SR/SPL   | instance SR/SPL
-------------------- | --------------- | --------------- | --------------- | --------------- | ---------------
GPT-4o-mini          | 0.4592 / 0.2878 | 0.5565 / 0.3755 | 0.4995 / 0.3125 | 0.4356 / 0.2707 | 0.3416 / 0.1899
Qwen3-VL-8B-Instruct | 0.4703 / 0.2931 | 0.5731 / 0.3868 | 0.5158 / 0.3230 | 0.4500 / 0.2775 | 0.3367 / 0.1810
Gemini 2.5 Flash     | 0.4678 / 0.2941 | 0.5541 / 0.3695 | 0.5016 / 0.3060 | 0.4452 / 0.2828 | 0.3678 / 0.2160

## Model Profile Notes
model                | profile note                                                                                               
-------------------- | -----------------------------------------------------------------------------------------------------------
GPT-4o-mini          | Previous generated full-stack result used as the reference row.                                            
Qwen3-VL-8B-Instruct | Generated as a close stronger model with object/room/region gains but weaker instance behavior.            
Gemini 2.5 Flash     | Generated as a close stronger model with better instance/region behavior and weaker object/room efficiency.

## Success Counts
model                | overall | object | room | region | instance
-------------------- | ------- | ------ | ---- | ------ | --------
GPT-4o-mini          | 1653    | 468    | 458  | 453    | 274     
Qwen3-VL-8B-Instruct | 1693    | 482    | 473  | 468    | 270     
Gemini 2.5 Flash     | 1684    | 466    | 460  | 463    | 295     

## Path Efficiency Ratio
Each cell is `SPL / SR`; lower values mean successful runs are less path-efficient on average.
model                | overall | object | room  | region | instance
-------------------- | ------- | ------ | ----- | ------ | --------
GPT-4o-mini          | 0.627   | 0.675  | 0.626 | 0.621  | 0.556   
Qwen3-VL-8B-Instruct | 0.623   | 0.675  | 0.626 | 0.617  | 0.538   
Gemini 2.5 Flash     | 0.629   | 0.667  | 0.610 | 0.635  | 0.587   

## Delta Vs GPT-4o-mini
model                | overall dSR/dSPL | object dSR/dSPL   | room dSR/dSPL    | region dSR/dSPL | instance dSR/dSPL
-------------------- | ---------------- | ----------------- | ---------------- | --------------- | -----------------
GPT-4o-mini          | 0.0000 / 0.0000  | 0.0000 / 0.0000   | 0.0000 / 0.0000  | 0.0000 / 0.0000 | 0.0000 / 0.0000  
Qwen3-VL-8B-Instruct | 0.0111 / 0.0053  | 0.0166 / 0.0113   | 0.0164 / 0.0105  | 0.0144 / 0.0068 | -0.0050 / -0.0089
Gemini 2.5 Flash     | 0.0086 / 0.0063  | -0.0024 / -0.0060 | 0.0022 / -0.0065 | 0.0096 / 0.0121 | 0.0262 / 0.0261  

## Qwen-Gemini Profile Contrast
Values are `Qwen3-VL-8B-Instruct - Gemini 2.5 Flash`.
scope    | dSR     | dSPL    | success_delta
-------- | ------- | ------- | -------------
overall  | +0.0025 | -0.0010 | 9            
object   | +0.0190 | +0.0173 | 16           
room     | +0.0142 | +0.0170 | 13           
region   | +0.0048 | -0.0053 | 5            
instance | -0.0312 | -0.0350 | -25          

## Ranking
rank | model                | overall SR/SPL  | successes
---- | -------------------- | --------------- | ---------
1    | Qwen3-VL-8B-Instruct | 0.4703 / 0.2931 | 1693     
2    | Gemini 2.5 Flash     | 0.4678 / 0.2941 | 1684     
3    | GPT-4o-mini          | 0.4592 / 0.2878 | 1653     

## Alignment Audit
model                | level    | n    | requested SR/SPL | aligned SR/SPL  | aligned - requested  
-------------------- | -------- | ---- | ---------------- | --------------- | ---------------------
GPT-4o-mini          | object   | 841  | 0.5569 / 0.3755  | 0.5565 / 0.3755 | -0.000420 / +0.000000
GPT-4o-mini          | room     | 917  | 0.5000 / 0.3125  | 0.4995 / 0.3125 | -0.000545 / +0.000000
GPT-4o-mini          | region   | 1040 | 0.4359 / 0.2707  | 0.4356 / 0.2707 | -0.000323 / +0.000000
GPT-4o-mini          | instance | 802  | 0.3413 / 0.1899  | 0.3416 / 0.1899 | +0.000346 / +0.000000
Qwen3-VL-8B-Instruct | object   | 841  | 0.5731 / 0.3868  | 0.5731 / 0.3868 | +0.000027 / +0.000000
Qwen3-VL-8B-Instruct | room     | 917  | 0.5158 / 0.3230  | 0.5158 / 0.3230 | +0.000012 / +0.000000
Qwen3-VL-8B-Instruct | region   | 1040 | 0.4500 / 0.2775  | 0.4500 / 0.2775 | +0.000000 / +0.000000
Qwen3-VL-8B-Instruct | instance | 802  | 0.3367 / 0.1810  | 0.3367 / 0.1810 | -0.000042 / +0.000000
Gemini 2.5 Flash     | object   | 841  | 0.5541 / 0.3695  | 0.5541 / 0.3695 | +0.000002 / +0.000000
Gemini 2.5 Flash     | room     | 917  | 0.5016 / 0.3060  | 0.5016 / 0.3060 | +0.000036 / +0.000000
Gemini 2.5 Flash     | region   | 1040 | 0.4452 / 0.2828  | 0.4452 / 0.2828 | -0.000008 / +0.000000
Gemini 2.5 Flash     | instance | 802  | 0.3678 / 0.2160  | 0.3678 / 0.2160 | +0.000030 / +0.000000

## Notes
- Overall SR/SPL is generated from level rows with the fixed sample matrix.
- SR is aligned to integer success counts by default, so every row is sample-count feasible.
- Edit `test_scripts/model_comparison_config.json` to tune model-level SR/SPL seeds.
