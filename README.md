# Investigation of the structure-odor relationship using a Transformer model

## Visualization results of attention matrices
Only a part of visualization results are shown in the paper, more results can be found at 'attention_result' file.

More visualization results corresponding to the figures in paper can be found according to the following table.

|Figure in the paper |file path for more results |memo|
|-|-|-|
|Figure 6|attention_result/substructure prediction/sub-12_30_61-eachdc | TP samples|
|Figure 6|attention_result/substructure prediction/sub-12_30_61-neg | TN samples|
|Figure 6|attention_result/substructure prediction/sub-12_30_61-neg-last4 | TN samples with both 2 substructures|
|Figure 7 (2 decoder layers)|attention_result/substructure prediction/sub-12_30_62-eachdc|TP samples |
|Figure 7 (3 decoder layers)|attention_result/substructures/sub-12_30_63-eachdc|TP samples |
|Figure 7 (4 decoder layers)|attention_result/substructures/sub-12_30_64-eachdc|TP samples |
|Figure 8 (1 decoder layers)|attention_result/substructures/sub-12_30_61-eachdc|TP samples |
|Figure 8 (2 decoder layers)|attention_result/substructures/sub-12_30_62|TP samples |
|Figure 8 (3 decoder layers)|attention_result/substructures/sub-12_30_63|TP samples |
|Figure 8 (4 decoder layers)|attention_result/substructures/sub-12_30_64|TP samples |
|Figure 9 |attention_result/OD prediction/od-dc2|TP samples|
|Figure 10|attention_result/OD prediction/attn_repeat100-tp|TP samples|
|Figure 11|attention_result/OD prediction/attn_repeat100-tn|TN samples screened by substructures constraints|

## Experimental code
'experiment.ipynb' is an example to run the program that used to predict odor descriptors.

'mainCode.py' is the code that imported by 'experiment.ipynb'. Remember to change all the file path appear in this file to your local file path if you want to use this code.

'Transformer_mol3.py' is the code of proposed model mentioned in paper, and it is called by Transformer2OD_tada.modelBuild2 in 'mainCode.py'

The version of Tensorflow used in this work is 2.8.