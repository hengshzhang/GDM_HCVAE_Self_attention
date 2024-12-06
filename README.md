# GDM_HCVAE_Self_attention
The code for this study was developed based on Python 3.7 , and the folder where the code resides contains the following code files: 
1)	class_D_HCVAE.py:  This file is mainly used to define the proposed model in our study, and the model named as Discrete Hierarchical Conditional Variational Auto-Encoder.
2)	Discrete_HCVAE_train.py: This file is used to train the model defined in this study, and the data used is located in the folder “New data4”.
3)	Discrete_HCVAE_cross_validation_1.py, 
Discrete_HCVAE_cross_validation_2.py, Discrete_HCVAE_cross_validation_3.py, Discrete_HCVAE_cross_validation_2.py, Discrete_HCVAE_cross_validation_5.py, these 5 files correspond to the 5-fold cross-validation when generating the consistency matrix using the DHCVAE model. 
4)	Discrete_HCVAE_self_generation.py: This file is the complete large-scale group decision-making process, which can be executed to obtain the final decision results, as well as the corresponding data analysis results.

In the project, the folder “New data4” contains the data used for the project and “Discrete HCVAE_model_save” contains the trained model. All files included in the project can be run under PyCharm 2022.1.3 (Community Edition) no modification or configuration required,  and the brief descriptions of the functions are included in the code files.  
