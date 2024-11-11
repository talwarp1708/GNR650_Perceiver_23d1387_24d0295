# GNR650_Perceiver_23d1387_24d0295
## Paper

- Title: Perceiver: General Perception with Iterative Attention
- Authors: Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, Joao Carreira
- Link: https://arxiv.org/pdf/2103.03206
- Tags: ViT, Transformers
- Year: 2021

# Installation 
- Download the code base: git clone https://github.com/talwarp1708/GNR650_Perceiver_23d1387_24d0295.git

- Install dependencies
pip install -r requirements.txt
or Uncomment the first line as mentioned in the ipynb file and execute 
!pip install torchgeo perceiver-pytorch evaluate datasets 

# Usage
- Update the root folder in Perceiver_23D1387_24D0295.ipynb/.py file, which contains the train.zip file of CustomMillionAID train_ds = CustomMillionAID(root='./data', task="multi-class", split="train")

- Update the save_dir to save the models: save_dir  = 'your directory'

- Update the output_filename in the train() to save the logs per batch execution: output_filename = 'Your file path'

# Execute 
- Execute the file .py or .ipynb file

# Results
- View the results in the output_filename for loss and accruacy results during training. The validation results are available on command line

# Model Architecture
<img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/91e9fdd2-ed7c-4fae-943a-d8e861b9eb92">

# Training Results 
- Loss Curve
<img width="700" alt="image" src="https://github.com/user-attachments/assets/86a9f4e1-3006-408d-9295-c4de5587c076">

- Accuracy Curve
<img width="700" alt="image" src="https://github.com/user-attachments/assets/73fa4d93-195c-42a3-880b-d9acc9e4184b">

# Testing Results
![image](https://github.com/user-attachments/assets/3024fb93-0886-4817-a338-8b55375009dd)










Thank you for having a look at this repository. I hope you had a good time and great learning. This work is done by Priyanka Talwar and Danny Savla (collaborator of this repository) to fulfill the course work GNR650 — Advanced Topics in Deep Learning for Image Analysis by Prof. Biplab Banerjee.
