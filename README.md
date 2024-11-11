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

<img width="500" alt="image" src="https://github.com/user-attachments/assets/9d4a637b-2339-4271-a240-6048e98d04e7">


\\ 


# Explanation 

1. **Input Image (224 * 224 * 3)**:
   - This represents the image input with dimensions \( 224 \times 224 \times 3 \) (height, width, color channels) from **Million-AID** dataset.

2. **Byte Array (M × C)**:
   - The input image is transformed into a **byte array** with dimensions \( M = 224 \times 224 \times 256 \). Here, each pixel is encoded into a higher-dimensional embedding space (256 channels).
   - This corresponds to the **input preprocessor** step in the model: 
     - `Conv2d(3, 256, kernel_size=(1, 1))` converts the 3-channel input to 256 channels.
     - `PerceiverTrainablePositionEncoding` and `positions_projection` add positional information and map to the embedding space.

3. **Latent Array (N × D)**:
   - This is the **latent array** with dimensions \( N = 1024 \) and \( D \) matching the hidden dimension of the model (1024 here).
   - This latent array is initialized and repeatedly processed with **cross-attention** and **latent transformer** layers.
   - This corresponds to the **latent initialization** in the model and is the alike the **hidden state** of RCNN.

4. **Cross-Attention and Latent Transformer Layers**:
   - The **cross-attention** layers map between the input byte array and the latent array by aligning relevant features.
   - The **latent transformer** layers (self-attention and MLP) perform repeated transformations on the latent array.
   - This is represented in the model by:
     - `cross_attention`: The initial cross-attention layers between the byte array and the latent array.
     - `self_attends`: These are stacked PerceiverLayers that refine the latent representations.

5. **Final Cross-Attention and Transformation**:
   - Final cross-attention step aligns the final latent representation with the classification task.
   - This part is handled by the **decoder** in the model (`PerceiverClassificationDecoder`), which includes a final cross-attention layer for decoding the latent representation.

6. **Logits Output**:
   - The latent array is averaged, and the output layer (with dimensions corresponding to the number of classes, here 51) produces logits for classification.
   - This is handled by the **final layer** in `PerceiverBasicDecoder`, which maps from the latent space dimension (1024) to **51 classes** of Million-AID dataset.

To summarize:

- **Input Image**: `input_preprocessor`
- **Byte Array**: `Conv2d`, `position_embeddings`
- **Latent Array**: Latent representation initialized in `PerceiverModel`
- **Cross-Attention and Latent Transformer Layers**: `cross_attention` and `self_attends`
- **Final Cross-Attention and Transformation**: `decoder` and final `cross_attention` in `PerceiverClassificationDecoder`
- **Logits Output**: Final Fully Connected Layer to align to **51 classes** of Million-AID dataset

# Training Results 
- Loss Curve
<img width="700" alt="image" src="https://github.com/user-attachments/assets/86a9f4e1-3006-408d-9295-c4de5587c076">

- Accuracy Curve
<img width="700" alt="image" src="https://github.com/user-attachments/assets/73fa4d93-195c-42a3-880b-d9acc9e4184b">

# Testing Results
![image](https://github.com/user-attachments/assets/3024fb93-0886-4817-a338-8b55375009dd)










Thank you for having a look at this repository. I hope you had a good time and great learning. This work is done by Priyanka Talwar and Danny Savla (collaborator of this repository) to fulfill the course work GNR650 — Advanced Topics in Deep Learning for Image Analysis by Prof. Biplab Banerjee.
