## Automatic Sarcasm Detection in Long-Form Forum Comments
**Shared team repository for MIDS W251 Spring 2020 final project**

Automatic sarcasm detection is a difficult problem for machines because the exact same sentence can be interpreted both literally and sarcastically depending on the context and opinions of the author. It is a task that humans have a lot of difficulty detecting as well, with many commenters commonly using the ‘\s’ tag to directly indicate when they are actually being sarcastic on social media. What typically helps humans identify sarcasm is the context that the statement was made in and any incongruity of the sarcastic statement with its context. In our dataset, we started with using the Self-Annotated Reddit Corpus (SARC) dataset (https://nlp.cs.princeton.edu/SARC/2.0/)  and further augmented this data with manually retrieved records. While past papers on sarcasm detection informed our modeling approach, we leveraged recent advances in Transfer Learning and utilized HuggingFace Pre-Trained BERT (https://github.com/huggingface/transformers) Transformer architecture to build our classification model.

----------------------------------------------------------

### Running Training
 
 - Build the docker image with docker/PYTORCH.build file
 - Run the docker container headless using sudo docker/docker_run.sh (Container is named w251-project)
 - Update parameters as desired in run_train.sh file 
 - Launch training : ```sudo docker exec w251-project /bin/bash -c "./run_train.sh" ```


### Running Inference

 - Build the docker image with docker/dockerfile_jetson file
 - Run the docker container headless using sudo docker/docker_run_jetson.sh (Container is named w251-project)
 - Launch prediction app using : 
  ``` sudo docker exec w251-project /bin/bash -c "python preduct.py" ```
  
 
 ----------------------------------------------------------

