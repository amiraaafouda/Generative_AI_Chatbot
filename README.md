## Generative_AI_Chatbot
# Interactively run the model      

      :model_name=117M : String, which model to use    
      :seed=None : Integer seed for random number generators, fix seed to reproduce results     
      :nsamples=1 : Number of samples to return total      
      :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.     
      :length=None : Number of tokens in generated text, if None (default), is determined by model hyperparameters     
      :temperature=1 : Float value controlling randomness in boltzmann distribution. 
                       Lower temperature results in less random completions. 
                       As the temperature approaches zero, the model will become deterministic and repetitive. 
                       Higher temperature results in more random completions.
      :top_k=0 : Integer value controlling diversity. 1 means only 1 word is considered for each step (token), 
                 resulting in deterministic completions, while 40 means 40 words are considered at each step. 
                 0 (default) is a special setting meaning no restrictions. 40 generally is a good value.
