import torch

class Prompt_Ensemble():
    def __init__(self, cla_len, tokenizer):
        super().__init__()
      
        self.special_token = '<|class|>'   # special_token denotes the learnable product category

        #self.prompt_templates_abnormal = ["a photo of damaged <|class|>.","a flawed photo of <|class|>.", "a broken photo of <|class|>.", "the photo of damaged <|class|>.","the damaged photo of <|class|>.","a damaged photo of <|class|>.", "<|class|> with defect."]
        #self.prompt_templates_normal = ["a photo of perfect <|class|>.", "a unblemished photo of <|class|>.", "a good photo of <|class|>.", "the photo of perfect <|class|>.","the perfect photo of <|class|>.","a perfect photo of <|class|>.", "<|class|> without defect."]

        #self.prompt_templates_abnormal = ["a photo of flawed <|class|>"]
        #self.prompt_templates_normal = ["a photo of perfec <|class|>"]


        #self.prompt_templates_abnormal = ["a photo of a imperfect <|class|>"]
        #self.prompt_templates_normal = ["a photo of a flawless <|class|>"]

        #self.prompt_templates_abnormal = ["a photo of a abnormal <|class|>."]
        #self.prompt_templates_normal = ["a photo of a normal <|class|>."]


        #self.prompt_templates_abnormal = ["a photo of a damaged <|class|>", "This is a damaged photo of <|class|>", "a damaged photo of <|class|>", "a photo of a <|class|> with defect", "It is a photo of a <|class|> with damage"]
        #self.prompt_templates_normal = ["a photo of a good <|class|>", "This is a good photo of <|class|>", "a good photo of <|class|>", "a photo of a <|class|> without defect", "It is a photo of a <|class|> without damage"]


        #self.prompt_templates_abnormal = ["There is a damaged <|class|> in the photo", "It is a damaged, flawed and broken picture of <|class|>"]
        #self.prompt_templates_normal = ["There is not a damaged <|class|> in the photo", "It is a good, perfect and pristine picture of <|class|>"]



        self.prompt_templates_abnormal = ["a photo of a damaged <|class|>"]
        self.prompt_templates_normal = ["a photo of a good <|class|>"]

        self.tokenizer =  tokenizer


    
    def forward_ensemble(self, model, vison_feature, device, prompt_id = 0):

        prompted_sentence_normal = self.tokenizer(self.prompt_templates_normal).to(device)
        prompted_sentence_abnormal = self.tokenizer(self.prompt_templates_abnormal).to(device)
        normal_embeddings = model.encode_text(prompted_sentence_normal, vison_feature) 
        abnormal_embeddings = model.encode_text(prompted_sentence_abnormal, vison_feature)
        text_prompts = torch.cat([normal_embeddings, abnormal_embeddings], dim =1)

        return text_prompts







        












        

    


        