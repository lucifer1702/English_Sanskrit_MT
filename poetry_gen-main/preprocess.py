'''used to preprocess the dataset to get dataframe for ift'''
import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransToolkit import IndicProcessor


def transliterate_to_sanskrit(df):
    """
    Transliterate English text in a pandas DataFrame column to Sanskrit using indictrans2 with GPU acceleration.
    """
    model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    ip = IndicProcessor(inference=True)
    src_lang, tgt_lang = "eng_Latn", "san_Deva"
    batch = ip.preprocess_batch(
        df,
        src_lang=src_lang,
        tgt_lang=tgt_lang)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Tokenize the sentences and generate input encodings
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE) 
    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=512,
            num_beams=5,
            num_return_sequences=1,
            )

# Decode the generated tokens into text
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )   
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    dataframe = pd.DataFrame({
        "san_answer":translations
    })
    return dataframe


if __name__ == "__main__":
    df = pd.read_csv('ift_data_sample.csv')
    data = df["answers"]
    data_str = data.tolist()
    
    sample =[]
    for i in range(0,50):
          sample.append(data_str[i])

    # print(sample)
    # Perform transliteration
    result = transliterate_to_sanskrit(sample)
    print("\nSample results:")
    print(result.head())
    result.to_csv("translated.csv",index=False)
 
    






