import transformers
import argparse
from datasets import load_dataset
import utils
from datetime import datetime
import pandas as pd
from IndicTransTokenizer import IndicProcessor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input dataset")
    parser.add_argument("-B", "--batch_size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    model_name = "ai4bharat/indictrans2-en-indic-1B"
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, trust_remote_code=True
    )
    expdir = f"trained_models/indictrans2/{datetime.now().strftime('%d-%mT%H.%M')}"

    # Data loading
    ip = IndicProcessor(inference=True)
    utils.ip = ip
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    utils.tokenizer = tokenizer
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model
    )

    dataset = load_dataset("csv", data_files=args.data, header=0, split="train")
    dataset = dataset.train_test_split(test_size=0.1, shuffle=False)
    dataset_tkn = dataset.map(utils.preprocess_indictrans, batched=True)

    # Taken from https://huggingface.co/docs/transformers/tasks/translation

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=expdir,
        evaluation_strategy="epoch",
        learning_rate=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.001,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        report_to="none",
        generation_max_length=256,
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tkn["train"],
        eval_dataset=dataset_tkn["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=utils.compute_metrics,
    )

    # trainer.evaluate()
    preds, predlabels, metrics = trainer.predict(
        dataset_tkn["test"], metric_key_prefix="test", max_length=256
    )
    print(metrics)
    decoded_preds = utils.postprocess_indictrans(preds)

    predictions_data = pd.DataFrame(
        {
            "english_INPUT": dataset_tkn["test"]["English"],
            "sanskrit_PRED": decoded_preds,
            "sanskrit_GT": dataset_tkn["test"]["Sanskrit"],
        }
    )

    predictions_data.to_csv(f"{expdir}/generated_poetry.csv")
    import ipdb

    ipdb.set_trace()
