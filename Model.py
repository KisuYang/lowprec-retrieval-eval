import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification)


class Model:

    def __init__(
        self,
        model_name:str,
        model_dtype:str,
        scoring_dtype:str,
        cache_dir:str):

        self.model_name = model_name
        self.model_dtype = "auto" if model_dtype is None else getattr(torch, model_dtype)
        self.scoring_dtype = None if scoring_dtype is None else getattr(torch, scoring_dtype)
        self.cache_dir = cache_dir

        self.tokenizer, self.model = self.load_tokenizer_and_model()

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:

        tokenizer, model = None, None

        if self.model_name == "Qwen/Qwen3-Reranker-0.6B":

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                cache_dir=self.cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self.model_dtype, # default bfloat16
                attn_implementation=None if self.model_dtype is torch.float32 else "flash_attention_2",
                device_map="cuda:0")
            model.eval()

        elif self.model_name == "Qwen/Qwen3-Embedding-0.6B":

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                cache_dir=self.cache_dir)
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self.model_dtype, # default bfloat16
                attn_implementation=None if self.model_dtype is torch.float32 else "flash_attention_2",
                device_map="cuda:0")
            model.eval()

        elif self.model_name == "BAAI/bge-reranker-v2-m3":

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self.model_dtype, # default float32
                device_map="cuda:0")
            model.eval()

        elif self.model_name == "Alibaba-NLP/gte-multilingual-reranker-base":

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                torch_dtype=self.model_dtype, # default float16
                device_map="cuda:0")
            model.eval()

        elif self.model_name == "jinaai/jina-reranker-v2-base-multilingual":

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                torch_dtype=self.model_dtype, # default bfloat16
                device_map="cuda:0")
            
        elif self.model_name == "intfloat/multilingual-e5-large":

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir)
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=self.model_dtype, # default float32
                device_map="cuda:0")

        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        return (tokenizer, model)

    def inference(
        self,
        dataset:Dict[str, List[str] | List[List[str]] | List[List[bool]]],
        eval_batch_size:int=16,
        max_length:int=4096
        ) -> List[List[float]]:

        if self.model_name == "Qwen/Qwen3-Reranker-0.6B":

            token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_token_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            suffix_token_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

            def format_instruction(query:str, passage:str, instruction:str=None) -> str:
                if instruction is None:
                    instruction = "Given a web search query, retrieve relevant passages that answer the query"
                output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {passage}"
                return output

            def process_inputs(prompts:List[str]) -> Dict[str, torch.Tensor]:

                inputs = self.tokenizer(
                    prompts,
                    max_length=max_length - len(prefix_token_ids) - len(suffix_token_ids),
                    truncation="longest_first",
                    padding=False,
                    return_attention_mask=False)

                for i, input_ids in enumerate(inputs["input_ids"]):
                    inputs["input_ids"][i] = prefix_token_ids + input_ids + suffix_token_ids

                inputs = self.tokenizer.pad(
                    inputs,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt")

                for key in inputs:
                    inputs[key] = inputs[key].to(self.model.device)

                return inputs

            @torch.inference_mode()
            def compute_scores(inputs, **kwargs) -> List[float]:
                logits_last = self.model(**inputs).logits[:, -1, :] # (bsz, vsz)
                logits_pair = torch.stack([logits_last[:, token_false_id], logits_last[:, token_true_id]], dim=1) # (bsz, 2)
                probs = F.softmax(logits_pair, dim=1, dtype=self.scoring_dtype) # (bsz, 2)
                return probs[:, 1].tolist()

            probs = []
            for query, candidates in tqdm(
                zip(dataset["query"], dataset["candidates"]),
                desc="inference"):

                probs_batch = []
                for i_batch in range((len(candidates) - 1) // eval_batch_size + 1):
                    cands = candidates[eval_batch_size * i_batch : eval_batch_size * (i_batch + 1)]

                    prompts = [format_instruction(query=query, passage=cand) for cand in cands]
                    inputs = process_inputs(prompts)
                    probs_batch += compute_scores(inputs)

                probs.append(probs_batch)

        elif self.model_name == "Qwen/Qwen3-Embedding-0.6B":

            def last_token_pool(
                last_hidden_states: torch.Tensor,
                attention_mask: torch.Tensor
                ) -> torch.Tensor:

                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[
                        torch.arange(batch_size, device=last_hidden_states.device),
                        sequence_lengths]

            def get_detailed_instruct(task_description: str, query: str) -> str:
                return f"Instruct: {task_description}\nQuery:{query}"

            probs = []
            for query, candidates in tqdm(
                zip(dataset["query"], dataset["candidates"]),
                desc="inference"):

                probs_batch = []
                for i_batch in range((len(candidates) - 1) // eval_batch_size + 1):
                    cands = candidates[eval_batch_size * i_batch : eval_batch_size * (i_batch + 1)]

                    task_desc = "Given a web search query, retrieve relevant passages that answer the query"
                    query = get_detailed_instruct(task_desc, query)

                    input_texts = [query] + cands
                    batch_dict = self.tokenizer(
                        input_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt")
                    batch_dict.to(self.model.device)

                    with torch.inference_mode():
                        outputs = self.model(**batch_dict)
                        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                        scores = (
                            embeddings[0].to(dtype=self.scoring_dtype) \
                            @ embeddings[1:].T.to(dtype=self.scoring_dtype) # no changes if None
                            ).tolist()

                    probs_batch += scores
                
                probs.append(probs_batch)

        elif self.model_name in [
            "BAAI/bge-reranker-v2-m3",
            "Alibaba-NLP/gte-multilingual-reranker-base",
            "jinaai/jina-reranker-v2-base-multilingual",
            ]:

            probs = []
            for query, candidates in tqdm(
                zip(dataset["query"], dataset["candidates"]),
                desc="inference"):

                probs_batch = []
                for i_batch in range((len(candidates) - 1) // eval_batch_size + 1):
                    cands = candidates[eval_batch_size * i_batch : eval_batch_size * (i_batch + 1)]

                    sample_dataset = [(query, cand) for cand in cands]
                    inputs = self.tokenizer(
                        sample_dataset,
                        padding=True,
                        truncation=True,
                        max_length=max_length, # 512
                        return_tensors="pt")
                    for key in inputs:
                        inputs[key] = inputs[key].to(self.model.device)

                    with torch.inference_mode():
                        logits = self.model(**inputs, return_dict=True).logits.view(-1,) # (bsz,)
                        scores = F.sigmoid(logits.to(dtype=self.scoring_dtype)).tolist()
                    probs_batch += scores

                probs.append(probs_batch)

        elif self.model_name == "intfloat/multilingual-e5-large":

            def average_pool(
                last_hidden_states: torch.Tensor,
                attention_mask: torch.Tensor
                ) -> torch.Tensor:
                last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
                return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

            probs = []
            for query, candidates in tqdm(
                zip(dataset["query"], dataset["candidates"]),
                desc="inference"):

                probs_batch = []
                for i_batch in range((len(candidates) - 1) // eval_batch_size + 1):
                    cands = candidates[eval_batch_size * i_batch : eval_batch_size * (i_batch + 1)]

                    input_texts = ["query: " + query]
                    input_texts += ["passage: " + cand for cand in cands]
                    batch_dict = self.tokenizer(
                        input_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt")
                    batch_dict.to(self.model.device)

                    with torch.inference_mode():
                        outputs = self.model(**batch_dict)

                        embeddings = average_pool(
                            outputs.last_hidden_state,
                            batch_dict["attention_mask"])
                        
                        embeddings = F.normalize(embeddings, p=2, dim=1) # normalize embeddings
                        scores = (
                            embeddings[0].to(dtype=self.scoring_dtype) \
                            @ embeddings[1:].T.to(dtype=self.scoring_dtype) # no changes if None
                            ).tolist()
    
                    probs_batch += scores
                
                probs.append(probs_batch)

        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        assert len(dataset["candidates"]) == len(probs) == len(dataset["relevances"])

        return probs