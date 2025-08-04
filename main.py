from pprint import pprint
from argparse import Namespace, ArgumentParser

from Model import Model
from Evaluator import Evaluator
from DataLoader import DataLoader


def main(args:Namespace):

    loader = DataLoader(cache_dir=args.dataset_cache_dir)
    dataset = loader.load_dataset(args.dataset, bm25_top_n=args.bm25_top_n) # returns dict_of_list

    model = Model(args.model, args.model_dtype, args.scoring_dtype, args.model_cache_dir)
    print(f"# of params = {sum(p.numel() for p in model.model.parameters()):,}")
    probs = model.inference(dataset, args.eval_batch_size, args.max_length)
    print(f"An example of probs:\n{sorted(probs[0], reverse=True)}", end="\n\n")

    evaluator = Evaluator()
    results = evaluator.evaluate(dataset["relevances"], probs, args.metrics, args.k_list)

    output_log = ""
    output_log += f"{str(vars(args))}\n"
    for metric, option2scores in results.items():
        output_log += f"■ {metric}\n"
        output_log += "k\t" + "\t".join([str(k) for k in args.k_list]) + "\n"
        for option, scores in option2scores.items():
            output_log += f"{option}\t"
            for x in scores:
                output_log += f"{x * 100:.3f}\t"
            output_log += "\n"
        output_log += "\n"
    output_log += "# ------------------------------------------------------------\n"
    print(output_log)

    with open("results.log", "a", encoding="utf-8") as af:
        af.write(output_log)


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)

    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B", choices=[
        "Qwen/Qwen3-Reranker-0.6B", # cross-encoder-softmax
        "Qwen/Qwen3-Embedding-0.6B", # bi-encoder-product
        "BAAI/bge-reranker-v2-m3", # cross-encoder-sigmoid # supports 8k default 1k
        "Alibaba-NLP/gte-multilingual-reranker-base", # cross-encoder-sigmoid # supports 8k default 512
        # "jinaai/jina-reranker-v2-base-multilingual", # cross-encoder-sigmoid # supports 1k default 1k
        "intfloat/multilingual-e5-large", # bi-encoder-product # 560M # 512
        ])
    parser.add_argument("--model_dtype", default="bfloat16", choices=[
        "bfloat16", "float16", "float32"]) # auto if None
    parser.add_argument("--scoring_dtype", default=None, choices=[
        "bfloat16", "float16", "float32"]) # the same as dtype if None
    parser.add_argument("--model_cache_dir", default="/data/disk1/.cache/huggingface/hub/")
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--max_length", default=4096, type=int)

    parser.add_argument("--dataset", default="ArguAna", choices=[
        "MIRACLReranking", "MIRACLRetrieval",
        "AskUbuntuDupQuestions", "MindSmallReranking",
        "ArguAna", "ClimateFEVERHardNegatives",
        "CQADupstackGamingRetrieval", "CQADupstackUnixRetrieval",
        "FEVERHardNegatives", "FiQA2018",
        "HotpotQAHardNegatives", "SCIDOCS",
        "Touche2020Retrieval.v3", "TRECCOVID"])
    parser.add_argument("--dataset_cache_dir", default="/data/disk1/.cache/huggingface/datasets/")
    parser.add_argument("--bm25_top_n", default=1000, type=int)
    parser.add_argument("--metrics", nargs="+", type=str,
        default=["ndcg", "mrr", "map", "recall", "precision", "f1"],
        choices=["ndcg", "mrr", "map", "recall", "precision", "f1"])
    parser.add_argument("--k_list", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50, 100])

    args = parser.parse_args()
    pprint(vars(args))
    main(args)