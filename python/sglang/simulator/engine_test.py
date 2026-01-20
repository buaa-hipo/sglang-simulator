import sglang


def main():
    llm = sglang.simulator.srt.entrypoints.engine.EngineSimulator(
        model_path="/data/llama-3-8b-instruct/",
        tp_size=2,
        base_gpu_id=0,
        disable_cuda_graph=True,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
